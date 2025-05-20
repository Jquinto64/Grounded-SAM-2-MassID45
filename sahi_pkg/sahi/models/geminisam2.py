"""
SAHI Integration for Gemini + SAM2
This implementation allows using Gemini API for bounding box detection with SAM2 for segmentation in the SAHI framework
"""

import os
import json
import logging
import numpy as np
import torch
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
from PIL import Image

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)

class GeminiSam2DetectionModel(DetectionModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        text_prompt: str = "Detect and return bounding boxes for all small brown insects, ants, flies, and gnats in this image",
        sam2_checkpoint: Optional[str] = None,
        sam2_config: Optional[str] = None,
        temperature: float = 0.5,
        **kwargs
    ):
        """
        Initialize GeminiSam2DetectionModel for object detection and instance segmentation.
        
        Args:
            model_path: str
                Not used for Gemini API, kept for compatibility
            config_path: str
                Not used for Gemini API, kept for compatibility
            device: str
                Device to run the model on ("cpu" or "cuda")
            mask_threshold: float
                Threshold value for mask pixels (between 0 and 1)
            confidence_threshold: float
                Confidence threshold for predictions
            category_mapping: dict
                Mapping from category id to category name
            category_remapping: dict
                Remap category ids based on category names
            load_at_init: bool
                Load the model during initialization
            image_size: int
                Input image size for inference (optional)
            api_key: str
                Gemini API key (can also be set via GOOGLE_API_KEY env var)
            model_name: str
                Gemini model name to use
            text_prompt: str
                Text prompt for Gemini API (e.g., "Detect all insects in this image")
            sam2_checkpoint: str
                Path to the SAM2 model checkpoint
            sam2_config: str
                Path to the SAM2 model config
            temperature: float
                Temperature parameter for the Gemini API (controls randomness)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Provide via api_key parameter or GOOGLE_API_KEY environment variable.")
        
        self.model_name = model_name
        self._text_prompt = text_prompt
        self.temperature = temperature
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        
        # Call parent init
        super().__init__(
            model_path=model_path,
            model=model,
            config_path=config_path,
            device=device,
            mask_threshold=mask_threshold,
            confidence_threshold=confidence_threshold,
            category_mapping=category_mapping,
            category_remapping=category_remapping,
            load_at_init=load_at_init,
            image_size=image_size,
        )

    def check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        check_requirements(["google-generativeai", "torch", "numpy", "PIL"])
        
    def load_model(self):
        """
        Load the Gemini API model and SAM2.
        """
        try:
            from google import genai
            from google.genai import types
            import google.genai as genai
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Check if SAM2 paths are provided
            if not self.sam2_checkpoint or not self.sam2_config:
                raise ValueError("SAM2 checkpoint and config must be provided")
            
            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
            # Configure Gemini API client
            client = genai.Client(api_key=self.api_key)
            
            # Load SAM2 model
            logger.info(f"Loading SAM2 model from {self.sam2_checkpoint}")
            sam2_model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
            sam2_predictor = SAM2ImagePredictor(sam2_model)
            
            self.set_model({
                "gemini_client": client,
                "sam2_model": sam2_model,
                "sam2_predictor": sam2_predictor
            })
                
        except ImportError as e:
            raise ImportError(f"Failed to import dependencies: {e}. Please install required packages.")

    def set_model(self, model: Dict[str, Any]):
        """
        Set the model from already loaded models.
        
        Args:
            model: Dict[str, Any]
                Dictionary containing gemini_client, sam2_model, and sam2_predictor
        """
        self.model = model
        
        # Set default category_mapping if not provided
        if self.category_mapping is None:
            self.category_mapping = {"0": "insect"}

    def parse_json(self, json_output):
        """
        Parses JSON from a string, removing potential markdown code block formatting.
        """
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        return json_output
    
    def perform_inference(self, image: np.ndarray):
        """
        Perform inference using Gemini API for bounding boxes and SAM2 for segmentation.
        
        Args:
            image: np.ndarray
                Image in RGB format (HWC)
        """
        from google.genai import types
        
        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        
        gemini_client = self.model["gemini_client"]
        sam2_predictor = self.model["sam2_predictor"]
        
        # Store original image dimensions
        self.image_height, self.image_width = image.shape[:2]
        
        # Convert numpy array to PIL Image for Gemini
        if isinstance(image, np.ndarray):
            image_source = Image.fromarray(image).convert("RGB")
        else:
            image_source = Image.open(image).convert("RGB") if isinstance(image, str) else image
        
        # Set image for SAM2 (expects numpy array in HWC format)
        sam2_predictor.set_image(np.array(image_source))
        
        # Generate content with Gemini API
        try:
            # More specific system instruction to limit output size
            system_instruction = """
                Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 50 objects. 
                Never repeat or duplicate bounding boxes. If an object is present multiple times, return the same label for each instance.
            """
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
            ]

            response = gemini_client.models.generate_content(
                model=self.model_name,
                contents=[self._text_prompt, image_source],
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.temperature,
                    safety_settings=safety_settings,
                )
            )
            
            # Parse response and fix it if it's truncated
            json_text = self.parse_json(response.text)
            
            # Check if the JSON ends properly with a closing bracket
            if not json_text.rstrip().endswith("]"):
                logger.warning("JSON response appears to be truncated, attempting to fix...")
                json_text = self.fix_truncated_json(json_text)
            
            # Parse the JSON
            try:
                boxes = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Response text: {response.text}")
                # Attempt more aggressive fixing of the JSON
                json_text = self.fix_truncated_json(json_text, aggressive=True)
                try:
                    boxes = json.loads(json_text)
                except json.JSONDecodeError:
                    logger.error("Failed to fix JSON response")
                    boxes = []
            
            # Skip if no detections
            if len(boxes) == 0:
                self._original_predictions = {
                    "boxes": [],
                    "confidences": [],
                    "labels": [],
                    "masks": [],
                    "scores": []
                }
                return
            
            # Remove duplicate bounding boxes (Gemini sometimes returns many duplicates)
            boxes = self.remove_duplicate_boxes(boxes)
            
            # Extract bounding boxes
            input_boxes = []
            class_names = []
            
            # Filter by area threshold
            area_threshold = 0.4  # 40% of image area
            image_area = self.image_width * self.image_height
            
            for i, item in enumerate(boxes):
                if "box_2d" in item:
                    # Format: [ymin, xmin, ymax, xmax]
                    y_min, x_min, y_max, x_max = item["box_2d"]
                    
                    # Scale normalized coordinates (0-1000) to absolute image coordinates
                    x_min_abs = int(x_min * self.image_width / 1000)
                    y_min_abs = int(y_min * self.image_height / 1000)
                    x_max_abs = int(x_max * self.image_width / 1000)
                    y_max_abs = int(y_max * self.image_height / 1000)

                    # check if box > 0.4 of image area
                    box_width = abs(x_max_abs - x_min_abs)
                    box_height = abs(y_max_abs - y_min_abs)
                    box_area = box_width * box_height
                    area_ratio = box_area/image_area
                    
                    # Only accept bounding boxes that are below this area threshold
                    if area_ratio < area_threshold:
                        input_boxes.append([x_min_abs, y_min_abs, x_max_abs, y_max_abs])
                        
                        # Get label if available
                        if "label" in item:
                            class_names.append(item["label"])
                        else:
                            class_names.append("insect")
            
            # Convert to numpy array
            input_boxes = np.array(input_boxes)
            
            # Skip if no valid boxes
            if len(input_boxes) == 0:
                self._original_predictions = {
                    "boxes": [],
                    "confidences": [],
                    "labels": [],
                    "masks": [],
                    "scores": []
                }
                return
            
            # Run SAM2 inference
            masks, scores, sam_logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            # Ensure masks have the right shape
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            
            # Gemini doesn't provide confidence scores, so we'll use a default value
            # or the SAM2 scores as a proxy
            confidences = np.ones(len(input_boxes)) * 0.9  # Default high confidence
            
            # Store the results for later processing
            self._original_predictions = {
                "boxes": input_boxes,
                "confidences": confidences,
                "labels": class_names,
                "masks": masks,
                "scores": scores
            }  
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            self._original_predictions = {
                "boxes": [],
                "confidences": [],
                "labels": [],
                "masks": [],
                "scores": []
            }

    def fix_truncated_json(self, json_str, aggressive=False):
        """
        Fix a truncated JSON string by finding the last complete object and adding closing brackets.
        
        Args:
            json_str: str
                The truncated JSON string
            aggressive: bool
                If True, use a more aggressive approach to fix the JSON
                
        Returns:
            str: The fixed JSON string
        """
        # If it's already valid, return as is
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
        
        if aggressive:
            # Find the last complete object by looking for the last occurrence of "},"
            last_obj_end = json_str.rfind("},")
            if last_obj_end > 0:
                # Keep everything up to and including this object, then add closing bracket
                return json_str[:last_obj_end+1] + "]"
        else:
            # Find the position of the last complete object
            last_obj_end = json_str.rfind("}")
            if last_obj_end > 0:
                # Check if there's a comma after the last complete object
                if json_str[last_obj_end+1:].strip().startswith(","):
                    # Cut off at the last complete object
                    return json_str[:last_obj_end+1] + "]"
                else:
                    # The last object is the final one, just add the closing bracket
                    return json_str[:last_obj_end+1] + "]"
        
        # If we can't find a good place to truncate, return an empty array
        return "[]"

    def remove_duplicate_boxes(self, boxes):
        """
        Remove duplicate bounding boxes from the list.
        
        Args:
            boxes: list
                List of box dictionaries
                
        Returns:
            list: Deduplicated list of boxes
        """
        # Use a set to track seen boxes
        seen = set()
        unique_boxes = []
        
        for box in boxes:
            if "box_2d" in box:
                # Convert box coordinates to tuple for hashing
                box_tuple = tuple(box["box_2d"])
                if box_tuple not in seen:
                    seen.add(box_tuple)
                    unique_boxes.append(box)
        
        return unique_boxes

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        Convert the original predictions to a list of ObjectPrediction objects.
        
        Args:
            shift_amount_list: List[List[int]]
                List of [shift_x, shift_y] amounts to shift predictions
            full_shape_list: List[List[int]]
                List of [height, width] shapes for the full images
        """
        # Import required functions
        from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
        
        if not hasattr(self, '_original_predictions') or self._original_predictions is None:
            self._object_prediction_list_per_image = [[]]
            return
        
        original_predictions = self._original_predictions
        
        # Fix compatibility with older versions of SAHI
        if not isinstance(shift_amount_list[0], list):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and not isinstance(full_shape_list[0], list):
            full_shape_list = [full_shape_list]
        
        # Only supporting single image for now
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]
        
        # Parse boxes, masks, scores, category_ids from predictions
        boxes = original_predictions["boxes"]
        confidences = original_predictions["confidences"]
        labels = original_predictions["labels"]
        masks = original_predictions.get("masks", None)
        scores = original_predictions["scores"]
        
        # Skip if no detections
        if len(boxes) == 0:
            self._object_prediction_list_per_image = [[]]
            return
            
        # Create object predictions
        object_prediction_list = []
        
        # Process each detection
        for i, (box, confidence, label, mask, score) in enumerate(zip(boxes, confidences, labels, masks, scores)):
            # Skip if below confidence threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Skip if mask doesn't produce a valid bbox or segmentation
            mask_array = mask.astype(bool)
            if (get_bbox_from_bool_mask(mask_array) is None or 
                not ((segmentation := get_coco_segmentation_from_bool_mask(mask_array)) and len(segmentation) > 0)):
                continue
            
            # Create a copy of the box to avoid modifying the original
            box_fixed = box.copy() if isinstance(box, np.ndarray) else box.copy() if hasattr(box, 'copy') else list(box)
            
            # Fix negative box coords
            box_fixed[0] = max(0, box_fixed[0])
            box_fixed[1] = max(0, box_fixed[1])
            box_fixed[2] = max(0, box_fixed[2])
            box_fixed[3] = max(0, box_fixed[3])
            
            # Fix out of image box coords
            if full_shape is not None:
                box_fixed[0] = min(full_shape[1], box_fixed[0])
                box_fixed[1] = min(full_shape[0], box_fixed[1])
                box_fixed[2] = min(full_shape[1], box_fixed[2])
                box_fixed[3] = min(full_shape[0], box_fixed[3])
            
            # Ignore invalid predictions
            if not (box_fixed[0] < box_fixed[2]) or not (box_fixed[1] < box_fixed[3]):
                logger.warning(f"Ignoring invalid prediction with bbox: {box_fixed}")
                continue
            
            # Handle category name
            if isinstance(label, str):
                category_name = label
                # Try to find the category ID
                category_id = next((int(k) for k, v in self.category_mapping.items() if v == label), i)
            else:
                category_id = int(label) if not isinstance(label, int) else label
                category_name = self.category_mapping.get(str(category_id), f"class_{category_id}")
            
            # Create the object prediction
            object_prediction = ObjectPrediction(
                bbox=box_fixed.tolist() if isinstance(box_fixed, np.ndarray) else box_fixed,
                category_id=int(category_id),
                score=float(score),
                segmentation=segmentation,  # Use segmentation created with walrus operator
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            
            object_prediction_list.append(object_prediction)
        
        # Store the predictions
        self._object_prediction_list_per_image = [object_prediction_list]

    @property
    def text_prompt(self) -> str:
        """Get the text prompt used for Gemini."""
        return self._text_prompt
    
    @text_prompt.setter
    def text_prompt(self, value: str) -> None:
        """
        Set the text prompt for Gemini.
        
        Args:
            value: str
                Text prompt for object detection
        """
        self._text_prompt = value

    @property
    def has_mask(self) -> bool:
        """Return whether the model outputs masks (always True for Gemini+SAM2)."""
        return True
    
    @property
    def num_categories(self) -> int:
        """Return the number of categories."""
        return len(self.category_mapping)