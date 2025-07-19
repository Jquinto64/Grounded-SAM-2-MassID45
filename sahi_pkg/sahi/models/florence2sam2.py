# SAHI Integration for Florence-2 + SAM2
# This implementation allows using Florence-2 (Open Vocabulary Detection) with SAM2 in the SAHI framework

import logging
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
from PIL import Image
import grounding_dino.groundingdino.datasets.transforms as T

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

logger = logging.getLogger(__name__)

class Florence2Sam2DetectionModel(DetectionModel):
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
        sam2_checkpoint: Optional[str] = None,
        sam2_config: Optional[str] = None,
        text_prompt: str = "small brown yellow insects",
        task_prompt: str = "<OPEN_VOCABULARY_DETECTION>",
        florence2_model_id: str = "microsoft/Florence-2-large-ft",
    ):
        """
        Initialize Florence2Sam2DetectionModel for object detection and instance segmentation.
        
        Args:
            model_path: str
                Not used for Florence-2, kept for compatibility
            config_path: str
                Not used for Florence-2, kept for compatibility
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
                Input image size for inference
            sam2_checkpoint: str
                Path to the SAM2 model checkpoint
            sam2_config: str
                Path to the SAM2 model config
            text_prompt: str
                Text prompt for Open Vocabulary Detection (e.g., "car <and> person")
                Note: use "<and>" to separate multiple classes
            florence2_model_id: str
                HuggingFace model ID for Florence-2
        """
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self._text_prompt = text_prompt
        self.florence2_model_id = florence2_model_id
        self._task_prompt = task_prompt
        
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
        check_requirements(["torch", "transformers", "numpy", "PIL"])

    def load_model(self):
        """Load both Florence-2 and SAM2 models."""
        from transformers import AutoProcessor, AutoModelForCausalLM
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Check if SAM2 paths are provided
        if not self.sam2_checkpoint or not self.sam2_config:
            raise ValueError("SAM2 checkpoint and config must be provided")
        
        # Set device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set dtype
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load Florence-2 model
        logger.info(f"Loading Florence-2 model from {self.florence2_model_id}")
        florence2_model = AutoModelForCausalLM.from_pretrained(
            self.florence2_model_id, 
            trust_remote_code=True, 
            torch_dtype='auto'
        ).eval().to(self.device)
        
        florence2_processor = AutoProcessor.from_pretrained(
            self.florence2_model_id, 
            trust_remote_code=True
        )
        
        # Load SAM2 model
        logger.info(f"Loading SAM2 model from {self.sam2_checkpoint}")
        sam2_model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        self.set_model({
            "florence2_model": florence2_model,
            "florence2_processor": florence2_processor,
            "sam2_model": sam2_model,
            "sam2_predictor": sam2_predictor
        })

    def set_model(self, model: Dict[str, Any]):
        """
        Set the model from already loaded models.
        
        Args:
            model: Dict[str, Any]
                Dictionary containing florence2_model, florence2_processor, sam2_model, and sam2_predictor
        """
        self.model = model
        
        # Set default category_mapping if not provided
        if self._task_prompt == "<OPEN_VOCABULARY_DETECTION>":
            if self.category_mapping is None:
                # For Florence-2 with Open Vocabulary Detection, categories are determined by the text prompt
                categories = self._text_prompt.split("<and>")
                categories = [cat.strip() for cat in categories]
                self.category_mapping = {str(i): cat for i, cat in enumerate(categories)}
        else:
            self.category_mapping = {}

    def perform_inference(self, image: np.ndarray):
        """
        Perform inference using both Florence-2 and SAM2.
        
        Args:
            image: np.ndarray
                Image in RGB format (HWC)
        """
        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        
        florence2_model = self.model["florence2_model"]
        florence2_processor = self.model["florence2_processor"]
        sam2_predictor = self.model["sam2_predictor"]
        
        # Store original image dimensions
        self.image_height, self.image_width = image.shape[:2]
        
        # Convert numpy array to PIL Image for Florence-2
        if isinstance(image, np.ndarray):
            image_source = Image.fromarray(image).convert("RGB")
        else:
            image_source = Image.open(image).convert("RGB") if isinstance(image, str) else image

        # Set image for SAM2 (expects numpy array in HWC format)
        sam2_predictor.set_image(np.array(image_source))

        # # Prepare image for Grounding DINO
        # transform = T.Compose(
        #     [
        #         T.RandomResize([1024], max_size=1024),
        #         # T.ToTensor(),
        #         # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
        # image_transformed, _ = transform(image_source, None)
        
        # Run Florence-2 inference with Open Vocabulary Detection
        results = self._run_florence2(
            task_prompt=self._task_prompt,
            text_input=self._text_prompt,
            model=florence2_model,
            processor=florence2_processor,
            image=image_source
        )
        
        # Parse Florence-2 results
        results = results[self._task_prompt]
        
        # Skip if no detections
        if "bboxes" not in results or len(results["bboxes"]) == 0:
            self._original_predictions = {
                "boxes": [],
                "confidences": [],
                "labels": [],
                "masks": [],
                "scores": []
            }
            return
            
        # Get boxes and labels
        input_boxes = np.array(results["bboxes"])
        if self._task_prompt == "<OPEN_VOCABULARY_DETECTION>":
            class_names = results["bboxes_labels"]
        else: 
            class_names = results["labels"]

        # Filter out bounding boxes that take up more than 40% of the image area
        filtered_boxes = []
        filtered_class_names = []
        image_area = self.image_height * self.image_width
        area_threshold = 0.4  # 40% of image area
        
        # Also filter out duplicate boxes, if any
        seen = set()
        for i, box in enumerate(input_boxes):
            # Calculate box area [x1, y1, x2, y2]
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            box_area = box_width * box_height
            
            # Calculate ratio of box area to image area
            area_ratio = box_area / image_area
            
            # Keep boxes that take up less than 40% of the image
            if area_ratio <= area_threshold:
                box_tuple = tuple(map(float, box))  # Convert to tuple of floats
                if box_tuple not in seen:
                    seen.add(box_tuple)
                    filtered_boxes.append(box)
                    filtered_class_names.append(class_names[i])
        
        input_boxes = np.array(filtered_boxes)
        class_names = filtered_class_names

         # Skip if no detections
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
        
        # Florence-2 doesn't provide confidence scores, so we'll use a default value
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

    def _run_florence2(self, task_prompt, text_input, model, processor, image):
        """
        Run Florence-2 inference with the given task and text prompt.
        
        Args:
            task_prompt: str
                The task prompt for Florence-2 (e.g., "<OPEN_VOCABULARY_DETECTION>")
            text_input: str
                The text input for the task
            model: Florence-2 model
                The loaded Florence-2 model
            processor: Florence-2 processor
                The loaded Florence-2 processor
            image: PIL.Image
                The input image
                
        Returns:
            dict: The parsed output from Florence-2
        """
        device = model.device
        
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"].to(device),
                pixel_values=inputs["pixel_values"].to(device),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        return parsed_answer

    # def _create_object_prediction_list_from_original_predictions(
    #     self,
    #     shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
    #     full_shape_list: Optional[List[List[int]]] = None,
    # ):
    #     """
    #     Convert the original predictions to a list of ObjectPrediction objects.
        
    #     Args:
    #         shift_amount_list: List[List[int]]
    #             List of [shift_x, shift_y] amounts to shift predictions
    #         full_shape_list: List[List[int]]
    #             List of [height, width] shapes for the full images
    #     """
    #     if not hasattr(self, '_original_predictions') or self._original_predictions is None:
    #         self._object_prediction_list_per_image = [[]]
    #         return
        
    #     # Get the predictions
    #     boxes = self._original_predictions["boxes"]
    #     confidences = self._original_predictions["confidences"]
    #     labels = self._original_predictions["labels"]
    #     masks = self._original_predictions["masks"]
    #     scores = self._original_predictions["scores"]
        
    #     # Skip if no detections
    #     if len(boxes) == 0:
    #         self._object_prediction_list_per_image = [[]]
    #         return
            
    #     # Fix compatibility with older versions of SAHI
    #     if not isinstance(shift_amount_list[0], list):
    #         shift_amount_list = [shift_amount_list]
    #     if full_shape_list is not None and not isinstance(full_shape_list[0], list):
    #         full_shape_list = [full_shape_list]
        
    #     # Only supporting single image for now
    #     shift_amount = shift_amount_list[0]
    #     full_shape = None if full_shape_list is None else full_shape_list[0]
        
    #     # Create object predictions
    #     object_prediction_list = []
        
    #     # Process each detection
    #     for i, (box, confidence, label, mask, score) in enumerate(zip(boxes, confidences, labels, masks, scores)):
    #         # Skip if below confidence threshold
    #         if confidence < self.confidence_threshold:
    #             continue
            
    #         # Convert mask to COCO RLE format
    #         segmentation = get_coco_segmentation_from_bool_mask(mask.astype(bool))
    #         if len(segmentation) == 0:
    #             continue
            
    #         # Fix negative box coords
    #         box[0] = max(0, box[0])
    #         box[1] = max(0, box[1])
    #         box[2] = max(0, box[2])
    #         box[3] = max(0, box[3])
            
    #         # Fix out of image box coords
    #         if full_shape is not None:
    #             box[0] = min(full_shape[1], box[0])
    #             box[1] = min(full_shape[0], box[1])
    #             box[2] = min(full_shape[1], box[2])
    #             box[3] = min(full_shape[0], box[3])
            
    #         # Ignore invalid predictions
    #         if not (box[0] < box[2]) or not (box[1] < box[3]):
    #             logger.warning(f"Ignoring invalid prediction with bbox: {box}")
    #             continue
            
    #         # Use the label as category if present, otherwise default to index
    #         if isinstance(label, str):
    #             category_name = label
    #             # Try to find the category ID
    #             category_id = next((int(k) for k, v in self.category_mapping.items() if v == label), i)
    #         else:
    #             category_id = i
    #             category_name = self.category_mapping.get(str(category_id), f"class_{category_id}")
            
    #         # Create the object prediction
    #         object_prediction = ObjectPrediction(
    #             bbox=box.tolist() if isinstance(box, np.ndarray) else box,
    #             category_id=int(category_id),
    #             score=float(score),
    #             segmentation=segmentation,
    #             category_name=category_name,
    #             shift_amount=shift_amount,
    #             full_shape=full_shape,
    #         )
            
    #         object_prediction_list.append(object_prediction)
        
    #     # Store the predictions
    #     self._object_prediction_list_per_image = [object_prediction_list]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
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
        """Get the text prompt used for Open Vocabulary Detection."""
        return self._text_prompt
    
    @text_prompt.setter
    def text_prompt(self, value: str) -> None:
        """
        Set the text prompt for Open Vocabulary Detection.
        
        Args:
            value: str
                Text prompt for object detection, e.g., "car <and> person"
        """
        self._text_prompt = value
        
        # Update category mapping based on text prompt
        categories = value.split("<and>")
        categories = [cat.strip() for cat in categories]
        self.category_mapping = {str(i): cat for i, cat in enumerate(categories)}

    @property
    def has_mask(self) -> bool:
        """Return whether the model outputs masks (always True for Florence2+SAM2)."""
        return True
    
    @property
    def num_categories(self) -> int:
        """Return the number of categories."""
        return len(self.category_mapping)