# SAHI Integration for Grounded-SAM2
# This implementation allows using Grounding DINO + SAM2 with the SAHI framework

import logging
import torch
from torchvision.ops import box_convert
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

logger = logging.getLogger(__name__)

class GroundedSam2DetectionModel(DetectionModel):
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
        text_prompt: str = "bug.",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        """
        Initialize GroundedSam2DetectionModel for object detection and instance segmentation.
        
        Args:
            model_path: str
                Path to the Grounding DINO model weights
            config_path: str
                Path to the Grounding DINO model config file
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
                Text prompt for Grounding DINO (e.g., "person. car. dog.")
            box_threshold: float
                Box confidence threshold for Grounding DINO
            text_threshold: float
                Text confidence threshold for Grounding DINO
        """
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self._text_prompt = text_prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        
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
        check_requirements(["torch", "numpy", "supervision"])

    def load_model(self):
        """Load both Grounding DINO and SAM2 models."""
        from grounding_dino.groundingdino.util.inference import load_model
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Check if SAM2 paths are provided
        if not self.sam2_checkpoint or not self.sam2_config:
            raise ValueError("SAM2 checkpoint and config must be provided")
        
        # Load Grounding DINO model
        grounding_model = load_model(
            model_config_path=self.config_path, 
            model_checkpoint_path=self.model_path,
            device=self.device
        )
        
        # Load SAM2 model
        sam2_model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        self.set_model({
            "grounding_model": grounding_model,
            "sam2_model": sam2_model,
            "sam2_predictor": sam2_predictor
        })
        
        # Set category mapping based on text prompt if not provided
        if self.category_mapping is None:
            categories = [cat.strip().rstrip('.') for cat in self._text_prompt.split('.') if cat.strip()]
            self.category_mapping = {str(i): cat for i, cat in enumerate(categories)}

    def set_model(self, model: Dict[str, Any]):
        """
        Set the model from already loaded models.
        
        Args:
            model: Dict[str, Any]
                Dictionary containing grounding_model, sam2_model, and sam2_predictor
        """
        self.model = model
        
        # Set default category_mapping if not provided
        if self.category_mapping is None:
            # For Grounding DINO, categories are determined by the text prompt
            categories = [cat.strip().rstrip('.') for cat in self._text_prompt.split('.') if cat.strip()]
            self.category_mapping = {str(i): cat for i, cat in enumerate(categories)}

    def perform_inference(self, image: np.ndarray):
        """
        Perform inference using both Grounding DINO and SAM2.
        
        Args:
            image: np.ndarray
                Image in RGB format (HWC)
        """
        import torch
        from PIL import Image
        import grounding_dino.groundingdino.datasets.transforms as T
        from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap
        
        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        
        grounding_model = self.model["grounding_model"]
        sam2_predictor = self.model["sam2_predictor"]
        
        # Store original image dimensions
        self.image_height, self.image_width = image.shape[:2]
        
        # Convert numpy array to PIL Image if it's not already
        if isinstance(image, np.ndarray):
            image_source = Image.fromarray(image).convert("RGB")
        else:
            # This shouldn't normally happen, but just in case
            image_source = Image.open(image).convert("RGB") if isinstance(image, str) else image
        
        # Set image for SAM2
        sam2_predictor.set_image(np.array(image_source))
        
        # Prepare image for Grounding DINO
        transform = T.Compose(
            [
                T.RandomResize([1024], max_size=1024),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image_source, None)
        
        # Run Grounding DINO inference
        text_prompt = self._text_prompt.lower().strip()
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'
                
        grounding_model = grounding_model.to(self.device)
        image_transformed = image_transformed.to(self.device)
        
        with torch.no_grad():
            outputs = grounding_model(image_transformed[None], captions=[text_prompt])
        
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter predictions based on confidence
        box_scores = prediction_logits.max(dim=1)[0]
        mask = box_scores > self._box_threshold # CHANGE BACK TO DEFAULT BOX THRESH
        # mask = box_scores > self.confidence_threshold
        logits = prediction_logits[mask]  # (n, 256)
        boxes = prediction_boxes[mask]  # (n, 4)
        confidences = box_scores[mask].numpy()  # Store confidence scores
        
        # Get phrases using Grounding DINO's utility function
        tokenizer = grounding_model.tokenizer
        tokenized = tokenizer(text_prompt)
        
        # Use Grounding DINO's get_phrases_from_posmap function
        phrases = []
        for logit in logits:
            phrase = get_phrases_from_posmap(
                logit > self._text_threshold, 
                tokenized, 
                tokenizer
            )
            # Remove trailing period if present
            if phrase.endswith('.'):
                phrase = phrase[:-1]
            phrases.append(phrase)
        
        # Skip further processing if no boxes detected
        if len(boxes) == 0:
            self._original_predictions = {
                "boxes": [],
                "confidences": [],
                "labels": [],
                "masks": [],
                "scores": []
            }
            return
        
        # Process box prompts for SAM2
        h, w = np.array(image_source).shape[:2]
        boxes_scaled = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
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
        
        # Store the results for later processing
        # Use the confidence scores we already computed from Grounding DINO
        self._original_predictions = {
            "boxes": input_boxes,
            "confidences": confidences,
            "labels": phrases,
            "masks": masks,
            "scores": scores
        }

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
        if not hasattr(self, '_original_predictions') or self._original_predictions is None:
            self._object_prediction_list_per_image = [[]]
            return
        
        # Get the predictions
        boxes = self._original_predictions["boxes"]
        confidences = self._original_predictions["confidences"]
        labels = self._original_predictions["labels"]
        masks = self._original_predictions["masks"]
        scores = self._original_predictions["scores"]
        
        # Skip if no detections
        if len(boxes) == 0:
            self._object_prediction_list_per_image = [[]]
            return
        
        # Fix compatibility with older versions
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        
        # Only supporting single image for now
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]
        
        # Create object predictions
        object_prediction_list = []
        
        # Process each detection
        for i, (box, confidence, label, mask, score) in enumerate(zip(boxes, confidences, labels, masks, scores)):
            # Skip if below confidence threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Convert mask to COCO RLE format
            segmentation = get_coco_segmentation_from_bool_mask(mask.astype(bool))
            if len(segmentation) == 0:
                continue
            
            # Fix negative box coords
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = max(0, box[2])
            box[3] = max(0, box[3])
            
            # Fix out of image box coords
            if full_shape is not None:
                box[0] = min(full_shape[1], box[0])
                box[1] = min(full_shape[0], box[1])
                box[2] = min(full_shape[1], box[2])
                box[3] = min(full_shape[0], box[3])
            
            # Ignore invalid predictions
            if not (box[0] < box[2]) or not (box[1] < box[3]):
                logger.warning(f"Ignoring invalid prediction with bbox: {box}")
                continue
            
            # Use the label as category if present, otherwise default to index
            if isinstance(label, str):
                category_name = label
                # Try to find the category ID
                category_id = next((int(k) for k, v in self.category_mapping.items() if v == label), i)
            else:
                category_id = i
                category_name = self.category_mapping.get(str(category_id), f"class_{category_id}")
            
            # Create the object prediction
            object_prediction = ObjectPrediction(
                bbox=box.tolist(),
                category_id=int(category_id),
                score=float(score),
                segmentation=segmentation,
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            
            object_prediction_list.append(object_prediction)
        
        # Store the predictions
        self._object_prediction_list_per_image = [object_prediction_list]

    @property
    def text_prompt(self) -> str:
        """Get the text prompt used for Grounding DINO."""
        return self._text_prompt
    
    @text_prompt.setter
    def text_prompt(self, value: str) -> None:
        """
        Set the text prompt for Grounding DINO.
        
        Args:
            value: str
                Text prompt for object detection, e.g., "person. car. dog."
        """
        self._text_prompt = value
        
        # Update category mapping based on text prompt
        categories = [cat.strip().rstrip('.') for cat in value.split('.') if cat.strip()]
        self.category_mapping = {str(i): cat for i, cat in enumerate(categories)}
    
    @property
    def box_threshold(self) -> float:
        """Get the box confidence threshold for Grounding DINO."""
        return self._box_threshold
    
    @box_threshold.setter
    def box_threshold(self, value: float) -> None:
        """
        Set the box confidence threshold for Grounding DINO.
        
        Args:
            value: float
                Box confidence threshold, e.g., 0.35
        """
        self._box_threshold = value
    
    @property
    def text_threshold(self) -> float:
        """Get the text confidence threshold for Grounding DINO."""
        return self._text_threshold
    
    @text_threshold.setter
    def text_threshold(self, value: float) -> None:
        """
        Set the text confidence threshold for Grounding DINO.
        
        Args:
            value: float
                Text confidence threshold, e.g., 0.25
        """
        self._text_threshold = value

    @property
    def has_mask(self) -> bool:
        """Return whether the model outputs masks (always True for Grounded SAM2)."""
        return True
    
    @property
    def num_categories(self) -> int:
        """Return the number of categories."""
        return len(self.category_mapping)