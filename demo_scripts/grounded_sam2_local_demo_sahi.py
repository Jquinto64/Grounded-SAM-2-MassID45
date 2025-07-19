import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import grounding_dino.groundingdino.datasets.transforms as T

"""
Hyper parameters
"""
TEXT_PROMPT = "bug."
IMG_PATH = "/h/jquinto/lifeplan_b_v9_cropped_center/val2017/GUSA7Z.png"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo_sahi")
DUMP_JSON_RESULTS = True

# Sliced inference parameters
WITH_SLICE_INFERENCE = True  # Set to False to use original approach
SLICE_WH = (512, 512)  # Size of each slice (width, height)
OVERLAP_RATIO = (0.6, 0.6)  # Overlap between slices (width, height)

# Class agnostic option
CLASS_AGNOSTIC = False  # Set to True to make predictions class agnostic

# SAM2 batching parameters to avoid OOM
SAM2_BATCH_SIZE = 10  # Process this many boxes at a time with SAM2

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH

# Load the image
image_source, image = load_image(img_path)
h, w, _ = image_source.shape

# Set the SAM2 predictor image
sam2_predictor.set_image(image_source)

# Parse classes from text prompt for reference
classes = [x.strip().lower() for x in TEXT_PROMPT.split('.') if x]

if WITH_SLICE_INFERENCE:
    # Define a callback function for the InferenceSlicer
    def callback(image_slice: np.ndarray) -> sv.Detections:
        # Convert the image slice to the format expected by Grounding DINO
        transform = T.Compose([
            T.RandomResize([1024], max_size=1024),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_pil = Image.fromarray(image_slice)
        image_transformed, _ = transform(image_pil, None)
        
        # Run Grounding DINO on the slice
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image_transformed,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )
        
        # Process the box prompt for the slice
        slice_h, slice_w = image_slice.shape[:2]
        boxes = boxes * torch.Tensor([slice_w, slice_h, slice_w, slice_h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        if input_boxes.size == 0:
            # Return empty detections if no objects are detected
            return sv.Detections.empty()
            
        # Convert to the format expected by InferenceSlicer
        class_ids = np.array(list(range(len(labels))))
        confidences = confidences.detach().cpu().numpy()
        
        return sv.Detections(
            xyxy=input_boxes,
            confidence=confidences,
            class_id=class_ids
        )
    
    # Create the slicer
    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=SLICE_WH,
        overlap_ratio_wh=OVERLAP_RATIO,
        iou_threshold=0.5,
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION
    )
    
    # Run the slicer on the image
    detections = slicer(image_source)
    
    # Extract the results
    input_boxes = detections.xyxy
    confidences = detections.confidence
    class_ids = detections.class_id
    
    # Skip if no objects are detected
    if len(input_boxes) == 0:
        print(f"No objects detected, skipping this image: {IMG_PATH}")
        exit()
    
    if CLASS_AGNOSTIC:
        # In class agnostic mode, all objects have the same class "object"
        class_names = ["object"] * len(class_ids)
        class_ids = np.zeros_like(class_ids)  # All objects have class_id = 0
    else:    
        # Map class IDs to original labels 
        # Since we can't preserve exact labels across slices, we use the original prompt classes
        class_names = []
        for id in class_ids:
            if id < len(classes):
                class_names.append(classes[id])
            else:
                class_names.append(f"object_{id}")
    
else:
    # Original logic
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    if input_boxes.size == 0:
        print(f"No objects detected, skipping this image: {IMG_PATH}")
        exit()
    
    if CLASS_AGNOSTIC:
        # In class agnostic mode, all objects have the same class "object"
        class_ids = np.zeros(len(labels))
        class_names = ["object"] * len(labels)
    else:
        # Set the class IDs and names
        class_ids = np.array(list(range(len(labels))))
        class_names = labels
    
    confidences = confidences.detach().cpu().numpy()

# CUDA optimizations
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Process SAM2 masks in batches to avoid OOM
batch_size = SAM2_BATCH_SIZE
all_masks = []
all_scores = []
all_logits = []

print(f"Processing {len(input_boxes)} detections in batches of {batch_size}...")

for i in range(0, len(input_boxes), batch_size):
    batch_boxes = input_boxes[i:i+batch_size]
    
    # Free memory before predicting
    torch.cuda.empty_cache()
    
    # Predict masks for this batch
    masks_batch, scores_batch, logits_batch = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=batch_boxes,
        multimask_output=False,
    )
    
    # Normalize dimensions - ensure consistent shape for all batches
    if masks_batch.ndim == 4:
        masks_batch = masks_batch.squeeze(1)  # Convert [batch, 1, h, w] to [batch, h, w]
    elif masks_batch.ndim == 2:
        masks_batch = masks_batch[np.newaxis, :, :]  # Convert [h, w] to [1, h, w]
    
    # Similarly normalize scores and logits dimensions
    if scores_batch.ndim == 0:
        scores_batch = np.array([scores_batch])
    if logits_batch.ndim == 2:
        logits_batch = logits_batch[np.newaxis, :, :]
    
    all_masks.append(masks_batch)
    all_scores.append(scores_batch)
    all_logits.append(logits_batch)
    
    print(f"Processed batch {i//batch_size + 1}/{(len(input_boxes) + batch_size - 1)//batch_size}")
    print(f"Batch shape: {masks_batch.shape}")

# Concatenate results
if len(all_masks) > 0:
    try:
        masks = np.concatenate(all_masks, axis=0) if len(all_masks) > 1 else all_masks[0]
        scores = np.concatenate(all_scores, axis=0) if len(all_scores) > 1 else all_scores[0]
        logits = np.concatenate(all_logits, axis=0) if len(all_logits) > 1 else all_logits[0]
    except ValueError as e:
        print(f"Concatenation error: {e}")
        print("Attempting to normalize array dimensions...")
        
        # Debug and fix dimensions
        shapes = [m.shape for m in all_masks]
        print(f"Mask shapes: {shapes}")
        
        # Convert all to 3D [batch, height, width]
        for i in range(len(all_masks)):
            if all_masks[i].ndim == 4:
                all_masks[i] = all_masks[i].squeeze(1)
            elif all_masks[i].ndim == 2:
                all_masks[i] = all_masks[i][np.newaxis, :, :]
        
        # Try concatenation again
        masks = np.concatenate(all_masks, axis=0) if len(all_masks) > 1 else all_masks[0]
        scores = np.concatenate(all_scores, axis=0) if len(all_scores) > 1 else all_scores[0]
        logits = np.concatenate(all_logits, axis=0) if len(all_logits) > 1 else all_logits[0]
else:
    print("No masks were generated.")
    exit()

# No need to post-process masks as we've already normalized them
print(f"Final masks shape: {masks.shape}")

# Prepare labels for visualization
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

# Visualize results
img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes,
    mask=masks.astype(bool),
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

# Save results as JSON if requested
if DUMP_JSON_RESULTS:
    def mask_to_polygons(mask):
        """Convert a binary mask to polygons using OpenCV contours"""
        # Ensure mask is binary and in the right format for OpenCV
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Format contours as COCO polygons
        polygons = []
        for contour in contours:
            # Flatten the contour and convert to list
            contour = contour.flatten().tolist()
            # COCO format requires [x1,y1,x2,y2,...] format
            if len(contour) >= 6:  # At least 3 points (x,y)
                polygons.append(contour)
        
        return polygons
    
    def is_polygon_format(mask):
        """Check if the mask is already in polygon format"""
        # Simple heuristic: if it's a list and each element is a list of coordinates
        return isinstance(mask, list) and all(isinstance(item, list) for item in mask)
    
    def single_mask_to_rle(mask):
        """Convert mask to RLE format for when polygon representation isn't possible/suitable"""
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    
    # Process each mask - convert to polygons if possible, otherwise use RLE
    segmentations = []
    for mask in masks:
        if is_polygon_format(mask):
            # Already in polygon format
            segmentations.append(mask)
        else:
            # Convert binary mask to polygons
            polygons = mask_to_polygons(mask)
            if polygons:  # If we got valid polygons
                segmentations.append(polygons)
            else:
                # Fall back to RLE if polygon conversion fails
                segmentations.append(single_mask_to_rle(mask))
    
    input_boxes_list = input_boxes.tolist()
    scores_list = scores.tolist()
    
    # Save results
    results = {
        "image_path": img_path,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": segmentation,
                "score": score,
            }
            for class_name, box, segmentation, score in zip(class_names, input_boxes_list, segmentations, scores_list)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }
    
    # Add inference parameters to filename
    inference_type = "sliced" if WITH_SLICE_INFERENCE else "regular"
    class_type = "class_agnostic" if CLASS_AGNOSTIC else "class_specific"
    output_filename = f"grounded_sam2_local_{inference_type}_{class_type}_results.json"
    
    with open(os.path.join(OUTPUT_DIR, output_filename), "w") as f:
        json.dump(results, f, indent=4)
        
print(f"Processing complete. Results saved to {OUTPUT_DIR}")
print(f"Inference mode: {'Sliced' if WITH_SLICE_INFERENCE else 'Regular'}")
print(f"Class mode: {'Class-agnostic' if CLASS_AGNOSTIC else 'Class-specific'}")

# import os
# import cv2
# import json
# import torch
# import numpy as np
# import supervision as sv
# import pycocotools.mask as mask_util
# from pathlib import Path
# from torchvision.ops import box_convert
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
# from PIL import Image
# import grounding_dino.groundingdino.datasets.transforms as T

# """
# Hyper parameters
# """
# # TEXT_PROMPT = "bug. insect. ant."
# TEXT_PROMPT = "insect."
# IMG_PATH = "/h/jquinto/lifeplan_b_v9_cropped_center/val2017/GUSA7Z.png"

# # SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
# # SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_small.pt"
# SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
# GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"

# # GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# # GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

# BOX_THRESHOLD = 0.35
# TEXT_THRESHOLD = 0.25
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo_sahi")
# DUMP_JSON_RESULTS = True

# # Sliced inference parameters
# WITH_SLICE_INFERENCE = True  # Set to False to use original approach
# SLICE_WH = (512, 512)  # Size of each slice (width, height)
# OVERLAP_RATIO = (0.6, 0.6)  # Overlap between slices (width, height)

# # create output directory
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # environment settings
# # use bfloat16

# # build SAM2 image predictor
# sam2_checkpoint = SAM2_CHECKPOINT
# model_cfg = SAM2_MODEL_CONFIG
# sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
# sam2_predictor = SAM2ImagePredictor(sam2_model)

# # build grounding dino model
# grounding_model = load_model(
#     model_config_path=GROUNDING_DINO_CONFIG, 
#     model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
#     device=DEVICE
# )

# # setup the input image and text prompt for SAM 2 and Grounding DINO
# # VERY important: text queries need to be lowercased + end with a dot
# text = TEXT_PROMPT
# img_path = IMG_PATH

# # Load the image
# image_source, image = load_image(img_path)
# h, w, _ = image_source.shape

# # Set the SAM2 predictor image
# sam2_predictor.set_image(image_source)

# # Parse classes from text prompt for reference
# classes = [x.strip().lower() for x in TEXT_PROMPT.split('.') if x]

# if WITH_SLICE_INFERENCE:
#     # Define a callback function for the InferenceSlicer
#     def callback(image_slice: np.ndarray) -> sv.Detections:
#         # Convert the image slice to the format expected by Grounding DINO
#         transform = T.Compose([
#             T.RandomResize([1024], max_size=1024),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])
#         image_pil = Image.fromarray(image_slice)
#         image_transformed, _ = transform(image_pil, None)
        
#         # Run Grounding DINO on the slice
#         boxes, confidences, labels = predict(
#             model=grounding_model,
#             image=image_transformed,
#             caption=text,
#             box_threshold=BOX_THRESHOLD,
#             text_threshold=TEXT_THRESHOLD,
#             device=DEVICE
#         )
        
#         # Process the box prompt for the slice
#         slice_h, slice_w = image_slice.shape[:2]
#         boxes = boxes * torch.Tensor([slice_w, slice_h, slice_w, slice_h])
#         input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
#         if input_boxes.size == 0:
#             # Return empty detections if no objects are detected
#             return sv.Detections.empty()
            
#         # Convert to the format expected by InferenceSlicer
#         class_ids = np.array(list(range(len(labels))))
#         confidences = confidences.detach().cpu().numpy()
        
#         return sv.Detections(
#             xyxy=input_boxes,
#             confidence=confidences,
#             class_id=class_ids
#         )
    
#     # Create the slicer
#     slicer = sv.InferenceSlicer(
#         callback=callback,
#         slice_wh=SLICE_WH,
#         overlap_ratio_wh=OVERLAP_RATIO,
#         iou_threshold=0.5,
#         overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION
#     )
    
#     # Run the slicer on the image
#     detections = slicer(image_source)
    
#     # Extract the results
#     input_boxes = detections.xyxy
#     confidences = detections.confidence
#     class_ids = detections.class_id
    
#     # Skip if no objects are detected
#     if len(input_boxes) == 0:
#         print(f"No objects detected, skipping this image: {IMG_PATH}")
#         exit()
        
#     # Map class IDs to original labels 
#     # Since we can't preserve exact labels across slices, we use the original prompt classes
#     class_names = []
#     for id in class_ids:
#         if id < len(classes):
#             class_names.append(classes[id])
#         else:
#             class_names.append(f"object_{id}")
    
# else:
#     # Original logic
#     boxes, confidences, labels = predict(
#         model=grounding_model,
#         image=image,
#         caption=text,
#         box_threshold=BOX_THRESHOLD,
#         text_threshold=TEXT_THRESHOLD,
#     )

#     # process the box prompt for SAM 2
#     boxes = boxes * torch.Tensor([w, h, w, h])
#     input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

#     if input_boxes.size == 0:
#         print(f"No objects detected, skipping this image: {IMG_PATH}")
#         exit()
    
#     # Set the class IDs and names
#     class_ids = np.array(list(range(len(labels))))
#     class_names = labels
#     confidences = confidences.detach().cpu().numpy()

# # CUDA optimizations
# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
#     # turn on tfloat32 for Ampere GPUs
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

# # Predict masks with SAM2
# masks, scores, logits = sam2_predictor.predict(
#     point_coords=None,
#     point_labels=None,
#     box=input_boxes,
#     multimask_output=False,
# )

# # Post-process masks
# if masks.ndim == 4:
#     masks = masks.squeeze(1)

# # Prepare labels for visualization
# labels = [
#     f"{class_name} {confidence:.2f}"
#     for class_name, confidence
#     in zip(class_names, confidences)
# ]

# # Visualize results
# img = cv2.imread(img_path)
# detections = sv.Detections(
#     xyxy=input_boxes,
#     mask=masks.astype(bool),
#     class_id=class_ids
# )

# box_annotator = sv.BoxAnnotator()
# annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

# label_annotator = sv.LabelAnnotator()
# annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
# cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

# mask_annotator = sv.MaskAnnotator()
# annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
# cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

# # Save results as JSON if requested
# if DUMP_JSON_RESULTS:
#     def mask_to_polygons(mask):
#         """Convert a binary mask to polygons using OpenCV contours"""
#         # Ensure mask is binary and in the right format for OpenCV
#         mask_uint8 = mask.astype(np.uint8) * 255
#         contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Format contours as COCO polygons
#         polygons = []
#         for contour in contours:
#             # Flatten the contour and convert to list
#             contour = contour.flatten().tolist()
#             # COCO format requires [x1,y1,x2,y2,...] format
#             if len(contour) >= 6:  # At least 3 points (x,y)
#                 polygons.append(contour)
        
#         return polygons
    
#     def is_polygon_format(mask):
#         """Check if the mask is already in polygon format"""
#         # Simple heuristic: if it's a list and each element is a list of coordinates
#         return isinstance(mask, list) and all(isinstance(item, list) for item in mask)
    
#     def single_mask_to_rle(mask):
#         """Convert mask to RLE format for when polygon representation isn't possible/suitable"""
#         rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
#         rle["counts"] = rle["counts"].decode("utf-8")
#         return rle
    
#     # Process each mask - convert to polygons if possible, otherwise use RLE
#     segmentations = []
#     for mask in masks:
#         if is_polygon_format(mask):
#             # Already in polygon format
#             segmentations.append(mask)
#         else:
#             # Convert binary mask to polygons
#             polygons = mask_to_polygons(mask)
#             if polygons:  # If we got valid polygons
#                 segmentations.append(polygons)
#             else:
#                 # Fall back to RLE if polygon conversion fails
#                 segmentations.append(single_mask_to_rle(mask))
    
#     input_boxes_list = input_boxes.tolist()
#     scores_list = scores.tolist()
    
#     # Save results
#     results = {
#         "image_path": img_path,
#         "annotations" : [
#             {
#                 "class_name": class_name,
#                 "bbox": box,
#                 "segmentation": segmentation,
#                 "score": score,
#             }
#             for class_name, box, segmentation, score in zip(class_names, input_boxes_list, segmentations, scores_list)
#         ],
#         "box_format": "xyxy",
#         "img_width": w,
#         "img_height": h,
#     }
    
#     inference_type = "sliced" if WITH_SLICE_INFERENCE else "regular"
#     output_filename = f"grounded_sam2_local_{inference_type}_results.json"
    
#     with open(os.path.join(OUTPUT_DIR, output_filename), "w") as f:
#         json.dump(results, f, indent=4)
        
# print(f"Processing complete. Results saved to {OUTPUT_DIR}")