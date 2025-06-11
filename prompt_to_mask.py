import torch
from transformers import SamProcessor, SamModel
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from PIL import Image
import numpy as np
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
sam_model = SamModel.from_pretrained("facebook/sam-vit-large").to(device)

image = Image.open("image.jpg").convert("RGB")


w, h = image.size
scale = min(768 / max(h, w), 1.0)
new_h, new_w = int(h * scale), int(w * scale)

prompt = ["car", "person"] 

inputs = dino_processor(
    images=image, 
    text=prompt,
    return_tensors="pt",
    do_resize=True,
    size={"height": new_h, "width": new_w}
).to(device)

with torch.no_grad():
    outputs = dino_model(**inputs)

results = dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.5,
            text_threshold=0.5,
            target_sizes=[(h, w)])

boxes = results[0]['boxes'].cpu().numpy().tolist()
scores = results[0]['scores'].cpu().numpy().tolist()
labels = results[0]['labels']


if len(boxes) > 0:
    sam_inputs = sam_processor(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
    
    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)
    
    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks, sam_inputs["original_sizes"], sam_inputs["reshaped_input_sizes"]
    )
    
    iou_scores = sam_outputs.iou_scores.cpu().numpy()
    final_mask = np.zeros((h, w), dtype=np.float32)
    
    for i in range(len(boxes)):
        detection_masks = masks[0][i].cpu().numpy()
        best_idx = iou_scores[0][i].argmax()
        final_mask += detection_masks[best_idx]
    
    final_mask = np.clip(final_mask, 0, 1)
    print(f"Processed {len(boxes)} detections in single batch")
else:
    final_mask = np.zeros((h, w), dtype=np.float32)

cv2.imwrite("mask.png", (final_mask * 255).astype(np.uint8))


