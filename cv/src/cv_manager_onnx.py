import io
import json
import numpy as np
from PIL import Image
from typing import Any
import torch
import torchvision.transforms as T
import torchvision.ops as ops
import onnxruntime as ort


class CVManager:
    def __init__(self):
        self.sess = ort.InferenceSession("model.onnx")
        self.conf_threshold = 0.6
        self.iou_threshold = 0.5
        self.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

    # def preprocess(self, image: Image.Image):
    #     """Resize and normalize image."""
    #     orig_size = torch.tensor(image.size[::-1])[None]  # (H, W)
    #     tensor_img = self.transforms(image)[None]  # shape: [1, 3, H, W]
    #     return tensor_img, orig_size

    def postprocess(self, labels, boxes, scores):
        """Apply confidence threshold and NMS."""
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        labels = torch.tensor(labels)

        keep = scores > self.conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        if boxes.size(0) == 0:
            return []

        final_preds = []
        for cls in labels.unique():
            cls_mask = labels == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            if cls_scores.size(0) == 0:
                continue

            keep_idxs = ops.nms(cls_boxes, cls_scores, self.iou_threshold)
            for idx in keep_idxs:
                x1, y1, x2, y2 = cls_boxes[idx].tolist()
                final_preds.append({
                    "bbox": [round(x1, 2), round(y1, 2), round(x2-x1, 2), round(y2-y1, 2)],
                    "category_id": int(cls.item())
                })

        return final_preds

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        im_pil = Image.open(io.BytesIO(image)).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None]
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None]

        # ONNX inference
        ort_inputs = {
            'images': im_data.numpy(),
            'orig_target_sizes': orig_size.numpy()
        }
        labels, boxes, scores = self.sess.run(['labels', 'boxes', 'scores'], ort_inputs)

        predictions = self.postprocess(labels, boxes, scores)
        
        return predictions
    
# if __name__ == "__main__":
#     with open("998.jpg", "rb") as f:
#         image_bytes = f.read()
#     manager = CVManager()
#     predictions = manager.cv(image_bytes)
#     print(predictions)