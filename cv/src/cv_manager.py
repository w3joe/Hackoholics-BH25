from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from typing import Any
import json
import io

from rtdetrv2_tensorrt import TRTInference  # Adjust import to your TRTInference definition

class CVManager:
    def __init__(self):
        self.engine_path = "model.engine"
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = 0.6
        self.iou_threshold = 0.6

        self.model = TRTInference(self.engine_path, device=self.device)

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        im_pil = Image.open(io.BytesIO(image)).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(self.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor()
        ])
        im_data = transforms(im_pil)[None].to(self.device)

        blob = {
            "images": im_data,
            "orig_target_sizes": orig_size
        }

        outputs = self.model(blob)
        labels = outputs["labels"]
        boxes = outputs["boxes"]
        scores = outputs["scores"]

        keep = scores > self.conf_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        from torchvision.ops import nms
        final_boxes = []
        final_labels = []

        for cls in np.unique(labels.cpu().numpy()):
            cls_mask = labels == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            if len(cls_boxes) == 0:
                continue

            nms_indices = nms(cls_boxes, cls_scores, self.iou_threshold)
            final_boxes.append(cls_boxes[nms_indices])
            final_labels.append(labels[cls_mask][nms_indices])

        if final_boxes:
            final_boxes = torch.cat(final_boxes)
            final_labels = torch.cat(final_labels)
        else:
            final_boxes = torch.empty((0, 4))
            final_labels = torch.empty((0,), dtype=torch.long)

        scene_preds = []
        for i in range(len(final_labels)):
            x1, y1, x2, y2 = final_boxes[i].tolist()
            w, h = x2 - x1, y2 - y1
            scene_preds.append({
                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                "category_id": int(final_labels[i])
            })
        # with open("predictions.json", "w") as f:
        #     json.dump(scene_preds, f, indent=2)
        return scene_preds