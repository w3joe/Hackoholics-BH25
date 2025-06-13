"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw

from src.core import YAMLConfig

CONFIG_PATH = "./configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml"
MODEL_PATH = "./output/rtdetrv2_r50vd_m_7x_coco/best.pth"
IMAGE_DIR_PATH = "./998.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'results_{i}.jpg')


def main():
    """main
    """
    cfg = YAMLConfig(CONFIG_PATH, resume=MODEL_PATH)

    if MODEL_PATH:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(DEVICE)

    im_pil = Image.open(IMAGE_DIR_PATH).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(DEVICE)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(DEVICE)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    results = {"predictions": []}
    scene_preds = []

    import torchvision.ops as ops

    def filter_and_nms(boxes, labels, scores, threshold=0.6, iou_threshold=0.5):
        keep = scores > threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        final_boxes = []
        final_labels = []

        for cls in labels.unique():
            cls_mask = labels == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            if cls_boxes.size(0) == 0:
                continue

            nms_indices = ops.nms(cls_boxes, cls_scores, iou_threshold)
            final_boxes.append(cls_boxes[nms_indices])
            final_labels.append(labels[cls_mask][nms_indices])

        if final_boxes:
            final_boxes = torch.cat(final_boxes)
            final_labels = torch.cat(final_labels)
        else:
            final_boxes = torch.empty((0, 4))
            final_labels = torch.empty((0,), dtype=torch.long)

        return final_boxes, final_labels
    
    filtered_boxes, filtered_labels = filter_and_nms(boxes[0], labels[0], scores[0])

    for i in range(len(filtered_labels)):
        x1, y1, x2, y2 = filtered_boxes[i].tolist()
        w = x2 - x1
        h = y2 - y1
        scene_preds.append({
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "category_id": int(filtered_labels[i])
        })

    results["predictions"].append(scene_preds)

    import json
    with open("predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    
#     draw([im_pil], labels, boxes, scores)
    
#     print(boxes)


if __name__ == '__main__':
    main()
