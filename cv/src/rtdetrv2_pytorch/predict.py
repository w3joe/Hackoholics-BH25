import torch
from PIL import Image
import torchvision.transforms as T
from src.models.modeling import build_model
from src.config import get_config
import yaml
import os

# Load config
cfg_file = './configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml'
with open(cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)

cfg = get_config(cfg)

# Build model
model = build_model(cfg)
checkpoint = torch.load('./output/rtdetrv2_r50vd_m_7x_coco/best.pth', map_location='cuda')
model.load_state_dict(checkpoint['model'])
model.cuda().eval()

# Preprocessing
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
])

# Load and transform image
img_path = './998.jpg'
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).cuda()

# Run inference
with torch.no_grad():
    outputs = model(input_tensor)

# Postprocess results
# Use cfg.POSTPROCESS and model.decode_outputs if defined
# You will likely need to threshold results, apply NMS, and convert outputs to (x, y, w, h, category_id)

print("Predictions:", outputs)
