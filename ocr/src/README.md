# 🧾 OCRManager – Hackoholics BH25 (TIL-AI 2025 OCR Challenge)

This repository contains a lightweight and optimized **OCR pipeline** built specifically for the **TIL-AI 2025 OCR Challenge**.  
The goal is to extract clean text from document-style images, striking a strong balance between **accuracy** and **runtime speed** under constrained environments.

---

## 🛠️ Overview

The core of the system is the `OCRManager` class — a single-image OCR pipeline utilizing:

- 🧠 **Tesseract OCR** (via PyTesseract)
- 🖼️ **OpenCV** and **Pillow** for image decoding and resizing
- ⚙️ Optional grayscale and Otsu binarization (**disabled for best results**)
- ⏱️ Timeout-based handling for unstable or slow inputs

---

## 🔍 Key Decisions & Optimizations

### ✅ Final Configuration

| Component         | Setting                                  | Reason                                         |
|------------------|------------------------------------------|------------------------------------------------|
| Binarization      | ❌ **Disabled**                           | Reduced both speed and accuracy                |
| Image Resizing    | ✅ **Max dimension: 960px**               | Optimal trade-off between accuracy and speed   |
| Inference Mode    | ✅ **Single-image only**                  | Better control on memory and speed             |
| Tesseract Config  | `--oem 3 --psm 6`                         | Best mode for printed block-style text         |

---

## 🧪 PaddleOCR ONNX Experiment

I tested [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR), which wraps PaddleOCR with an ONNX backend.

| OCR Engine         | Accuracy | Speed |
|--------------------|----------|-------|
| Tesseract (Final)  | 0.94     | 0.73  |
| PaddleOCR (ONNX)   | 0.98     | 0.30  |

> Although PaddleOCR ONNX yielded higher accuracy, its runtime was far too slow for our competition constraints.

---

## 🚀 Usage Example

Minimal usage to run OCR on a single image:

```python
from ocr_manager import OCRManager

# Initialize the OCR engine (binarization disabled for optimal accuracy/speed)
ocr = OCRManager(binarize=False)

# Read image and perform OCR
with open("sample.jpg", "rb") as f:
    image_bytes = f.read()

text = ocr.ocr(image_bytes)
print(text)


