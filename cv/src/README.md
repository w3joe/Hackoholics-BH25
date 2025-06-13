# Overview

- This folder contains the necessary training and inference codes for the Computer Vision challenge of TIL-AI 2025.
- We utilized the Real-Time DEtection TRansformer implementation and trained the RT-DETRv2-M model with the default hyperparameters (found in configs folder) on the dataset of 18 categories for 82 epoches. We also used the default augmentations for the initial training, which includes Random Horizontal Flip, Random Photometric Distort, Random Zoom Out.
- For fine-tuning, the learning rate of the model was reduced from 0.0001 to 0.00005, and we introduced an additional augmentation of Random Gaussian Noise. Other hyperparameters remained the same. We then resumed training from the best checkpoint for an additional 20 epoches.

# Usage

- To train the RT-DETRv2-M model, run the tools/train.py script
- To export the best checkpoint to onnx format, run the tools/export_onnx.py script
- To export the onnx model to tensorrt engine for optimized inference, run the tools/export_trt.py script
  * It is very important that the architecture and TensorRT version which the tensorrt engine is compiled on, should be the same as the architecture and TensorRT version that it will be deployed on for inference. There are also some minor code differences between TensorRT 8.x and TensorRT 10.x, hence there are 2 export_trt scripts (1 for 8.x and 1 for 10.x). If the onnx model was used instead of the TensorRT model, the inference would be significantly slower.

# Inference

The cv_manager.py script defines a CVManager class that streamlines object detection from raw image data. It initializes by setting up the model path, computational device (GPU or CPU), and confidence/IoU thresholds for filtering. The core cv method takes image bytes, preprocesses them by resizing and converting to a tensor, then feeds them to an optimized TRTInference model. The raw detections are then refined by filtering based on confidence and applying Non-Maximum Suppression (NMS) to remove overlapping boxes. Finally, the class returns a clean list of detected objects, each with its bounding box and category ID.

# Results

For qualifiers, we trained the YOLOv8 model and the best result achieved was a score of 0.531 with a speed of 0.954 (mAP of 0.84 for internal test).
After qualifiers, we then worked on the RT-DETRv2-M model which achieved a better performance. The best result achieved was a score of 0.619 with a speed of 0.955 (mAP of 0.97 for internal test).
