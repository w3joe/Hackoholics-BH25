This folder contains the necessary training and inference codes for the Computer Vision challenge of TIL-AI 2025.

We utilized the Real-Time DEtection TRansformer implementation for the challenge, and more specifically, we trained the RT-DETRv2-M model on the dataset of 18 categories for 82 epoches.

The training parameters used for the model can be found in rtdetrv2_r50vd_m_7x_coco.yml

To train the model, run the tools/train.py script

To export the best checkpoint to onnx format, run the tools/export_onnx.py script

To export the onnx model to tensorrt engine for optimized inference, run the tools/export_trt.py script
*** It is very important that the architecture and TensorRT version which the tensorrt engine is compiled on, should be the same as the architecture and TensorRT version that it will be deployed on for inference.
There are also some minor code differences between TensorRT 8.x and TensorRT 10.x, hence there are 2 export_trt scripts (1 for 8.x and 1 for 10.x)

The cv_manager.py script defines a CVManager class that streamlines object detection from raw image data. It initializes by setting up the model path, computational device (GPU or CPU), and confidence/IoU thresholds for filtering. The core cv method takes image bytes, preprocesses them by resizing and converting to a tensor, then feeds them to an optimized TRTInference model. The raw detections are then refined by filtering based on confidence and applying Non-Maximum Suppression (NMS) to remove overlapping boxes. Finally, the class returns a clean list of detected objects, each with its bounding box and category ID.
