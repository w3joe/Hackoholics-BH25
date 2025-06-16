import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging


class OCRManager:
    def __init__(self, timeout_sec: int = 5, max_workers: int = None, binarize: bool = False):
        self.timeout = timeout_sec
        self.max_workers = max_workers or max(multiprocessing.cpu_count() - 1, 1)
        self.binarize = binarize
        self.tesseract_config = '--oem 3 --psm 6' 
        logging.basicConfig(level=logging.WARNING)

    def preprocess(self, image_bytes: bytes) -> np.ndarray | None:
        try:
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            if max(img.shape) > 960:
                scale = 960.0 / max(img.shape)
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

            if self.binarize:
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return img

        except Exception as e:
            logging.warning("Preprocessing failed: %s", e)
            return None

    def tesseract(self, image: np.ndarray) -> str:
        try:
            pil_img = Image.fromarray(image)
            return pytesseract.image_to_string(pil_img, config=self.tesseract_config, timeout=self.timeout).strip()
        except Exception as e:
            logging.warning("OCR failed: %s", e)
            return ""

    def ocr_single(self, image_bytes: bytes) -> str:
        processed = self.preprocess(image_bytes)
        if processed is None:
            return ""
        return self.tesseract(processed)

    def ocr(self, image_bytes: bytes) -> str:
        return self.ocr_single(image_bytes)
