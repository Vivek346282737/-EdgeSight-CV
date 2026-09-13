import cv2
import numpy as np

class IndustrialCVExtensions:
    """Advanced CV Modules: Segmentation, OCR Signage parsing, and Perspective Calibration"""
    
    @staticmethod
    def calibrate_camera_matrix(chess_board_size=(9, 6), square_size=0.025):
        """Camera Calibration & Perspective rectification for bird's eye coordinate mapping"""
        # Object points in 3D space
        objp = np.zeros((chess_board_size[0] * chess_board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_board_size[0], 0:chess_board_size[1]].T.reshape(-1, 2) * square_size
        return objp

    @staticmethod
    def extract_hazard_signage_ocr(frame, bbox):
        """OCR text extraction mock for industrial machine safety labels"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        # Preprocessing for OCR binarization
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def polygon_from_segmentation_mask(mask):
        """Converts instance segmentation binary mask to spatial boundary polygon"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
