import cv2
import numpy as np
from config import IR_THRESHOLD, IR_MIN_AREA, IR_MAX_AREA


class IRDetector:
    """
    Detects bright IR/white light spots in camera frames.
    Filters out background noise and reflections using thresholding,
    blurring, and contour geometry analysis.
    """
    def __init__(self, threshold=IR_THRESHOLD, min_area=IR_MIN_AREA, max_area=IR_MAX_AREA):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame: np.ndarray, threshold: int = None):
        """
        Locate the centroid of the brightest spot in the frame.

        Args:
            frame: RGB numpy array (H, W, 3) or single-channel Grayscale (H, W).
            threshold: Optional threshold override (e.g. during calibration).

        Returns:
            (x, y) tuple of the centroid in camera pixel coordinates, or None.
        """
        if frame is None:
            return None

        # Convert to single-channel grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Apply slight Gaussian blur to suppress camera sensor shot noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold for intensely bright pixels
        thresh_val = threshold if threshold is not None else self.threshold
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        # Find external contours of bright blobs
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filter contours by acceptable area
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area <= area <= self.max_area:
                valid_contours.append((area, c))

        if not valid_contours:
            return None

        # Select the largest valid bright spot (strongest IR signal)
        valid_contours.sort(key=lambda item: item[0], reverse=True)
        best_contour = valid_contours[0][1]

        # Calculate centroid using spatial image moments
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        return (cx, cy)


if __name__ == "__main__":
    print("Testing IRDetector with synthetic image...")
    # Create a 640x480 dark image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a small bright spot at (320, 240)
    cv2.circle(test_img, (320, 240), 6, (255, 255, 255), -1)

    det = IRDetector(threshold=240)
    pt = det.detect(test_img)
    print(f"Detected point: {pt} (expected ~320, 240)")
    assert pt is not None and abs(pt[0] - 320) < 2 and abs(pt[1] - 240) < 2, "Test failed!"
    print("IRDetector synthetic test passed successfully!")
