import cv2
import numpy as np
from config import IR_THRESHOLD, IR_MIN_AREA, IR_MAX_AREA


class IRDetector:
    """
    Detects bright IR/white light spots in camera frames.
    Filters out background noise and reflections using thresholding,
    blurring, and contour geometry analysis.
    Supports standard detection and full OpenCV debug inspection.
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
            threshold: Optional threshold override.

        Returns:
            (x, y) tuple of the centroid in camera pixel coordinates, or None.
        """
        pt, _, _, _ = self.detect_debug(frame, threshold=threshold)
        return pt

    def detect_debug(self, frame: np.ndarray, threshold: int = None):
        """
        Extended detection for development/testing mode.
        Generates OpenCV visualization frames with annotated contours,
        crosshairs, binary threshold masks, and detection telemetry.

        Returns:
            point: (cx, cy) tuple or None
            annotated_frame: RGB image with OpenCV contour & crosshair annotations
            thresh_rgb: RGB image of the binary threshold mask
            metadata: dict containing area, num_contours, threshold, and status
        """
        if frame is None:
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            return None, empty, empty, {"status": "NO_FRAME", "area": 0.0, "threshold": self.threshold}

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

        # Convert threshold mask to RGB for Pygame display
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        annotated = frame.copy()

        # Find external contours of bright blobs
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        metadata = {
            "threshold": thresh_val,
            "num_contours": len(contours) if contours else 0,
            "area": 0.0,
            "point": None,
            "status": "SEARCHING"
        }

        if not contours:
            cv2.putText(annotated, "NO BRIGHT SPOT", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 50, 50), 2)
            cv2.putText(thresh_rgb, f"THRESH: {thresh_val}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            return None, annotated, thresh_rgb, metadata

        # Filter contours by acceptable area
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area <= area <= self.max_area:
                valid_contours.append((area, c))

        if not valid_contours:
            cv2.putText(annotated, f"IGNORED {len(contours)} BLOBS (AREA OUT OF RANGE)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 140, 40), 2)
            cv2.putText(thresh_rgb, f"THRESH: {thresh_val}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            metadata["status"] = "AREA_FILTERED"
            return None, annotated, thresh_rgb, metadata

        # Select the largest valid bright spot (strongest IR signal)
        valid_contours.sort(key=lambda item: item[0], reverse=True)
        best_area, best_contour = valid_contours[0]

        # Draw contour boundary in cyan/yellow
        cv2.drawContours(annotated, [best_contour], -1, (255, 230, 0), 2)

        # Calculate centroid using spatial image moments
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None, annotated, thresh_rgb, metadata

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        pt = (cx, cy)
        icx, icy = int(cx), int(cy)

        # Draw green target reticle on annotated camera frame
        cv2.circle(annotated, (icx, icy), 12, (0, 255, 0), 2)
        cv2.circle(annotated, (icx, icy), 3, (0, 255, 0), -1)
        cv2.line(annotated, (icx - 18, icy), (icx + 18, icy), (0, 255, 0), 2)
        cv2.line(annotated, (icx, icy - 18), (icx, icy + 18), (0, 255, 0), 2)

        # Annotation text on camera view
        label = f"IR: ({icx}, {icy}) Area: {best_area:.0f}px"
        cv2.putText(annotated, label, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(thresh_rgb, f"THRESH: {thresh_val} (DETECTED)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Draw small green marker on threshold mask
        cv2.circle(thresh_rgb, (icx, icy), 6, (0, 255, 0), 1)

        metadata["area"] = best_area
        metadata["point"] = pt
        metadata["status"] = "TRACKING"

        return pt, annotated, thresh_rgb, metadata


if __name__ == "__main__":
    print("Testing IRDetector with synthetic image...")
    # Create a 640x480 dark image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a small bright spot at (320, 240)
    cv2.circle(test_img, (320, 240), 6, (255, 255, 255), -1)

    det = IRDetector(threshold=240)
    pt, debug_img, mask_img, meta = det.detect_debug(test_img)
    print(f"Detected point: {pt} (expected ~320, 240)")
    print(f"Metadata: {meta}")
    assert pt is not None and abs(pt[0] - 320) < 2 and abs(pt[1] - 240) < 2, "Test failed!"
    print("IRDetector debug test passed successfully!")
