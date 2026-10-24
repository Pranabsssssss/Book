import cv2
import numpy as np
from config import (
    IR_THRESHOLD, IR_MIN_AREA, IR_MAX_AREA,
    CAMERA_EDGE_MARGIN_TOP, CAMERA_EDGE_MARGIN_BOTTOM, CAMERA_EDGE_MARGIN_SIDE,
    CAMERA_WIDTH, CAMERA_HEIGHT
)


class IRDetector:
    """
    Precision IR and bright light spot detector optimized for IR pens and remote LEDs.
    Rejects ambient ceiling glare, bezel reflections, and window lights using
    edge-exclusion zones, compactness scoring, and peak-brightness evaluation.
    """
    def __init__(self, threshold=IR_THRESHOLD, min_area=IR_MIN_AREA, max_area=IR_MAX_AREA):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame: np.ndarray, threshold: int = None):
        """
        Locate the centroid of the IR pen / remote LED in the camera frame.
        Returns:
            (x, y) tuple in camera pixel coordinates, or None.
        """
        pt, _, _, _ = self.detect_debug(frame, threshold=threshold)
        return pt

    def detect_debug(self, frame: np.ndarray, threshold: int = None):
        """
        Comprehensive detection with OpenCV visualization and telemetry.
        Returns:
            point: (cx, cy) tuple or None
            annotated_frame: RGB image with active target reticle and glare indicators
            thresh_rgb: RGB image of the binary threshold mask
            metadata: dict containing area, peak brightness, status, etc.
        """
        if frame is None:
            empty = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            return None, empty, empty, {"status": "NO_FRAME", "area": 0.0, "threshold": self.threshold}

        h, w = frame.shape[:2]

        # Convert to single-channel grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Use 3x3 blur to preserve small pinpoint LED cores
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Threshold for intensely bright pixels
        thresh_val = threshold if threshold is not None else self.threshold
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        annotated = frame.copy()

        # Draw subtle edge-exclusion boundary (top ceiling margin & borders)
        cv2.rectangle(
            annotated,
            (CAMERA_EDGE_MARGIN_SIDE, CAMERA_EDGE_MARGIN_TOP),
            (w - CAMERA_EDGE_MARGIN_SIDE, h - CAMERA_EDGE_MARGIN_BOTTOM),
            (60, 60, 90), 1
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        metadata = {
            "threshold": thresh_val,
            "num_contours": len(contours) if contours else 0,
            "area": 0.0,
            "peak_brightness": 0,
            "point": None,
            "status": "SEARCHING"
        }

        if not contours:
            cv2.putText(annotated, "NO LIGHT DETECTED", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 60, 60), 2)
            cv2.putText(thresh_rgb, f"THRESH: {thresh_val}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            return None, annotated, thresh_rgb, metadata

        scored_candidates = []

        for c in contours:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            # 1. Edge Exclusion Filter: discard blobs on the top ceiling edge or outer borders
            if (cy < CAMERA_EDGE_MARGIN_TOP or
                cy > (h - CAMERA_EDGE_MARGIN_BOTTOM) or
                cx < CAMERA_EDGE_MARGIN_SIDE or
                cx > (w - CAMERA_EDGE_MARGIN_SIDE)):
                # Mark as rejected border glare
                cv2.drawContours(annotated, [c], -1, (220, 50, 50), 1)
                continue

            # 2. Area Filter: reject oversized ambient fixtures or tiny noise
            if not (self.min_area <= area <= self.max_area):
                cv2.drawContours(annotated, [c], -1, (200, 120, 40), 1)
                continue

            # 3. Peak Brightness in contour region
            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            roi_gray = gray[y_b:y_b + h_b, x_b:x_b + w_b]
            peak_val = int(np.max(roi_gray)) if roi_gray.size > 0 else 0

            # 4. Circularity / Compactness Score
            peri = cv2.arcLength(c, True)
            circularity = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0.0

            # Composite Score: favors bright saturated core and compact circular shape
            # (Remote LEDs and IR pen tips are compact; elongated strips of ceiling glare are penalized)
            score = (peak_val * 2.0) + (circularity * 60.0) - (area * 0.2)

            scored_candidates.append({
                "score": score,
                "point": (cx, cy),
                "contour": c,
                "area": area,
                "peak": peak_val,
                "circularity": circularity
            })

        if not scored_candidates:
            cv2.putText(annotated, "GLARE / AMBIENT REJECTED", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 140, 40), 2)
            cv2.putText(thresh_rgb, f"THRESH: {thresh_val} (NO VALID LED)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
            metadata["status"] = "GLARE_REJECTED"
            return None, annotated, thresh_rgb, metadata

        # Pick the highest-scoring candidate (the actual IR LED / remote)
        scored_candidates.sort(key=lambda item: item["score"], reverse=True)
        best = scored_candidates[0]

        best_pt = best["point"]
        best_contour = best["contour"]
        icx, icy = int(best_pt[0]), int(best_pt[1])

        # Draw contour boundary in bright yellow
        cv2.drawContours(annotated, [best_contour], -1, (255, 230, 0), 2)

        # Draw precision target reticle in green
        cv2.circle(annotated, (icx, icy), 14, (0, 255, 0), 2)
        cv2.circle(annotated, (icx, icy), 3, (0, 255, 0), -1)
        cv2.line(annotated, (icx - 20, icy), (icx + 20, icy), (0, 255, 0), 2)
        cv2.line(annotated, (icx, icy - 20), (icx, icy + 20), (0, 255, 0), 2)

        # Text annotation on camera feed
        label = f"LED: ({icx}, {icy}) Area:{best['area']:.0f}px Peak:{best['peak']}"
        cv2.putText(annotated, label, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(thresh_rgb, f"THRESH: {thresh_val} (IR LED TRACKED)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Highlight on threshold mask as well
        cv2.circle(thresh_rgb, (icx, icy), 8, (0, 255, 0), 1)

        metadata["area"] = best["area"]
        metadata["peak_brightness"] = best["peak"]
        metadata["point"] = best_pt
        metadata["status"] = "TRACKING"

        return best_pt, annotated, thresh_rgb, metadata


if __name__ == "__main__":
    print("Testing IRDetector with remote crop test...")
    # Synthetic test
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add ambient glare at top edge (y=8, area=250)
    cv2.rectangle(img, (200, 2), (350, 14), (250, 250, 250), -1)
    # Add compact remote LED at (320, 240) (area ~ 25, peak=255)
    cv2.circle(img, (320, 240), 4, (255, 255, 255), -1)

    det = IRDetector(threshold=240)
    pt, debug_img, mask_img, meta = det.detect_debug(img)
    print(f"Detected point: {pt}")
    print(f"Metadata: {meta}")
    assert pt is not None, "Detector failed to find LED"
    assert abs(pt[0] - 320) < 3 and abs(pt[1] - 240) < 3, f"Incorrect point chosen: {pt} (expected 320, 240)"
    print("IRDetector edge-rejection and LED scoring verified successfully!")
