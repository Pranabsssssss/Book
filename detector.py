import cv2
import numpy as np
from config import (
    IR_THRESHOLD, IR_MIN_AREA, IR_MAX_AREA,
    CAMERA_EDGE_MARGIN_TOP, CAMERA_EDGE_MARGIN_BOTTOM, CAMERA_EDGE_MARGIN_SIDE,
    CAMERA_WIDTH, CAMERA_HEIGHT
)


class IRDetector:
    """
    Circular Dot & IR LED Light Detector.
    Specifically calculates and filters for compact circular dots and circles
    (IR pen and remote control LED). Rejects elongated strips, ceiling glare,
    and ambient light outside the calibrated display area.
    """
    def __init__(self, threshold=IR_THRESHOLD, min_area=IR_MIN_AREA, max_area=IR_MAX_AREA):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame: np.ndarray, threshold: int = None, display_quad: np.ndarray = None):
        """
        Locate the centroid of the circular IR dot in the camera frame.
        Returns:
            (x, y) tuple in camera pixel coordinates, or None.
        """
        pt, _, _, _ = self.detect_debug(frame, threshold=threshold, display_quad=display_quad)
        return pt

    def detect_debug(self, frame: np.ndarray, threshold: int = None, display_quad: np.ndarray = None):
        """
        Comprehensive circular dot detection with OpenCV visualization and telemetry.
        Draws the DocScanner-style display boundary quad and precision circle reticle.

        Returns:
            point: (cx, cy) tuple or None
            annotated_frame: RGB image with active target reticle and display quad overlay
            thresh_rgb: RGB image of the binary threshold mask
            metadata: dict containing area, circularity, aspect ratio, peak brightness, etc.
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

        # Use 3x3 blur to preserve small pinpoint circular LED cores
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Threshold for intensely bright pixels
        thresh_val = threshold if threshold is not None else self.threshold
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        annotated = frame.copy()

        # -----------------------------------------------------------------
        # Render DocScanner Display Boundary Quad (if calibrated)
        # -----------------------------------------------------------------
        if display_quad is not None:
            quad_int = display_quad.astype(np.int32)
            cv2.polylines(annotated, [quad_int], isClosed=True, color=(0, 240, 255), thickness=2)

            corner_labels = ["TL", "TR", "BR", "BL"]
            for i, (cx_pt, cy_pt) in enumerate(quad_int):
                cv2.circle(annotated, (cx_pt, cy_pt), 5, (0, 255, 100), -1)
                cv2.putText(annotated, corner_labels[i], (cx_pt + 6, cy_pt - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

            # Center tag
            center_x = int(np.mean(display_quad[:, 0]))
            center_y = int(np.mean(display_quad[:, 1]))
            cv2.putText(annotated, "DISPLAY BOUNDS", (center_x - 55, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        metadata = {
            "threshold": thresh_val,
            "num_contours": len(contours) if contours else 0,
            "area": 0.0,
            "circularity": 0.0,
            "aspect_ratio": 0.0,
            "peak_brightness": 0,
            "in_display_area": False,
            "point": None,
            "status": "SEARCHING"
        }

        if not contours:
            cv2.putText(annotated, "NO LIGHT DETECTED", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 60, 60), 2)
            cv2.putText(thresh_rgb, f"THRESH: {thresh_val}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            return None, annotated, thresh_rgb, metadata

        circular_candidates = []

        for c in contours:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            # 1. Edge Exclusion Filter: discard edge-touching border glare
            if (cy < CAMERA_EDGE_MARGIN_TOP or
                cy > (h - CAMERA_EDGE_MARGIN_BOTTOM) or
                cx < CAMERA_EDGE_MARGIN_SIDE or
                cx > (w - CAMERA_EDGE_MARGIN_SIDE)):
                cv2.drawContours(annotated, [c], -1, (180, 50, 50), 1)
                continue

            # 2. Area Filter: must be compact (3 - 150 px)
            if not (self.min_area <= area <= self.max_area):
                cv2.drawContours(annotated, [c], -1, (180, 100, 40), 1)
                continue

            # 3. Geometric Aspect Ratio Filter: must be roughly square / circular
            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            aspect_ratio = float(w_b) / max(1.0, float(h_b))
            if not (0.50 <= aspect_ratio <= 2.0):
                # Reject elongated lines or vertical/horizontal strips
                cv2.drawContours(annotated, [c], -1, (180, 60, 60), 1)
                continue

            # 4. Circularity Filter: 4*pi*area / perimeter^2
            peri = cv2.arcLength(c, True)
            circularity = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0.0
            if circularity < 0.40:
                # Reject jagged or non-circular shapes
                cv2.drawContours(annotated, [c], -1, (180, 60, 60), 1)
                continue

            # 5. Peak Brightness in contour region
            roi_gray = gray[y_b:y_b + h_b, x_b:x_b + w_b]
            peak_val = int(np.max(roi_gray)) if roi_gray.size > 0 else 0

            # 6. Check if inside calibrated display quad (DocScanner area)
            in_display = True
            if display_quad is not None:
                poly = display_quad.astype(np.float32)
                in_display = (cv2.pointPolygonTest(poly, (cx, cy), False) >= 0)

            # Composite Score: favors circular shape, high peak brightness, and display area containment
            score = (peak_val * 2.0) + (circularity * 80.0) - (area * 0.15)
            if not in_display:
                score -= 1000.0  # Penalize outside-display points

            circular_candidates.append({
                "score": score,
                "point": (cx, cy),
                "contour": c,
                "area": area,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "peak": peak_val,
                "in_display": in_display
            })

        if not circular_candidates:
            cv2.putText(annotated, "GLARE / NON-CIRCULAR REJECTED", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 140, 40), 2)
            cv2.putText(thresh_rgb, f"THRESH: {thresh_val} (NO CIRCULAR DOT)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
            metadata["status"] = "FILTERED_NON_CIRCULAR"
            return None, annotated, thresh_rgb, metadata

        # Pick the highest-scoring circular dot
        circular_candidates.sort(key=lambda item: item["score"], reverse=True)
        best = circular_candidates[0]

        best_pt = best["point"]
        best_contour = best["contour"]
        icx, icy = int(best_pt[0]), int(best_pt[1])
        in_display = best["in_display"]

        # Color-coded feedback: Green if inside display, Red if outside
        reticle_color = (0, 255, 0) if in_display else (0, 0, 255)

        # Draw contour boundary
        cv2.drawContours(annotated, [best_contour], -1, (255, 230, 0), 2)

        # Draw circular target reticle
        cv2.circle(annotated, (icx, icy), 14, reticle_color, 2)
        cv2.circle(annotated, (icx, icy), 3, reticle_color, -1)
        cv2.line(annotated, (icx - 20, icy), (icx + 20, icy), reticle_color, 2)
        cv2.line(annotated, (icx, icy - 20), (icx, icy + 20), reticle_color, 2)

        # Annotation text
        status_tag = "IN DISPLAY" if in_display else "OUTSIDE DISPLAY"
        label = f"DOT: ({icx}, {icy}) Circ:{best['circularity']:.2f} [{status_tag}]"
        cv2.putText(annotated, label, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, reticle_color, 2)
        cv2.putText(thresh_rgb, f"THRESH: {thresh_val} (CIRCULAR DOT DETECTED)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Mark on threshold mask
        cv2.circle(thresh_rgb, (icx, icy), 8, (0, 255, 0), 1)

        metadata["area"] = best["area"]
        metadata["circularity"] = best["circularity"]
        metadata["aspect_ratio"] = best["aspect_ratio"]
        metadata["peak_brightness"] = best["peak"]
        metadata["in_display_area"] = in_display
        metadata["point"] = best_pt
        metadata["status"] = "TRACKING_DOT" if in_display else "OUTSIDE_DISPLAY"

        # Only return point for drawing if it is INSIDE the display area!
        return (best_pt if in_display else None), annotated, thresh_rgb, metadata


if __name__ == "__main__":
    print("Testing IRDetector with circular dot and DocScanner quad...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a round circle at (300, 200)
    cv2.circle(img, (300, 200), 5, (255, 255, 255), -1)
    # Add an elongated strip at (100, 100) (area=60, aspect=6.0, non-circular)
    cv2.rectangle(img, (70, 95), (130, 105), (255, 255, 255), -1)

    det = IRDetector(threshold=240)
    quad = np.array([[50, 50], [550, 50], [550, 450], [50, 450]], dtype=np.float32)
    pt, debug_img, mask_img, meta = det.detect_debug(img, display_quad=quad)

    print(f"Detected circular dot: {pt}")
    print(f"Circularity: {meta['circularity']:.2f}, Aspect: {meta['aspect_ratio']:.2f}")
    assert pt is not None, "Failed to detect dot"
    assert abs(pt[0] - 300) < 2 and abs(pt[1] - 200) < 2, f"Wrong point: {pt}"
    print("Circular dot detection and DocScanner boundary test passed!")
