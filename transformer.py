import numpy as np
import cv2


class CoordinateTransformer:
    """
    Transforms coordinates from camera space to screen space using
    a perspective homography matrix computed during calibration.
    """
    def __init__(self, homography_matrix: np.ndarray):
        if homography_matrix is None or homography_matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be a 3x3 numpy array.")
        self.H = np.asarray(homography_matrix, dtype=np.float64)

    def transform(self, camera_point: tuple) -> tuple:
        """
        Transform a single 2D camera coordinate into screen coordinate space.

        Args:
            camera_point: (cam_x, cam_y)

        Returns:
            (screen_x, screen_y) as floats
        """
        cx, cy = float(camera_point[0]), float(camera_point[1])

        # Closed-form perspective projection (faster than numpy array wrapping per frame)
        denom = self.H[2, 0] * cx + self.H[2, 1] * cy + self.H[2, 2]
        if abs(denom) < 1e-9:
            return (0.0, 0.0)

        sx = (self.H[0, 0] * cx + self.H[0, 1] * cy + self.H[0, 2]) / denom
        sy = (self.H[1, 0] * cx + self.H[1, 1] * cy + self.H[1, 2]) / denom

        return (float(sx), float(sy))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform an array of points of shape (N, 2).
        """
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed.reshape(-1, 2)


if __name__ == "__main__":
    print("Testing CoordinateTransformer...")
    # Identity matrix test
    ident = np.eye(3, dtype=np.float64)
    tx = CoordinateTransformer(ident)
    p = tx.transform((123.4, 567.8))
    print(f"Transformed identity point: {p}")
    assert abs(p[0] - 123.4) < 1e-4 and abs(p[1] - 567.8) < 1e-4, "Identity test failed"

    # Scale and shift test
    # screen_x = cam_x * 2 + 10
    # screen_y = cam_y * 3 + 20
    test_H = np.array([
        [2.0, 0.0, 10.0],
        [0.0, 3.0, 20.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    tx2 = CoordinateTransformer(test_H)
    p2 = tx2.transform((50.0, 100.0))
    print(f"Transformed test point: {p2} (expected 110.0, 320.0)")
    assert abs(p2[0] - 110.0) < 1e-4 and abs(p2[1] - 320.0) < 1e-4, "Scale test failed"
    print("CoordinateTransformer tests passed successfully!")
