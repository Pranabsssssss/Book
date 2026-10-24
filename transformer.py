import numpy as np
import cv2
from config import SCREEN_WIDTH, SCREEN_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT


class CoordinateTransformer:
    """
    Transforms coordinates between camera space and screen space using
    a perspective homography matrix computed during calibration.
    Also calculates the DocScanner-style display boundary quadrilateral in camera view.
    """
    def __init__(self, homography_matrix: np.ndarray):
        if homography_matrix is None or homography_matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be a 3x3 numpy array.")
        self.H = np.asarray(homography_matrix, dtype=np.float64)

        # Compute inverse homography to map screen coordinates back into camera space
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H_inv = np.eye(3, dtype=np.float64)

        # Compute the 4 corners of the display as seen by the camera (DocScanner quad)
        # Order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        screen_corners = np.array([
            [[0.0, 0.0]],
            [[float(SCREEN_WIDTH), 0.0]],
            [[float(SCREEN_WIDTH), float(SCREEN_HEIGHT)]],
            [[0.0, float(SCREEN_HEIGHT)]]
        ], dtype=np.float64)

        try:
            cam_corners = cv2.perspectiveTransform(screen_corners, self.H_inv)
            quad = cam_corners.reshape(4, 2)
            # Validate that the computed quad has sensible area
            quad_area = cv2.contourArea(quad.astype(np.float32))
            if quad_area > 5000:
                self.display_quad = quad
            else:
                # Fallback to full camera view if quad is collapsed
                self.display_quad = np.array([
                    [0.0, 0.0],
                    [float(CAMERA_WIDTH), 0.0],
                    [float(CAMERA_WIDTH), float(CAMERA_HEIGHT)],
                    [0.0, float(CAMERA_HEIGHT)]
                ], dtype=np.float64)
        except Exception:
            self.display_quad = np.array([
                [0.0, 0.0],
                [float(CAMERA_WIDTH), 0.0],
                [float(CAMERA_WIDTH), float(CAMERA_HEIGHT)],
                [0.0, float(CAMERA_HEIGHT)]
            ], dtype=np.float64)

    def get_display_quad(self) -> np.ndarray:
        """
        Returns the 4 camera coordinates of the display screen boundaries
        in order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
        """
        return self.display_quad

    def is_point_in_display_area(self, cam_x: float, cam_y: float) -> bool:
        """
        Check if a camera point falls inside the calibrated display area
        (like DocScanner bounding paper edges).
        """
        poly = self.display_quad.astype(np.float32)
        res = cv2.pointPolygonTest(poly, (float(cam_x), float(cam_y)), False)
        return res >= 0

    def transform(self, camera_point: tuple) -> tuple:
        """
        Transform a single 2D camera coordinate into screen coordinate space.

        Args:
            camera_point: (cam_x, cam_y)

        Returns:
            (screen_x, screen_y) as floats
        """
        cx, cy = float(camera_point[0]), float(camera_point[1])

        denom = self.H[2, 0] * cx + self.H[2, 1] * cy + self.H[2, 2]
        if abs(denom) < 1e-9:
            # Fallback linear scale to screen size
            sx = (cx / float(CAMERA_WIDTH)) * float(SCREEN_WIDTH)
            sy = (cy / float(CAMERA_HEIGHT)) * float(SCREEN_HEIGHT)
            return (float(sx), float(sy))

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
    print("Testing CoordinateTransformer with fallback protection...")
    ident = np.eye(3, dtype=np.float64)
    tx = CoordinateTransformer(ident)
    quad = tx.get_display_quad()
    print("Display Quad:\n", quad)
    assert quad.shape == (4, 2), "Quad shape mismatch"
    assert tx.is_point_in_display_area(320, 240), "Center point test failed"
    print("CoordinateTransformer tests passed successfully!")
