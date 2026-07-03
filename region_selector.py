import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Loads an image from disk in BGR format.

    Raises FileNotFoundError if the path does not point to a readable image file.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    return image


def compute_display_scale(image: np.ndarray, max_height: int = 800) -> float:
    """
    Computes a uniform scale factor so the image fits vertically within max_height.

    Returns 1.0 if the image is already smaller than or equal to max_height.
    The same factor applies to both width and height to preserve aspect ratio.
    """
    h = image.shape[0]
    return min(1.0, max_height / h)


class RegionSelector:
    """
    Manages interactive 4-point polygon selection on an OpenCV display window.

    All interaction occurs in display (scaled-down) coordinates. The selected_points
    property automatically converts the result back to original image coordinates,
    so downstream analysis always operates at full resolution.

    Usage
    -----
    image = load_image("glass.jpg")
    scale = compute_display_scale(image)
    selector = RegionSelector(image, display_scale=scale)
    points = selector.run()
    """

    _OVERLAY_COLOR = (0, 220, 60)   # green for polygon lines and markers
    _STATUS_COLOR  = (0, 220, 255)  # yellow for status text
    _POINT_RADIUS  = 6
    _FONT          = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, original_image: np.ndarray, display_scale: float = 1.0):
        """
        Parameters
        ----------
        original_image : np.ndarray
            Full-resolution source image (BGR).
        display_scale : float
            Ratio of displayed resolution to original resolution.
            A value of 0.5 means the window shows the image at half its actual size.
        """
        self._original = original_image
        self._scale    = display_scale

        # Pre-compute the display-resolution version once.
        if display_scale < 1.0:
            dw = int(original_image.shape[1] * display_scale)
            dh = int(original_image.shape[0] * display_scale)
            self._display_image = cv2.resize(original_image, (dw, dh),
                                             interpolation=cv2.INTER_AREA)
        else:
            self._display_image = original_image.copy()

        self._canvas: np.ndarray = self._display_image.copy()
        self._display_points: list[tuple[int, int]] = []

    @property
    def selected_points(self) -> list[tuple[int, int]]:
        """
        Returns the 4 selected corner points mapped to original image coordinates.

        Dividing display-space coordinates by the scale factor reverses the
        downscaling applied during display.
        """
        return [
            (int(x / self._scale), int(y / self._scale))
            for x, y in self._display_points
        ]

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """
        OpenCV mouse callback. Registers a point on each left-click,
        up to a maximum of 4 points, then triggers a canvas redraw.
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(self._display_points) < 4:
            self._display_points.append((x, y))
            self._redraw()

    def _redraw(self) -> None:
        """
        Redraws the canvas from the clean display image, then overlays
        the current set of points and connecting lines.
        """
        self._canvas = self._display_image.copy()
        pts = self._display_points

        # Draw edges between consecutive points.
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(self._canvas, pts[i], pts[i + 1], self._OVERLAY_COLOR, 2)

        # Close the polygon once all 4 points are placed.
        if len(pts) == 4:
            cv2.line(self._canvas, pts[-1], pts[0], self._OVERLAY_COLOR, 2)

        # Draw a filled circle and index label at each point.
        for i, pt in enumerate(pts):
            cv2.circle(self._canvas, pt, self._POINT_RADIUS, self._OVERLAY_COLOR, -1)
            cv2.putText(self._canvas, str(i + 1),
                        (pt[0] + 8, pt[1] - 8),
                        self._FONT, 0.55, self._OVERLAY_COLOR, 2)

    def _reset(self) -> None:
        """Clears all recorded points and restores the canvas to the base image."""
        self._display_points.clear()
        self._canvas = self._display_image.copy()

    def run(self, window_title: str = "Select Region") -> list[tuple[int, int]]:
        """
        Opens the selection window and blocks until the user confirms the region.

        Controls
        --------
        Left click : Place a corner point (sequential, maximum 4).
        Enter      : Confirm the selection. Requires all 4 points to be placed.
        R          : Reset all points and start over.
        Q / Esc    : Abort and exit without returning a result.

        Returns
        -------
        List of 4 (x, y) tuples in original image coordinates.
        """
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_title, self._on_mouse)

        while True:
            frame = self._canvas.copy()
            n     = len(self._display_points)

            status = (
                f"Place point {n + 1} / 4"
                if n < 4
                else "ENTER: confirm    R: reset    Q: quit"
            )
            cv2.putText(frame, status, (10, 32), self._FONT, 0.9, self._STATUS_COLOR, 2)
            cv2.imshow(window_title, frame)

            key = cv2.waitKey(20) & 0xFF

            if key == 13 and n == 4:          # Enter — confirm selection
                break
            elif key == ord('r'):              # R — reset
                self._reset()
            elif key in (ord('q'), 27):        # Q or Esc — abort
                cv2.destroyAllWindows()
                return []

        cv2.destroyAllWindows()
        return self.selected_points


if __name__ == "__main__":
    IMAGE_PATH = "glass.jpg"  # Replace with the actual image path.

    image = load_image(IMAGE_PATH)
    scale = compute_display_scale(image, max_height=800)

    selector = RegionSelector(image, display_scale=scale)
    points   = selector.run()

    if points:
        print("Selected region — original image coordinates:")
        for i, (x, y) in enumerate(points):
            print(f"  Point {i + 1}:  x={x},  y={y}")
    else:
        print("Selection was aborted.")
