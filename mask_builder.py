import cv2
import numpy as np


def build_polygon_mask(image: np.ndarray,
                       points: list[tuple[int, int]]) -> np.ndarray:
    """
    Creates a single-channel binary mask the same size as the input image.

    Pixels inside the polygon defined by `points` are set to 255 (white).
    All other pixels are set to 0 (black).

    Parameters
    ----------
    image : np.ndarray
        Source image — used only to determine height and width.
    points : list of (int, int)
        Four (x, y) corner coordinates in original image space.

    Returns
    -------
    np.ndarray
        Binary mask of shape (H, W) with dtype uint8.
    """
    h, w  = image.shape[:2]
    mask  = np.zeros((h, w), dtype=np.uint8)
    pts   = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(mask, [pts], color=255)
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a binary mask to a BGR image.

    Pixels outside the mask region are set to black (0, 0, 0).
    This is a standard bitwise operation — no pixel values inside
    the mask are modified.

    Parameters
    ----------
    image : np.ndarray
        Full-resolution BGR image.
    mask : np.ndarray
        Binary mask of shape (H, W) with dtype uint8.

    Returns
    -------
    np.ndarray
        Masked BGR image, same shape as input.
    """
    return cv2.bitwise_and(image, image, mask=mask)


def preview_selection(image: np.ndarray,
                      mask: np.ndarray,
                      window_title: str = "Selection Preview",
                      max_height: int = 800,
                      max_width: int = 1600) -> None:
    """
    Displays a side-by-side preview: original image on the left,
    masked region on the right.

    The composite is scaled to fit within both max_height and max_width.
    Stacking two images horizontally doubles the width, so both axes must
    be checked independently to avoid the window exceeding screen bounds.
    Press any key to close the window.
    """
    masked = apply_mask(image, mask)

    # Place the two images side by side for easy comparison.
    composite = np.hstack([image, masked])

    h, w = composite.shape[:2]
    scale = min(1.0, max_height / h, max_width / w)

    if scale < 1.0:
        composite = cv2.resize(composite,
                               (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

    cv2.imshow(window_title, composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Quick integration test — connects Step 1 and Step 2 end-to-end.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from region_selector import load_image, compute_display_scale, RegionSelector

    IMAGE_PATH = "glass.jpg"  # Replace with the actual image path.

    image  = load_image(IMAGE_PATH)
    scale  = compute_display_scale(image, max_height=800)

    selector = RegionSelector(image, display_scale=scale)
    points   = selector.run()

    if not points:
        print("Selection aborted.")
    else:
        mask   = build_polygon_mask(image, points)
        preview_selection(image, mask)

        total_pixels  = int(np.sum(mask > 0))
        print(f"Selected region: {total_pixels:,} pixels")
        print(f"Approximate area: {total_pixels / (image.shape[0] * image.shape[1]) * 100:.1f}% of full image")
