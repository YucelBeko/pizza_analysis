import cv2
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Threshold constants.
#
# All values apply to the HSV color space.
# OpenCV ranges: H -> 0-179,  S -> 0-255,  V -> 0-255
#
# Three-tier soiling classification:
#
#   Charring (heavy)  :  V < THRESHOLD_DARK_V_MAX
#   Medium soiling    :  THRESHOLD_DARK_V_MAX <= V < THRESHOLD_MEDIUM_V_MAX
#   Haze (light)      :  V >= THRESHOLD_MEDIUM_V_MAX  AND  S < THRESHOLD_HAZE_S_MAX
#   Clean             :  everything else inside the ROI
#
# If results appear over- or under-segmented on new samples, adjust here only.
# ---------------------------------------------------------------------------

# Heavy soiling (charring): pixels whose brightness falls below this value.
THRESHOLD_DARK_V_MAX: int = 50

# Medium soiling: brownish / grey-haze band between charring and bright haze.
# The upper bound also serves as the lower bound for the haze classifier.
THRESHOLD_MEDIUM_V_MAX: int = 130

# Light haze / white deposits: bright pixels with near-zero saturation.
THRESHOLD_HAZE_S_MAX: int = 65

# Overlay blend weight — how strongly the colour layer appears over the original.
# 0.0 = only original image,  1.0 = fully opaque colour overlay.
OVERLAY_ALPHA: float = 0.45


@dataclass
class StainReport:
    """
    Holds pixel counts and coverage ratios for each soiling category
    within the analysed region.
    """
    total_pixels:  int
    dark_pixels:   int
    medium_pixels: int
    haze_pixels:   int
    clean_pixels:  int

    @property
    def dark_pct(self) -> float:
        """Percentage of the region classified as heavy charring."""
        return 100.0 * self.dark_pixels / self.total_pixels if self.total_pixels else 0.0

    @property
    def medium_pct(self) -> float:
        """Percentage of the region classified as medium soiling."""
        return 100.0 * self.medium_pixels / self.total_pixels if self.total_pixels else 0.0

    @property
    def haze_pct(self) -> float:
        """Percentage of the region classified as light haze / white deposits."""
        return 100.0 * self.haze_pixels / self.total_pixels if self.total_pixels else 0.0

    @property
    def soiling_pct(self) -> float:
        """Combined soiling percentage across all three soiling categories."""
        return self.dark_pct + self.medium_pct + self.haze_pct

    @property
    def clean_pct(self) -> float:
        """Percentage of the region that shows no detected soiling."""
        return 100.0 * self.clean_pixels / self.total_pixels if self.total_pixels else 0.0

    def summary(self) -> str:
        """Returns a formatted multi-line text summary of the report."""
        lines = [
            "=" * 42,
            "         Stain Analysis Report",
            "=" * 42,
            f"  Region area         : {self.total_pixels:>10,} px",
            f"  Charring  [RED]     : {self.dark_pixels:>10,} px  ({self.dark_pct:5.1f} %)",
            f"  Medium    [ORANGE]  : {self.medium_pixels:>10,} px  ({self.medium_pct:5.1f} %)",
            f"  Haze      [YELLOW]  : {self.haze_pixels:>10,} px  ({self.haze_pct:5.1f} %)",
            "  " + "-" * 38,
            f"  Total soiling       : {'':>10}     {self.soiling_pct:5.1f} %",
            f"  Clean area          : {self.clean_pixels:>10,} px  ({self.clean_pct:5.1f} %)",
            "=" * 42,
        ]
        return "\n".join(lines)


def analyze_stains(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[StainReport, np.ndarray]:
    """
    Classifies soiling within the masked region using fixed HSV thresholds.

    Each pixel inside the mask is assigned to exactly one of four categories:
    charring, medium soiling, haze, or clean. The classifiers are applied in
    priority order — a pixel can only belong to one category.

    Color coding on the overlay
    ---------------------------
    Red    (0, 0, 220)    : charring — heavy dark soiling
    Orange (0, 128, 255)  : medium soiling — brownish / dark grey haze
    Yellow (0, 210, 220)  : light haze / white deposits
    (no colour)           : clean

    Parameters
    ----------
    image : np.ndarray
        Full-resolution BGR source image.
    mask : np.ndarray
        Binary mask (uint8, 0 or 255) defining the region of interest.

    Returns
    -------
    report : StainReport
    overlay : np.ndarray
        BGR image at the same resolution as the input, with soiling regions
        highlighted and a semi-transparent blend applied inside the ROI.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)

    roi = mask > 0  # boolean array: True only inside the selected polygon

    # --- Classification (mutually exclusive, applied in priority order) ---

    # Heavy soiling: brightness alone determines charring.
    # Hue is ignored because heavily charred areas lose all colour information.
    dark_mask = (v_ch < THRESHOLD_DARK_V_MAX) & roi

    # Medium soiling: brownish / grey-haze band between the two extremes.
    # Pixels already classified as charring are explicitly excluded.
    medium_mask = (
        (v_ch >= THRESHOLD_DARK_V_MAX) &
        (v_ch <  THRESHOLD_MEDIUM_V_MAX) &
        roi & ~dark_mask
    )

    # Light haze / white deposits: high brightness combined with near-zero
    # saturation. Pixels already classified above are excluded.
    haze_mask = (
        (v_ch >= THRESHOLD_MEDIUM_V_MAX) &
        (s_ch <  THRESHOLD_HAZE_S_MAX) &
        roi & ~dark_mask & ~medium_mask
    )

    # Clean: inside the ROI but not matched by any soiling classifier.
    clean_mask = roi & ~dark_mask & ~medium_mask & ~haze_mask

    report = StainReport(
        total_pixels=int(np.sum(roi)),
        dark_pixels=int(np.sum(dark_mask)),
        medium_pixels=int(np.sum(medium_mask)),
        haze_pixels=int(np.sum(haze_mask)),
        clean_pixels=int(np.sum(clean_mask)),
    )

    # --- Overlay construction ---
    color_layer = image.copy()
    color_layer[dark_mask]   = (0,   0, 220)   # red    -> charring
    color_layer[medium_mask] = (0, 128, 255)   # orange -> medium soiling
    color_layer[haze_mask]   = (0, 210, 220)   # yellow -> light haze

    # Blend the colour layer with the original only inside the ROI.
    blended = cv2.addWeighted(image, 1.0 - OVERLAY_ALPHA,
                              color_layer, OVERLAY_ALPHA, 0)

    # Pixels outside the mask retain the unmodified original values.
    output = image.copy()
    output[roi] = blended[roi]

    return report, output


def show_result(
    original: np.ndarray,
    overlay: np.ndarray,
    report: StainReport,
    max_height: int = 800,
    max_width: int = 1600,
) -> None:
    """
    Displays the original and annotated overlay side by side,
    scaled to fit within the given screen dimensions.

    Prints the StainReport summary to stdout before opening the window.
    Press any key to close.
    """
    print(report.summary())

    composite = np.hstack([original, overlay])
    h, w = composite.shape[:2]
    scale = min(1.0, max_height / h, max_width / w)

    if scale < 1.0:
        composite = cv2.resize(composite,
                               (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

    cv2.imshow("Analysis Result  |  Original (left)  vs  Overlay (right)", composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
