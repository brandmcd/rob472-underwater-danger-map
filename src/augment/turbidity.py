"""
Turbidity augmentation for underwater images.

Simulates the visual degradation caused by suspended particles in water:
  1. Gaussian blur        — particle scattering diffuses light
  2. Backscatter veil     — particles reflect ambient light back as a haze
  3. Red-channel fade     — red wavelengths are absorbed fastest in water

Core function
─────────────
    result = apply_turbidity(img, level)

    img:   (H, W, 3) uint8 RGB image
    level: float in [0.0, 1.0]
             0.0 → no change (original image returned)
             1.0 → heavily degraded (strong blur, green-tinted haze, muted red)

Effect magnitudes at level = 1.0
─────────────────────────────────
    Gaussian sigma   4.0 px
    Veil opacity     45 %   (blended toward a slightly green haze colour)
    Red attenuation  35 %   (red channel multiplied by 0.65)

These values produce a plausible range of turbidity seen in reef and near-coastal
underwater footage.  Intermediate levels scale linearly.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# Colour of the backscatter veil in RGB [0, 1].
# Slightly greenish-white to mimic particle-laden coastal water.
_VEIL_RGB = np.array([0.55, 0.70, 0.60], dtype=np.float32)

# Maximum Gaussian blur sigma (pixels) at level = 1.0
_MAX_SIGMA: float = 4.0

# Maximum veil blend weight at level = 1.0  (0 = no veil, 1 = full veil colour)
_MAX_VEIL: float = 0.45

# Maximum red-channel attenuation at level = 1.0
_MAX_RED_ATTEN: float = 0.35


def apply_turbidity(img: np.ndarray, level: float) -> np.ndarray:
    """
    Apply simulated turbidity degradation to an RGB image.

    Three physically-motivated effects are applied in sequence:
      1. Gaussian blur (light scattering by suspended particles)
      2. Additive green-tinted veil (backscatter / particle haze)
      3. Red-channel attenuation (water absorbs red wavelengths fastest)

    Args:
        img:   Input image, shape (H, W, 3), dtype uint8, values in [0, 255].
               Must be RGB channel order.
        level: Degradation severity in [0.0, 1.0].
               0.0 = original image, 1.0 = maximum turbidity.

    Returns:
        Degraded image, shape (H, W, 3), dtype uint8.
        A copy is always returned — the input array is not modified.

    Raises:
        ValueError: If level is outside [0.0, 1.0].
    """
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"level must be in [0.0, 1.0], got {level}")

    if level == 0.0:
        return img.copy()

    # Work in float32 [0, 1] to avoid clipping artifacts during blending
    result = img.astype(np.float32) / 255.0   # (H, W, 3)

    # 1. Gaussian blur — sigma scales from 0 to _MAX_SIGMA
    sigma = level * _MAX_SIGMA
    result = np.stack(
        [gaussian_filter(result[..., c], sigma=sigma) for c in range(3)],
        axis=-1,
    )

    # 2. Backscatter veil — blend image toward the haze colour
    veil_w = level * _MAX_VEIL
    result = (1.0 - veil_w) * result + veil_w * _VEIL_RGB

    # 3. Red-channel attenuation
    red_scale = 1.0 - level * _MAX_RED_ATTEN
    result[..., 0] *= red_scale

    return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
