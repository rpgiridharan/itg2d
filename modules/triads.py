import math
from typing import Optional
import numpy as np

def make_triad(
    *,
    kx: float,
    ky: float,
    require_nonzero_y: bool = True,
    y_eps: float = 1e-12,
    qmag_min_frac: float = 0.05,
    qmag_max_frac: float = 0.95,
    max_tries: int = 100_000,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> tuple[float, float, float, float, float, float]:
    """Construct a triad in continuous k-space (not constrained to any grid).

    Enforces exact algebraic closure:
        k + p + q = 0

    Also enforces:
        - nonzero y-components for q,k,p (if require_nonzero_y)
        - |q| < |k| < |p|

    Notes
    -----
    - `q` is sampled by choosing a magnitude and angle.
    - Use `seed` (or pass `rng`) for reproducibility.
    Returns
    -------
    (qx, qy, kx, ky, px, py)
        The triad components satisfying k + p + q = 0.
    """

    kx = float(kx)
    ky = float(ky)
    k2 = kx * kx + ky * ky
    if k2 == 0.0:
        raise ValueError("k cannot be zero")
    if require_nonzero_y and abs(ky) <= y_eps:
        raise ValueError("ky ~ 0; choose a nonzonal k or relax y_eps")

    if not (0.0 < qmag_min_frac < qmag_max_frac < 1.0):
        raise ValueError("Require 0 < qmag_min_frac < qmag_max_frac < 1")

    if rng is None:
        rng = np.random.default_rng(seed)

    kmag = math.sqrt(k2)
    qmag_min = qmag_min_frac * kmag
    qmag_max = qmag_max_frac * kmag

    for _ in range(int(max_tries)):
        qmag = float(rng.uniform(qmag_min, qmag_max))
        theta = float(rng.uniform(0.0, 2 * math.pi))
        qx = qmag * math.cos(theta)
        qy = qmag * math.sin(theta)
        if require_nonzero_y and abs(qy) <= y_eps:
            continue

        px = -(kx + qx)
        py = -(ky + qy)
        if require_nonzero_y and abs(py) <= y_eps:
            continue

        q2 = qx * qx + qy * qy
        p2 = px * px + py * py
        if not (q2 < k2 < p2):
            continue

        return (qx, qy, kx, ky, px, py)

    raise RuntimeError(
        "Failed to find a continuous triad with |q|<|k|<|p| and nonzero y. "
        "Try changing k, relaxing y_eps, or adjusting qmag_min_frac/qmag_max_frac."
    )
