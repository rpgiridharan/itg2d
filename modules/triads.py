import math
import numpy as np

def make_triad(
    *,
    kx,
    ky,
    qmag_min=0.05,
    qmag_max=0.95,
    max_tries=100_000,
    seed=None,
):
    """Construct a triad in continuous k-space (not constrained to any grid).

    Enforces exact algebraic closure:
        k + p + q = 0

    Also enforces:
        - nonzero y-components for q,k,p
        - |q| < |k| < |p|

    Notes
    -----
    - `q` is sampled by choosing a magnitude and angle.
    - Use `seed` for reproducibility.
    """

    k2 = kx**2 + ky**2
    if abs(ky) <= 1e-6:
        raise ValueError("ky ~ 0; choose a nonzonal k")

    rng = np.random.default_rng(seed)

    for _ in range(int(max_tries)):
        qmag = rng.uniform(qmag_min, qmag_max)
        theta = rng.uniform(0.0, 2 * math.pi)
        qx = qmag * math.cos(theta)
        qy = qmag * math.sin(theta)
        if abs(qy) <= 1e-6:
            continue

        px = -(kx + qx)
        py = -(ky + qy)
        if abs(py) <= 1e-6:
            continue

        q2 = qx * qx + qy * qy
        p2 = px * px + py * py
        if not (q2 < k2 < p2):
            continue

        return (qx, qy, kx, ky, px, py)

    raise RuntimeError(
        "Failed to find a continuous triad with |q|<|k|<|p| and nonzero y"
    )
