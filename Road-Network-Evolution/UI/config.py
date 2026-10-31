"""Settings for the road-change pipeline.

Values marked ASSUMED are chosen defaults, not tuned or calibrated; change them if you
need different behaviour.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- model files -----------------------------------------------------
    crnet_weights: str = "../weights/crnet_best.pth"   # DeepGlobe fine-tuned CRNet
    lgmm_weights: str = "../weights/lgmm_best.pth"     # LEVIR-MCI road-only LGMM-Net

    # ---- imagery ---------------------------------------------------------
    gsd_m: float = 0.5                            # metres per pixel (both models trained at ~0.5 m)

    # ---- inference -------------------------------------------------------
    road_threshold: float = 0.5                   # CRNet probability -> road mask
    change_threshold: float = 0.5                 # LGMM change probability -> change mask
    crnet_tile: int = 1024                        # CRNet native size
    crnet_overlap: int = 128
    lgmm_bgr: bool = True                         # LGMM-Net was trained on BGR-ordered input (cv2.imread, no conversion)
    lgmm_tile: int = 256                          # LGMM-Net native size
    lgmm_overlap: int = 32

    # ---- characterisation --------------------------------------------------
    min_region_px: int = 30                       # ignore tiny change blobs (ASSUMED)
    min_road_overlap: float = 0.30                # tau_rho: regions with road overlap < 0.30 are removed (paper)
    road_buffer_px: int = 6                       # tolerance when matching change to roads (ASSUMED)
    node_buffer_px: int = 12                      # distance for "connection affected" (ASSUMED)
    added_frac: float = 0.5                       # share of road pixels only in T2 -> "added" (ASSUMED)
    removed_frac: float = 0.5                     # share of road pixels only in T1 -> "removed" (ASSUMED)

    # ---- RCRM (paper eq. 20-25):  S = wc*c + wa*A + wo*rho + wl*L + wn*I,  all terms in [0, 1] ---------
    w_conf: float = 0.20                          # change confidence      (paper Table V)
    w_area: float = 0.20                          # changed area           (paper Table V)
    w_assoc: float = 0.20                         # road association       (paper Table V)
    w_length: float = 0.20                        # affected road length   (paper Table V)
    w_conn: float = 0.20                          # connectivity impact    (paper Table V)
    area_max_m2: float = 500.0                    # A_max: area at which the indicator saturates at 1 (ASSUMED)
    length_ref_m: float = 100.0                   # L_max: length at which the indicator saturates at 1 (ASSUMED)
    nodes_max: int = 3                            # beta_max: junctions/end points at which it saturates (ASSUMED)

    # ---- priority bands on S (paper Table V) ------------------------------------------------------
    high_at: float = 0.70
    medium_at: float = 0.40
