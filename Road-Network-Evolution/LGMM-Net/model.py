"""LGMM-Net model entry point.

The architecture is defined in models/ (encoder.py = DLGPE, fusion.py = feature fusion,
decoder.py = PMM progressive Mamba-based decoder, model.py = the assembled LGMMNet).
Adapted from the public LGMM implementation (Apache-2.0, see LICENSE).
"""
from models.model import LGMMNet, get_net  # noqa: F401


def build_lgmm_net():
    """The configuration used in RoadEvo: 3-channel input, 2 classes (background / road change)."""
    return LGMMNet(3, 2, ablation_setting={"encoder_module": "lgmm", "decoder_module": "lgmm"})
