# Road Network Evolution (RoadEvo)

**A Hybrid Framework for Road Network Evolution: CRNet (road extraction) + LGMM-Net (change detection) + RCRM (prioritised map-update recommendations)**

Mepco Schlenk Engineering College, Department of Artificial Intelligence and Data Science.
Authors: Ananthika S, Boomika V. Guide: Dr. K. Thirumoorthy.

Given two satellite images of the same area taken at different times, the system
1. extracts the road network in each image (**CRNet**),
2. detects where roads changed between the two dates (**LGMM-Net**),
3. characterises each change (added / removed / modified), scores it and ranks it **High / Medium / Low** priority for map updating (**RCRM**).

A web application (`UI/`) runs the whole pipeline.

## Repository layout

```
Road-Network-Evolution/
├── CRNet/            road extraction (ResNet34 encoder, GLCDM + SSFRM)
│   ├── model.py      network definition
│   ├── train.py      Massachusetts -> DeepGlobe fine-tuning (written for Kaggle, paths editable)
│   ├── predict.py    tiled inference on an image of any size
│   └── evaluate.py   metrics on the 1,530 DeepGlobe test tiles
├── LGMM-Net/         road-change detection (DLGPE encoder, fusion, PMM decoder)
│   ├── model.py      entry point for the network (architecture in models/)
│   ├── train.py      training  (run through main.py)
│   ├── predict.py    tiled change detection for an image pair
│   └── evaluate.py   metrics on the LEVIR-MCI test pairs
├── RCRM/
│   └── road_change_analysis.py   change characterisation, Road Review Score, priority bands
├── UI/               FastAPI backend + web front end (frontend/ = React source, dist/ = built app)
├── weights/          put the downloaded model weights here (not stored in git)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/Road-Network-Evolution.git
cd Road-Network-Evolution
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt        # for an NVIDIA GPU, install torch from pytorch.org first
```

### Weights

Download the trained weights and place them in `weights/`:

| File | Model | Link |
|---|---|---|
| `crnet_best.pth` | CRNet, Massachusetts pre-trained, DeepGlobe fine-tuned | _add your link_ |
| `lgmm_best.pth`  | LGMM-Net, LEVIR-MCI (road-only labels) | _add your link_ |

### Datasets (not included)

Massachusetts Roads, DeepGlobe Road Extraction, LEVIR-MCI. Download them from their official sources and follow their licences.

## Usage

**Web application**
```bash
cd UI
uvicorn app:app --port 8000
```
Open http://localhost:8000, upload the two images (same size, aligned) and set the pixel size (ground sampling distance, metres per pixel; the models were trained at about 0.5 m).
Weight paths can be changed with the `ROADEVO_CRNET` and `ROADEVO_LGMM` environment variables.

**Road extraction**
```bash
cd CRNet
python predict.py --image area.png --out roads.png
```

**Change detection**
```bash
cd LGMM-Net
python predict.py --t1 before.png --t2 after.png --out change.png
```

**Evaluation**
```bash
cd CRNet     && python evaluate.py --root <DeepGlobe train folder>
cd LGMM-Net  && python evaluate.py --root <LEVIR-MCI folder>
```

**Training**
- CRNet: edit the paths at the top of `CRNet/train.py`, then run it (`MODE = "train"`).
- LGMM-Net: `cd LGMM-Net && python main.py --root_dir <dataset> --dataset_name LEVIR` (see `scripts/`).

## Method summary

- **Tiled inference at native resolution:** overlapping tiles (CRNet 1024 px / overlap 128, LGMM-Net 256 px / overlap 32), Hann-window blending, no resizing.
- **RCRM:** each change region is scored with
  `RRS = 0.20*C + 0.20*A + 0.20*RAS + 0.20*L + 0.20*K`
  (C: mean change probability, A: area, RAS: road association, L: length along the skeleton, K: connectivity impact from nearby junctions/end-points).
  Bands: High >= 0.70, Medium >= 0.40, Low otherwise. The weights and saturation values are chosen defaults, not calibrated against reviewer decisions (see `UI/config.py`).

## Results

| Model | Dataset | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| CRNet | Massachusetts | 78.21 | 73.86 | 75.81 | 61.31 |
| CRNet | DeepGlobe (1,530 test tiles) | 73.67 | 74.52 | 74.10 | 58.85 |
| LGMM-Net | LEVIR-MCI, road class (1,929 test pairs) | 86.23 | 70.04 | 77.30 | 63.00 |

## Limitations

- Both images must be aligned and the same size.
- Length and area depend on the pixel size you enter; a wrong value gives wrong metres.
- Comparison with OpenStreetMap, full road-graph connectivity analysis and calibration of the RCRM weights are future work.

## Acknowledgements and licence

- `CRNet/model.py` is adapted from the official CRNet implementation (CVer-Yang/CRNet).
- `LGMM-Net/` is adapted from the public LGMM implementation (Apache-2.0, see `LGMM-Net/LICENSE`).
- This repository is released under the Apache License 2.0 (see `LICENSE`).
