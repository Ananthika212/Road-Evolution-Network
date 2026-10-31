# Road Network Evolution

### A Dual-Module Framework for Road Network Extraction and Change Analysis

## Project Overview

This project presents a framework for analyzing road network evolution using multi-temporal satellite imagery.

It uses two parallel deep learning modules:

- **CRNet:** Extracts road networks from satellite images.
- **LGMM-Net:** Detects road changes between two temporal satellite images.

Their outputs are used for road change characterization and further analysis through the Road Change Reasoning Module (RCRM).

## System Architecture

1. Input satellite images (T1 and T2).
2. Image preprocessing.
3. Parallel processing using CRNet and LGMM-Net.
4. Road network extraction and road change detection.
5. Road change characterization.
6. Road change reasoning and analysis.
7. Result visualization through the user interface.

## Technologies Used

- Python
- PyTorch
- Deep Learning
- Computer Vision
- Satellite Image Processing
- React / [add your actual UI technology]

## Project Structure

Add the actual project folder structure here.

## Installation

```bash
git clone YOUR_REPOSITORY_URL
cd Road-Network-Evolution
pip install -r requirements.txt
```

## Dataset

This project uses satellite imagery datasets for road extraction and road change detection.

Dataset files are not included in this repository.

## Results

Add representative output images and evaluation metrics here.

## Authors

Add your team members' names here.

## License

Add an appropriate license after confirming the permissions for all included code.
