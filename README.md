# FIP: Endowing Robust Motion Capture on Daily Garments by Fusing Flex and Inertial Sensors

This repository contains the implementation of our CHI 2025 paper, *FIP: Endowing Robust Motion Capture on Daily Garments by Fusing Flex and Inertial Sensors*, including pretrained weights, training scripts, and evaluation code.

- [train_DiffusionVAE.py](./train_DiffusionVAE.py): trains the displacement latent diffusion model.
- [train_Poser.py](./train_Poser.py): trains the pose fusion predictor.
- [evaluate.py](./evaluate.py): evaluates angular error, elbow angular error, positional error, and jitter.

## Dataset

The FIP dataset is available at:
https://www.dropbox.com/scl/fo/ggrvm8x2xjhu1m0pjomc9/ADClW3gbt4swggoulhndBKA?rlkey=bagguhrnze7fdvgr2toggce0v&st=p3fj8g1e&dl=0

See [docs/FIP_Dataset.md](./docs/FIP_Dataset.md) for details.

## Acknowledgments

Some parts of the codebase are adapted from [PIP](https://github.com/Xinyu-Yi/PIP) and [LIP](https://github.com/ZuoCX1996/Loose-Inertial-Poser).
The synthesized IMU data used in this project are derived from [LIP](https://github.com/ZuoCX1996/Loose-Inertial-Poser).
The `SMPL_MALE` model can be downloaded from https://smpl.is.tue.mpg.de/.
