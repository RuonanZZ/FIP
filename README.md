# FIP: Endowing Robust Motion Capture on Daily Garments by Fusing Flex and Inertial Sensors

This repository contains the implementation of our CHI 2025 paper, *FIP: Endowing Robust Motion Capture on Daily Garments by Fusing Flex and Inertial Sensors*, including pretrained weights, training scripts, and evaluation code.

[Paper](https://dl.acm.org/doi/full/10.1145/3706598.3714140) | [Project Page](https://fangjw-0722.github.io/FIP.github.io/)

- [train_DiffusionVAE.py](./train_DiffusionVAE.py): trains the displacement latent diffusion model.
- [train_Poser.py](./train_Poser.py): trains the pose fusion predictor.
- [evaluate.py](./evaluate.py): evaluates angular error, elbow angular error, positional error, and jitter.

## Dataset

The FIP dataset is available at:
https://drive.google.com/drive/folders/1cdiN57Q80aBQ0xb7fL7qXd3fEYojTCD-?usp=sharing

See [docs/FIP_Dataset.md](./docs/FIP_Dataset.md) for details.

## Acknowledgments

Some parts of the codebase are adapted from [PIP](https://github.com/Xinyu-Yi/PIP) and [LIP](https://github.com/ZuoCX1996/Loose-Inertial-Poser).
The synthesized IMU data used in this project are derived from [LIP](https://github.com/ZuoCX1996/Loose-Inertial-Poser).
The `SMPL_MALE` model can be downloaded from https://smpl.is.tue.mpg.de/.
