# FIP Dataset

The dataset was collected from 10 subjects (`S01`-`S10`). Each subject performs 10 motions, denoted by suffixes `_01`-`_10` (for example, `S01_01`). The processed dataset contains 371,122 frames in total.

For each processed sequence, the dataset provides the following calibrated `.pt` files:

| File | Shape | Description |
| --- | --- | --- |
| `acc.pt` | `[T, 4, 3]` | Acceleration from four IMUs. |
| `rot.pt` | `[T, 4, 3, 3]` | Orientation from four IMUs, represented as rotation matrices. |
| `angle.pt` | `[T, 2]` | Elbow flexion angles obtained from flex sensors after the physics-informed calibration step, in degrees. |
| `pose.pt` | `[T, 24, 3]` | SMPL pose parameters captured with the Noitom Perception Neuron 3 system, represented as axis-angle rotations. |
| `joint.pt` | `[T, 24, 3]` | SMPL joint positions derived from `pose.pt` using the SMPL model. |

*Note: `T` is the number of frames in the sequence. All files store `float32` PyTorch tensors.*

The data can be loaded with `SensorData` in `data.py`, which handles the preprocessing used by the training and evaluation scripts.

Notes:

- IMU order in `acc.pt` and `rot.pt`: left forearm, right forearm, back, and waist (root).
- Angle order in `angle.pt`: left elbow and right elbow.
- In the raw files, vector, rotation, and SMPL data are represented in the SMPL coordinate frame.

Correction for Supplementary Material Tables 8-10:

Tables 8-10 in the supplementary material mistakenly use subject IDs in data collection order. The correct correspondence is:

| Main | Supp |
| --- | --- |
| S1 | S3 |
| S2 | S4 |
| S3 | S8 |
| S4 | S1 |
| S5 | S5 |
| S6 | S9 |
| S7 | S7 |
| S8 | S10 |
| S9 | S2 |
| S10 | S6 |
