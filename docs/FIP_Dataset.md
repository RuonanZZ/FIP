# FIP Dataset

The dataset was collected from 10 subjects (`S01`-`S10`) performing 10 motions (`_01`-`_10`).

For each subject, the processed dataset provides the following files after calibration:

1. `acc.pt`: acceleration data from the four IMUs.
2. `rot.pt`: orientation data from the four IMUs.
3. `angle.pt`: flex sensor signals after the physics-informed calibration step.
4. `pose.pt`: SMPL pose parameters captured with the Noitom Perception Neuron 3 system.
5. `joint.pt`: joint positions derived from `pose.pt` using the SMPL model.

Notes:

- IMU order: left forearm, right forearm, back, and waist (root).
- All data are represented in the SMPL coordinate frame.
