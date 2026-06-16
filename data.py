import os

import numpy as np
import torch

from Aplus.data import *
from Aplus.data.process import add_gaussian_noise
from Aplus.tools.annotations import timing
from Aplus.tools.clothes_imu_syn import *
from articulate.math import axis_angle_to_rotation_matrix, rotation_matrix_to_r6d


ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
TPOSE_GARMENT_PATH = os.path.join(ASSETS_DIR, 'T-Pose_garment.obj')

index_pose = torch.tensor([0, 3, 6, 9, 13, 14, 16, 17, 18, 19])
index_joint = torch.tensor([3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21])


def amass_read(path):
    data = torch.load(path)
    return torch.cat(data, dim=0)


def elbow_angle_caculate(joint_data: torch.Tensor, add_noise=False, encode=True):
    vec_1 = joint_data[:, 20] - joint_data[:, 18]
    vec_2 = joint_data[:, 18] - joint_data[:, 16]
    vec_1 = vec_1 / torch.linalg.norm(vec_1, ord=2, dim=1).unsqueeze(1)
    vec_2 = vec_2 / torch.linalg.norm(vec_2, ord=2, dim=1).unsqueeze(1)
    l_elbow_angle = torch.arccos(torch.sum(vec_1 * vec_2, dim=1).unsqueeze(1))

    vec_1 = joint_data[:, 21] - joint_data[:, 19]
    vec_2 = joint_data[:, 19] - joint_data[:, 17]
    vec_1 = vec_1 / torch.linalg.norm(vec_1, ord=2, dim=1).unsqueeze(1)
    vec_2 = vec_2 / torch.linalg.norm(vec_2, ord=2, dim=1).unsqueeze(1)
    r_elbow_angle = torch.arccos(torch.sum(vec_1 * vec_2, dim=1).unsqueeze(1))

    if encode is False:
        return l_elbow_angle, r_elbow_angle

    if add_noise:
        l_elbow_angle = add_gaussian_noise(l_elbow_angle, sigma=np.pi * 10 / 180)
        r_elbow_angle = add_gaussian_noise(r_elbow_angle, sigma=np.pi * 10 / 180)

    l_elbow_angle = torch.cat([torch.sin(l_elbow_angle), torch.cos(l_elbow_angle)], dim=-1)
    r_elbow_angle = torch.cat([torch.sin(r_elbow_angle), torch.cos(r_elbow_angle)], dim=-1)
    return l_elbow_angle, r_elbow_angle


def elbow_angle_process(angle):
    angle = angle * np.pi / 180
    l_elbow_angle = angle[:, :1]
    r_elbow_angle = angle[:, 1:]
    l_elbow_angle = torch.cat([torch.sin(l_elbow_angle), torch.cos(l_elbow_angle)], dim=-1)
    r_elbow_angle = torch.cat([torch.sin(r_elbow_angle), torch.cos(r_elbow_angle)], dim=-1)
    return l_elbow_angle, r_elbow_angle


class AmassData(BaseDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, y2=None, seq_len=20, shuffle=False, step=1):
        self.x = x[::step]
        self.y = y[::step]
        if y2 is not None:
            self.y2 = y2[::step]
        else:
            self.y2 = None
        self.data_len = len(self.x) - seq_len
        self.seq_len = seq_len
        if shuffle:
            self.indexer = random_index(data_len=self.data_len, seed=42)
        else:
            self.indexer = [i for i in range(self.data_len)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = index % self.data_len
        data = [
            self.x[self.indexer[i]:self.indexer[i] + self.seq_len],
            self.y[self.indexer[i]:self.indexer[i] + self.seq_len],
        ]
        if self.y2 is not None:
            data.append(self.y2[self.indexer[i]:self.indexer[i] + self.seq_len])
        return tuple(data)

    @staticmethod
    @timing
    def load_data(folder_path: str, use_elbow_angle=False, pose_type='r6d', syn=False, add_noise=True) -> dict:
        all_joint_num = len(index_pose)

        if syn:
            rot = amass_read(os.path.join(folder_path, 'syn_rot_on_garment.pt'))
            acc = amass_read(os.path.join(folder_path, 'syn_acc_on_garment.pt'))
        else:
            rot = amass_read(os.path.join(folder_path, 'vrot.pt'))
            acc = amass_read(os.path.join(folder_path, 'vacc.pt'))
        pose = amass_read(os.path.join(folder_path, 'pose.pt'))

        rot_dim = 3
        if pose_type == 'r6d':
            data_len = len(pose)
            len_pose_1 = data_len // 2
            len_pose_2 = data_len - len_pose_1

            pose_seg_1 = pose[:len_pose_1].view(len_pose_1 * 24, 3)
            pose_seg_2 = pose[len_pose_1:].view(len_pose_2 * 24, 3)

            pose_1 = axis_angle_to_rotation_matrix(pose_seg_1)
            pose_1 = rotation_matrix_to_r6d(pose_1).reshape(len_pose_1, 24, 6)

            pose_2 = axis_angle_to_rotation_matrix(pose_seg_2)
            pose_2 = rotation_matrix_to_r6d(pose_2).reshape(len_pose_2, 24, 6)

            pose = torch.cat([pose_1, pose_2], dim=0)
            rot_dim = 6

        acc = torch.clamp(acc, min=-60, max=60)
        acc = torch.cat((acc[:, :3] - acc[:, 3:], acc[:, 3:]), dim=1).bmm(rot[:, -1]) / 30
        rot = torch.cat((rot[:, 3:].transpose(2, 3).matmul(rot[:, :3]), rot[:, 3:]), dim=1)

        joint = amass_read(os.path.join(folder_path, 'joint.pt'))
        joint = joint - joint[:, :1, :]
        joint = joint.bmm(rot[:, -1])
        rot = rotation_matrix_to_r6d(rot.reshape(-1, 3, 3)).reshape(-1, 4, 6)

        if use_elbow_angle:
            elbow_l_angle, elbow_r_angle = elbow_angle_caculate(joint_data=joint, add_noise=add_noise, encode=True)
            imu = torch.cat([acc.flatten(1), rot.flatten(1), elbow_l_angle, elbow_r_angle], dim=1)
        else:
            imu = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1)

        pose_upper_body = pose[:, index_pose].reshape(len(pose), all_joint_num * rot_dim)
        joint_upper_body = joint[:, index_joint].reshape(len(pose), (all_joint_num + 1) * 3)

        return {
            'imu': imu,
            'joint': joint_upper_body,
            'pose': pose_upper_body,
        }


class SensorData(BaseDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, y2=None, seq_len=20, shuffle=False, step=1):
        self.x = x[::step]
        self.y = y[::step]
        if y2 is not None:
            self.y2 = y2[::step]
        else:
            self.y2 = None
        self.data_len = len(self.x) - seq_len
        self.seq_len = seq_len
        if shuffle:
            self.indexer = random_index(data_len=self.data_len, seed=42)
        else:
            self.indexer = [i for i in range(self.data_len)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = index % self.data_len
        data = [
            self.x[self.indexer[i]:self.indexer[i] + self.seq_len],
            self.y[self.indexer[i]:self.indexer[i] + self.seq_len],
        ]
        if self.y2 is not None:
            data.append(self.y2[self.indexer[i]:self.indexer[i] + self.seq_len])
        return tuple(data)

    @staticmethod
    @timing
    def load_data(folder_path=r'FIPData', use_elbow_angle=False, pose_type='r6d', type='all', angle_type='angle', encode=False) -> dict:
        all_joint_num = len(index_pose)

        rot, acc, angle, joint, pose = [], [], [], [], []

        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                if type != 'all' and dir_name.find(type) < 0:
                    continue
                dir_path = os.path.join(root, dir_name)
                print(f'loading {dir_name}')
                rot.append(torch.load(os.path.join(dir_path, 'rot.pt')))
                acc.append(torch.load(os.path.join(dir_path, 'acc.pt')))
                angle.append(torch.load(os.path.join(dir_path, f'{angle_type}.pt')))
                joint.append(torch.load(os.path.join(dir_path, 'joint.pt')).reshape(-1, 24, 3))
                pose.append(torch.load(os.path.join(dir_path, 'pose.pt')))

        rot = torch.cat(rot, dim=0)
        acc = torch.cat(acc, dim=0)
        angle = torch.cat(angle, dim=0)
        joint = torch.cat(joint, dim=0)
        pose = torch.cat(pose, dim=0)

        print(rot.shape)
        print(acc.shape)
        print(angle.shape)
        print(joint.shape)
        print(pose.shape)

        if pose_type == 'r6d':
            data_len = len(pose)
            pose = pose.view(data_len * 24, 3)
            pose = axis_angle_to_rotation_matrix(pose)
            pose = rotation_matrix_to_r6d(pose).reshape(data_len, 24, 6)
            rot_dim = 6
        else:
            rot_dim = 3
            pose = pose.reshape(-1, 24, 3)

        acc = torch.cat((acc[:, :3] - acc[:, 3:], acc[:, 3:]), dim=1).bmm(rot[:, -1]) / 30
        rot = torch.cat((rot[:, 3:].transpose(2, 3).matmul(rot[:, :3]), rot[:, 3:]), dim=1)
        joint = joint - joint[:, :1, :]
        joint = joint.bmm(rot[:, -1])
        rot = rotation_matrix_to_r6d(rot.reshape(-1, 9)).reshape(-1, 4, 6)

        if use_elbow_angle:
            if encode:
                elbow_l_angle, elbow_r_angle = elbow_angle_process(angle)
            else:
                elbow_l_angle, elbow_r_angle = angle[:, :1], angle[:, 1:]
            imu = torch.cat([acc.flatten(1), rot.flatten(1), elbow_l_angle, elbow_r_angle], dim=1)
        else:
            imu = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1)

        pose_upper_body = pose[:, index_pose].reshape(len(pose), all_joint_num * rot_dim)
        joint_upper_body = joint[:, index_joint].reshape(len(pose), (all_joint_num + 1) * 3)

        return {
            'imu': imu,
            'joint': joint_upper_body,
            'pose': pose_upper_body,
        }


class SynPairedIMUData(BaseDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, shuffle=True):
        self.x = x
        self.y = y
        self.data_len = len(x)
        if shuffle:
            self.indexer = random_index(data_len=self.data_len, seed=42)
        else:
            self.indexer = [i for i in range(self.data_len)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = index % self.data_len
        return self.x[self.indexer[i]], self.y[self.indexer[i]]

    @staticmethod
    @timing
    def load_data(folder_path: str, shuffle=False, normalization=True, clothes_imu_calibration=False, data_type='all', type='amass') -> dict:
        if type == 'amass':
            rot_bone = amass_read(os.path.join(folder_path, 'vrot.pt'))
            rot_imu = amass_read(os.path.join(folder_path, 'syn_rot_on_garment.pt'))
            acc_mesh = amass_read(os.path.join(folder_path, 'vacc.pt'))
            acc_imu = amass_read(os.path.join(folder_path, 'syn_acc_on_garment.pt'))
        else:
            rot_bone, rot_imu, acc_mesh, acc_imu = [], [], [], []

            for root, dirs, files in os.walk(folder_path):
                for dir_name in dirs:
                    if data_type != 'all' and dir_name.find(data_type) < 0:
                        continue
                    dir_path = os.path.join(root, dir_name)
                    print(f'loading {dir_name}')
                    rot_bone.append(torch.load(os.path.join(dir_path, 'vrot.pt')))
                    rot_imu.append(torch.load(os.path.join(dir_path, 'rot.pt')))
                    acc_mesh.append(torch.load(os.path.join(dir_path, 'vacc.pt')))
                    acc_imu.append(torch.load(os.path.join(dir_path, 'acc.pt')))

            rot_bone = torch.cat(rot_bone, dim=0)
            rot_imu = torch.cat(rot_imu, dim=0)
            acc_mesh = torch.cat(acc_mesh, dim=0)
            acc_imu = torch.cat(acc_imu, dim=0)

        tpose_clothes_v = obj_load_vertices(path=TPOSE_GARMENT_PATH)
        tpose_rot, _ = imu_syn(tpose_clothes_v)
        device2bone = tpose_rot.transpose(-2, -1)

        if clothes_imu_calibration:
            rot_imu = rot_imu.matmul(device2bone)

        data_len = len(rot_bone)
        acc_mesh = torch.clamp(acc_mesh, min=-60, max=60)
        acc_imu = torch.clamp(acc_imu, min=-60, max=60)

        if normalization:
            acc_mesh = torch.cat((acc_mesh[:, :3] - acc_mesh[:, 3:], acc_mesh[:, 3:]), dim=1).bmm(rot_bone[:, -1]) / 30
            acc_imu = torch.cat((acc_imu[:, :3] - acc_imu[:, 3:], acc_imu[:, 3:]), dim=1).bmm(rot_imu[:, -1]) / 30

        acc_mesh = acc_mesh.reshape(data_len, -1)
        acc_imu = acc_imu.reshape(data_len, -1)

        if normalization:
            rot_bone = torch.cat((rot_bone[:, 3:].transpose(2, 3).matmul(rot_bone[:, :3]), rot_bone[:, 3:]), dim=1)
            rot_imu = torch.cat((rot_imu[:, 3:].transpose(2, 3).matmul(rot_imu[:, :3]), rot_imu[:, 3:]), dim=1)

        rot_bone = rot_bone.view(data_len * 4, 3, 3)
        rot_imu = rot_imu.view(data_len * 4, 3, 3)

        rot_bone = rotation_matrix_to_r6d(rot_bone).reshape(data_len, -1)
        rot_imu = rotation_matrix_to_r6d(rot_imu).reshape(data_len, -1)

        data_mesh = torch.cat([acc_mesh, rot_bone], dim=-1)
        data_garment = torch.cat([acc_imu, rot_imu], dim=-1)

        if shuffle:
            new_idx = random_index(data_len=len(data_mesh), seed=42)
            data_mesh = data_mesh[new_idx]
            data_garment = data_garment[new_idx]

        return {
            'data_mesh': data_mesh,
            'data_garment': data_garment,
        }
