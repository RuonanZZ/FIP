import numpy as np
import torch

from Aplus.data import *
from Aplus.data.process import add_gaussian_noise
import os
from articulate.math import axis_angle_to_rotation_matrix, rotation_matrix_to_r6d, axis_angle_to_quaternion, euler_angle_to_rotation_matrix, rotation_matrix_to_euler_angle, quaternion_to_rotation_matrix
from tqdm import tqdm
from Aplus.tools.annotations import timing
import quaternion
from Aplus.tools.clothes_imu_syn import *

index_pose = torch.tensor([0, 3, 6, 9, 13, 14, 16, 17, 18, 19])
index_joint = torch.tensor([3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21])

def amass_read_seg(path, min_len=256):
    data = torch.load(path)
    selected_data = []
    seg_info = []
    for slice in data:
        # print(slice.shape)
        if len(slice) < min_len:
            continue
        else:
            selected_data.append(slice)
            seg_info.append(len(slice))
    data = torch.cat(selected_data, dim=0)
    # print(acc_t.shape)
    # index_info = seg_info_2_index_info(seg_info)
    # print(index_info)
    # find_seg_index(index_info, 2200)
    return data, seg_info

def seg_info_2_index_info(seg_info):
    index_info = [0]
    for v in seg_info:
        index_info.append(index_info[-1] + v)
    return index_info

def find_seg_index(index_info, data_index, n_seg=0):
    # index_info = np.array(index_info)
    # mask = np.array(index_info.__le__(data_index), dtype='int')
    # seq_index = int(mask.sum()) - 1

    seq_index = -1
    # n_seg = 0
    if n_seg != 0:
        seq_index = n_seg - 1

    # 0 1 2 3 4
    # 0 2 4 6 8
    # print(n_seg)
    # print(_index_info)
    # print(len(index_info))
    for v in index_info[n_seg:]:
        if v <= data_index:
            seq_index += 1
        else:
            break

    # index_info = np.array(index_info)
    # mask = np.array(index_info.__le__(data_index), dtype='int')
    # seq_index = int(mask.sum()) - 1
    # print(seq_index)
    inner_index = data_index - index_info[seq_index]
    return seq_index, inner_index

def bulid_rot(theta, rotation_axis):
    w = np.cos(theta * np.pi / 360)
    s = np.sin(theta * np.pi / 360)
    x = s * rotation_axis[0]
    y = s * rotation_axis[1]
    z = s * rotation_axis[2]

    q = quaternion.from_float_array([w, x, y, z])
    q = torch.Tensor([q.w, q.x, q.y, q.z]).float()
    rot = quaternion_to_rotation_matrix(q)

    return rot

def amass_read(path):
    data = torch.load(path)
    data = torch.cat(data, dim=0)
    # print(acc_t.shape)
    return data

def elbow_angle_caculate(joint_data:torch.Tensor, add_noise=False, encode=True):

    vec_1 = joint_data[:, 20] - joint_data[:, 18]
    vec_2 = joint_data[:, 18] - joint_data[:, 16]
    # print(torch.linalg.norm(vec_1, ord=2, dim=1).shape)
    vec_1 = vec_1 / torch.linalg.norm(vec_1, ord=2, dim=1).unsqueeze(1)
    vec_2 = vec_2 / torch.linalg.norm(vec_2, ord=2, dim=1).unsqueeze(1)
    l_elbow_angle = torch.arccos(torch.sum(vec_1 * vec_2, dim=1).unsqueeze(1))

    vec_1 = joint_data[:, 21] - joint_data[:, 19]
    vec_2 = joint_data[:, 19] - joint_data[:, 17]
    vec_1 = vec_1 / torch.linalg.norm(vec_1, ord=2, dim=1).unsqueeze(1)
    vec_2 = vec_2 / torch.linalg.norm(vec_2, ord=2, dim=1).unsqueeze(1)
    r_elbow_angle = torch.arccos(torch.sum(vec_1 * vec_2, dim=1).unsqueeze(1))

    if encode==False:
        return l_elbow_angle, r_elbow_angle

    if add_noise:
        l_elbow_angle, r_elbow_angle = add_gaussian_noise(l_elbow_angle, sigma=np.pi*10/180), \
                                       add_gaussian_noise(r_elbow_angle, sigma=np.pi*10/180)

    # l_elbow_angle = torch.cat([l_elbow_angle, l_elbow_angle*0, l_elbow_angle*0], dim=1)
    # l_elbow_angle = euler_angle_to_rotation_matrix(l_elbow_angle).view(len(l_elbow_angle), -1)[:, [4,5]]
    #
    # r_elbow_angle = torch.cat([r_elbow_angle, r_elbow_angle * 0, r_elbow_angle * 0], dim=1)
    # r_elbow_angle = euler_angle_to_rotation_matrix(r_elbow_angle).view(len(r_elbow_angle), -1)[:, [4,5]]

    l_elbow_angle = torch.cat([torch.sin(l_elbow_angle), torch.cos(l_elbow_angle)], dim=-1)
    r_elbow_angle = torch.cat([torch.sin(r_elbow_angle), torch.cos(r_elbow_angle)], dim=-1)

    print(l_elbow_angle[0])

    return l_elbow_angle, r_elbow_angle

def elbow_angle_process(angle):
    angle = angle * np.pi / 180
    l_elbow_angle = angle[:, :1]
    r_elbow_angle = angle[:, 1:]
    l_elbow_angle = torch.cat([torch.sin(l_elbow_angle), torch.cos(l_elbow_angle)], dim=-1)
    r_elbow_angle = torch.cat([torch.sin(r_elbow_angle), torch.cos(r_elbow_angle)], dim=-1)
    return l_elbow_angle, r_elbow_angle

def rotation_matrix_to_angle_1d(x:torch.Tensor):
    # print(x.shape)
    # x = rotation_matrix_to_axis_angle(x)

    x = axis_angle_to_quaternion(x)
    x = torch.arccos(x[:, [0]]) / np.pi
    print(x.shape)
    print(x.max())
    print(x.min())
    return x

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
        i = (index % self.data_len)
        data = [self.x[self.indexer[i]:self.indexer[i] + self.seq_len],
                self.y[self.indexer[i]:self.indexer[i] + self.seq_len]]
        if self.y2 is not None:
            data.append(self.y2[self.indexer[i]:self.indexer[i] + self.seq_len])
        return tuple(data)

    @staticmethod
    @timing
    def load_data(folder_path: str, use_elbow_angle=False, pose_type='r6d', syn=False, add_noise=True, syn_angle=False) -> dict:
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files
            s3_type: ['r6d', 'axis_angle']

        Returns:
            Dict of datas.
        """
        all_joint_num = len(index_pose)

        if syn:
            rot = amass_read(os.path.join(folder_path, 'syn_rot_on_garment.pt'))
            acc = amass_read(os.path.join(folder_path, 'syn_acc_on_garment.pt'))
        else:
            rot = amass_read(os.path.join(folder_path, 'vrot.pt'))
            acc = amass_read(os.path.join(folder_path, 'vacc.pt'))
        pose = amass_read(os.path.join(folder_path, 'pose.pt'))

        # pose转为r6d
        rot_dim = 3
        if pose_type == 'r6d':
            # 数据分2段处内存占用
            data_len = len(pose)
            # pose = pose.view(data_len * 24, 3)
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

        # 限制范围 防止异常值干扰
        acc = torch.clamp(acc, min=-60, max=60)
        acc = torch.cat((acc[:, :3] - acc[:, 3:], acc[:, 3:]), dim=1).bmm(rot[:, -1]) / 30

        # 转换为相对根节点的旋转
        rot = torch.cat((rot[:, 3:].transpose(2, 3).matmul(rot[:, :3]), rot[:, 3:]), dim=1)

        # 关节点空间坐标
        joint = amass_read(os.path.join(folder_path, 'joint.pt'))
        # 归一化
        joint = joint - joint[:, :1, :]
        #
        joint = joint.bmm(rot[:, -1])

        # rot转为r6d
        rot = rotation_matrix_to_r6d(rot.reshape(-1, 3, 3)).reshape(-1, 4, 6)

        if use_elbow_angle:
            if syn_angle:
                angle = torch.load(os.path.join(folder_path, 'syn_angle_on_garment.pt'))
                elbow_l_angle, elbow_r_angle = elbow_angle_process(angle)
            # 由于实际使用中角度传感器数值不稳定，因此对数值加入较大噪声训练
            # 这里有待改进，可以改成在训练过程中引入随机的响应程度，模拟实际使用中传感器完全弯曲/部分弯曲/不弯曲的情况
            else:
                elbow_l_angle, elbow_r_angle = elbow_angle_caculate(joint_data=joint, add_noise=add_noise, encode=True)
            x_s1 = torch.cat([acc.flatten(1), rot.flatten(1), elbow_l_angle, elbow_r_angle], dim=1)  # imu输入+关节角度

        else:
            x_s1 = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1)  # imu输入

        # pose_internal = pose[:, joint_set.internal_joint].reshape(len(pose), internal_joint_num * 6)  # s1输出的gt
        pose_upper_body = pose[:, index_pose].reshape(len(pose), all_joint_num * rot_dim)  # s2输出的gt
        # 加上手部2个节点 扣除根节点
        joint_upper_body = joint[:, index_joint].reshape(len(pose), (all_joint_num+2-1) * 3)

        return {'x_s1': x_s1,
                # 'x_s2': x_s2,
                # 'pose_internal': pose_internal,
                'joint_upper_body': joint_upper_body,
                'pose_all': pose_upper_body}

    def data_analysis(folder_path: str) -> dict:

        rot = torch.load(os.path.join(folder_path, 'vrot.pt'))
        acc = torch.load(os.path.join(folder_path, 'vacc.pt'))
        joint = torch.load(os.path.join(folder_path, 'joint.pt'))
        pose = torch.load(os.path.join(folder_path, 'pose.pt'))

        imu_idx = 3

        rot_x_list = []
        rot_y_list = []
        rot_z_list = []
        acc_list = []
        angle_left_list = []
        angle_left_v_list = []
        angle_right_list = []
        angle_right_v_list = []

        # pose转为r6d
        seg_num = len(pose)
        for seg_idx in tqdm(range(seg_num)):
            _rot = rot[seg_idx]
            _acc = acc[seg_idx]
            _joint = joint[seg_idx]
            _pose = pose[seg_idx]

            _acc = torch.cat((_acc[:, :3] - _acc[:, 3:], _acc[:, 3:]), dim=1).bmm(_rot[:, -1])
            # 转换为相对根节点的旋转
            _rot = torch.cat((_rot[:, 3:].transpose(2, 3).matmul(_rot[:, :3]), _rot[:, 3:]), dim=1)


            seg_length = len(_rot)
            _rot = rotation_matrix_to_euler_angle(_rot.reshape(seg_length, -1))
            _rot = _rot.reshape(-1, 4, 3)
            _rot = np.array(_rot[:, imu_idx])

            # print(_acc.shape)
            _acc = np.array(_acc[:, imu_idx])
            # print(_acc.shape)
            _acc = np.linalg.norm(_acc, axis=1)

            elbow_l_angle, elbow_r_angle = elbow_angle_caculate(joint_data=_joint, add_noise=False, encode=False)

            elbow_l_angle = np.array(elbow_l_angle) * 180 / np.pi
            elbow_r_angle = np.array(elbow_r_angle) * 180 / np.pi

            elbow_l_angle_v = elbow_l_angle[1:] - elbow_l_angle[:-1]
            elbow_r_angle_v = elbow_r_angle[1:] - elbow_r_angle[:-1]

            rot_x_list += _rot[:, 0].reshape(-1).tolist()
            rot_y_list += _rot[:, 1].reshape(-1).tolist()
            rot_z_list += _rot[:, 2].reshape(-1).tolist()
            acc_list += _acc.reshape(-1).tolist()
            angle_left_list += elbow_l_angle.reshape(-1).tolist()
            angle_left_v_list += elbow_l_angle_v.reshape(-1).tolist()
            angle_right_list += elbow_r_angle.reshape(-1).tolist()
            angle_right_v_list += elbow_r_angle_v.reshape(-1).tolist()

        rot_x = np.array(rot_x_list)* 180 / np.pi
        rot_y = np.array(rot_y_list)* 180 / np.pi
        rot_z = np.array(rot_z_list)* 180 / np.pi
        acc = np.array(acc_list)
        angle_left = np.array(angle_left_list)
        angle_left_v = np.array(angle_left_v_list) * 60
        angle_right = np.array(angle_right_list)
        angle_right_v = np.array(angle_right_v_list) * 60

        print(f'imu {imu_idx}')
        print('rot_x:', rot_x.min(), rot_x.max())
        print('rot_y:', rot_y.min(), rot_y.max())
        print('rot_z:', rot_z.min(), rot_z.max())
        print('acc:', acc.min(), acc.max(), np.percentile(acc, 25), np.percentile(acc, 50), np.percentile(acc, 99))
        print('angle_left:', angle_left.min(), angle_left.max())
        print('angle_left_v:', angle_left_v.min(), angle_left_v.max(), np.percentile(angle_left_v, 25), np.percentile(angle_left_v, 50), np.percentile(angle_left_v, 99))
        print('angle_right:', angle_right.min(), angle_right.max())
        print('angle_right_v:', angle_right_v.min(), angle_right_v.max(), np.percentile(angle_right_v, 25), np.percentile(angle_right_v, 50), np.percentile(angle_right_v, 99))


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
        i = (index % self.data_len)
        data = [self.x[self.indexer[i]:self.indexer[i] + self.seq_len], self.y[self.indexer[i]:self.indexer[i] + self.seq_len]]
        if self.y2 is not None:
            data.append(self.y2[self.indexer[i]:self.indexer[i] + self.seq_len])
        return tuple(data)

    @staticmethod
    @timing
    def load_data(folder_path='E:\H+\SensorData', use_elbow_angle=False, pose_type='r6d', type='all', angle_type = 'oangle', encode=False) -> dict:
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files
            s3_type: ['r6d', 'axis_angle']

        Returns:
            Dict of datas.
        """

        all_joint_num = len(index_pose)

        rot, acc, angle, joint, pose = [], [], [], [], []

        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                if type != 'all':
                    if dir_name.find(type) >= 0:
                        dir_path = os.path.join(root, dir_name)
                        print(f'loading {dir_name}')
                        rot.append(torch.load(os.path.join(dir_path, 'rot.pt')))
                        acc.append(torch.load(os.path.join(dir_path, 'acc.pt')))
                        angle.append(torch.load(os.path.join(dir_path, f'{angle_type}.pt')))
                        joint.append(torch.load(os.path.join(dir_path, 'joint.pt')).reshape(-1, 24, 3))
                        pose.append(torch.load(os.path.join(dir_path, 'pose.pt')))
                else:
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

        # pose转为r6d
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
        # 转换为相对根节点的旋转
        rot = torch.cat((rot[:, 3:].transpose(2, 3).matmul(rot[:, :3]), rot[:, 3:]), dim=1)
        # 归一化
        joint = joint - joint[:, :1, :]
        # 转到root坐标系
        joint = joint.bmm(rot[:, -1])
        # 转为r6d
        rot = rotation_matrix_to_r6d(rot.reshape(-1, 9)).reshape(-1, 4, 6)

        # angle = torch.clamp(angle, min=0, max=170)

        if use_elbow_angle:
            # elbow_l_angle, elbow_r_angle = elbow_angle_process(angle, l_a, l_b, r_a, r_b)
            if encode:
                elbow_l_angle, elbow_r_angle = elbow_angle_process(angle)
            else:
                elbow_l_angle, elbow_r_angle = angle[:, :1], angle[:, 1:]
            x_s1 = torch.cat([acc.flatten(1), rot.flatten(1), elbow_l_angle, elbow_r_angle], dim=1)  # imu输入+关节角度

        else:
            x_s1 = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1)  # imu输入

        # pose_internal = pose[:, joint_set.internal_joint].reshape(len(pose), internal_joint_num * 6)  # s1输出的gt

        # x_s2 = torch.cat((x_s1.flatten(1), pose_internal.flatten(1)), dim=1)  # s2的输入
        # pose_external = pose[:, joint_set.external_joint].reshape(len(pose), external_joint_num * 6)  # s2输出的gt
        pose_upper_body = pose[:, index_pose].reshape(len(pose), all_joint_num * rot_dim)  # s2输出的gt
        joint_upper_body = joint[:, index_joint].reshape(len(pose), (all_joint_num+2-1) * 3)

        return {'x_s1': x_s1,
                'joint_upper_body': joint_upper_body,
                'pose_all': pose_upper_body}


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
        i = (index % self.data_len)
        return self.x[self.indexer[i]], self.y[self.indexer[i]]

    @staticmethod
    @timing
    def load_data(folder_path: str, shuffle=False, normalization=True, clothes_imu_calibration=False, data_type='all', type='amass') -> dict:
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files
            s3_type: ['r6d', 'axis_angle']

        Returns:
            Dict of datas.
        """
        if type == 'amass':
            rot_bone = amass_read(os.path.join(folder_path, 'vrot.pt'))
            rot_imu = amass_read(os.path.join(folder_path, 'syn_rot_on_garment.pt'))
            acc_mesh = amass_read(os.path.join(folder_path, 'vacc.pt'))
            acc_imu = amass_read(os.path.join(folder_path, 'syn_acc_on_garment.pt'))
        else:
            rot_bone, rot_imu, acc_mesh, acc_imu = [], [], [], []

            for root, dirs, files in os.walk(folder_path):
                for dir_name in dirs:
                    if data_type != 'all':
                        if dir_name.find(data_type) >= 0:
                            dir_path = os.path.join(root, dir_name)
                            print(f'loading {dir_name}')
                            rot_bone.append(torch.load(os.path.join(dir_path, 'vrot.pt')))
                            rot_imu.append(torch.load(os.path.join(dir_path, 'rot.pt')))
                            acc_mesh.append(torch.load(os.path.join(dir_path, 'vacc.pt')))
                            acc_imu.append(torch.load(os.path.join(dir_path, 'acc.pt')))
                    else:
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

        tpose_clothes_v = obj_load_vertices(path='./T-Pose_garment.obj')
        tpose_rot, _ = imu_syn(tpose_clothes_v)
        device2bone = tpose_rot.transpose(-2,-1)

        if clothes_imu_calibration:
            rot_imu =rot_imu.matmul(device2bone)
        # print(device2bone)



        # rot_bone = torch.load(os.path.join(folder_path, 'vrot.pt'))
        # rot_imu = torch.load(os.path.join(folder_path, 'rot.pt'))
        # acc_mesh = torch.load(os.path.join(folder_path, 'vacc.pt'))
        # acc_imu = torch.load(os.path.join(folder_path, 'acc.pt'))

        data_len = len(rot_bone)

        # 防止异常值
        acc_mesh = torch.clamp(acc_mesh, min=-60, max=60)
        acc_imu = torch.clamp(acc_imu, min=-60, max=60)

        if normalization:
            # 转换为相对根节点加速度
            acc_mesh = torch.cat((acc_mesh[:, :3] - acc_mesh[:, 3:], acc_mesh[:, 3:]), dim=1).bmm(rot_bone[:, -1]) / 30
            acc_imu = torch.cat((acc_imu[:, :3] - acc_imu[:, 3:], acc_imu[:, 3:]), dim=1).bmm(rot_imu[:, -1]) / 30

        acc_mesh = acc_mesh.reshape(data_len, -1)
        acc_imu = acc_imu.reshape(data_len, -1)

        if normalization:
            # 转换为相对根节点的旋转
            rot_bone = torch.cat((rot_bone[:, 3:].transpose(2, 3).matmul(rot_bone[:, :3]), rot_bone[:, 3:]), dim=1)
            rot_imu = torch.cat((rot_imu[:, 3:].transpose(2, 3).matmul(rot_imu[:, :3]), rot_imu[:, 3:]), dim=1)

        # rot转为r6d
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

        return {'data_mesh': data_mesh,
                'data_garment': data_garment}



