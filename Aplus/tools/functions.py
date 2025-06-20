from torch.autograd import Function
import os
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
from typing import Any, Optional, Tuple
import torch
from articulate.evaluator import r6d_to_rotation_matrix, rotation_matrix_to_axis_angle
import articulate as art

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


# def plot_confusion_matrix(domain_predict, label, model_name, epoch):
#     """
#     绘制混淆矩阵并保存图像
#     Args:
#         domain_predict: 域预测结果
#         label: 数据label
#         name: 保存文件名称
#     """
#     path = f"./output/{model_name}"
#     if not os.path.exists(path):
#         os.mkdir(path)
#     domain_predict = np.concatenate([x.cpu().numpy() for x in domain_predict])
#     label = np.concatenate([x.cpu().numpy() for x in label])

#     cm = confusion_matrix(label, domain_predict)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title('Confusion Matrix')
#     plt.savefig(f'{path}/{epoch}.png')
#     plt.close()

def joint_caculate_elbow_angle(joint_data:torch.Tensor, encode=True):
    # input shape: [batch_size, seq_len, 33]
    batch_size = joint_data.shape[0]
    seq_len = joint_data.shape[1]
    joint_data = joint_data.reshape([batch_size * seq_len, -1, 3])

    vec_1 = joint_data[:, -2] - joint_data[:, -4]
    vec_2 = joint_data[:, -4] - joint_data[:, -6]
    norm_vec_1 = torch.linalg.norm(vec_1, ord=2, dim=1).unsqueeze(1)
    norm_vec_2 = torch.linalg.norm(vec_2, ord=2, dim=1).unsqueeze(1)
    vec_1 = vec_1 / torch.where(norm_vec_1 == 0, torch.tensor(1e-9).to(joint_data.device), norm_vec_1)
    vec_2 = vec_2 / torch.where(norm_vec_2 == 0, torch.tensor(1e-9).to(joint_data.device), norm_vec_2)
    vec = torch.sum(vec_1 * vec_2, dim=1).unsqueeze(1).clamp(-1, 1)
    l_elbow_angle = torch.arccos(vec)

    vec_1 = joint_data[:, -1] - joint_data[:, -3]
    vec_2 = joint_data[:, -3] - joint_data[:, -5]
    norm_vec_1 = torch.linalg.norm(vec_1, ord=2, dim=1).unsqueeze(1)
    norm_vec_2 = torch.linalg.norm(vec_2, ord=2, dim=1).unsqueeze(1)
    vec_1 = vec_1 / torch.where(norm_vec_1 == 0, torch.tensor(1e-9).to(joint_data.device), norm_vec_1)
    vec_2 = vec_2 / torch.where(norm_vec_2 == 0, torch.tensor(1e-9).to(joint_data.device), norm_vec_2)
    vec = torch.sum(vec_1 * vec_2, dim=1).unsqueeze(1).clamp(-1, 1)
    r_elbow_angle = torch.arccos(vec)

    if encode == False:
        return torch.cat([l_elbow_angle, r_elbow_angle], dim=-1).reshape([batch_size, seq_len, -1])

    l_elbow_angle = torch.cat([torch.sin(l_elbow_angle), torch.cos(l_elbow_angle)], dim=-1)
    r_elbow_angle = torch.cat([torch.sin(r_elbow_angle), torch.cos(r_elbow_angle)], dim=-1)

    return torch.cat([l_elbow_angle, r_elbow_angle], dim=-1).reshape([batch_size, seq_len, -1])



def pose_caculate_elbow_angle(pose_data:torch.Tensor, encode=True):
    # input shape: [batch_size, seq_len, 60]
    device = pose_data.get_device()
    batch_size = pose_data.shape[0]
    seq_len = pose_data.shape[1]
    index_joint=[3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21]
    index_pose=[0, 3, 6, 9, 13, 14, 16, 17, 18, 19]
    joint_num = len(index_pose)
    body_model = art.ParametricModel('E:\H+\Leizu4.1+数据集\smpl\smpl/SMPL_MALE.pkl', device=device)

    pose_data = pose_data.reshape(-1, 60)
    pose_data = pose_data.reshape(-1, 6)

    pose_data = r6d_to_rotation_matrix(pose_data).reshape(-1, joint_num, 3, 3)

    p_full_body = torch.eye(3).reshape(1,1,3,3).repeat(len(pose_data), 24, 1, 1).to(device)
    p_full_body[:, index_pose] = pose_data

    shape = torch.zeros(10).to(device)
    tran = torch.zeros(len(p_full_body), 3).to(device)

    p_grot, p_joint = body_model.forward_kinematics(p_full_body, shape, tran, calc_mesh=False)

    p_joint = p_joint[:, index_joint]
    p_joint = p_joint.reshape(-1,33)
    p_joint = p_joint.reshape([batch_size, seq_len, -1])
    return joint_caculate_elbow_angle(joint_data=p_joint, encode=encode)

import pickle
def export_pose(pose):
    pkl = {'body_pose': None}
    index_pose=[0, 3, 6, 9, 13, 14, 16, 17, 18, 19]
    pose = rotation_matrix_to_axis_angle(r6d_to_rotation_matrix(pose)).reshape(10, 3)
    p_full_body = torch.zeros((24, 3))
    p_full_body[index_pose] = pose
    # joint_index = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist']
    pkl['body_pose'] = np.array(p_full_body[1:])
    with open('ours.pkl', 'wb') as f:
        pickle.dump(pkl, f)