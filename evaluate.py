import torch
from model import *
from data import *
from trainer import PoseEvaluatorWithStd
from articulate.math import *
from Aplus.tools.functions import *

import os

seq_len = 128
use_elbow_angle = True
data_type = 'all'
evaluator = PoseEvaluatorWithStd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
sensor_data = SensorData.load_data(use_elbow_angle=use_elbow_angle, type=data_type, angle_type='suda_ori', encode=True)
data_test = SensorData(x=sensor_data['x_s1'],
                       y=sensor_data['pose_all'],
                       seq_len=seq_len)
input = data_test.x.unsqueeze(0).to(device)
gt = data_test.y.to(device)

# load model
model = Poser().to(device)
model.restore('./checkpoint/Poser_Diffusion_best.pth')
model.eval()

# predict and evaluate
_, elbow, all = model(input)
elbow_aa = rotation_matrix_to_axis_angle(r6d_to_rotation_matrix(elbow)).reshape(-1,2,3)
pred_aa = rotation_matrix_to_axis_angle(r6d_to_rotation_matrix(all)).reshape(-1,10,3)
pred_aa[:,-2:,-2:] = elbow_aa[:,:,-2:]
pred = rotation_matrix_to_r6d(axis_angle_to_rotation_matrix(pred_aa)).reshape(-1,60)
ang, elb, pos, jitter = evaluator(p=pred, t=gt)