import torch.nn as nn
import torch
from model import *
from data import *
from trainer import *

seq_len = 128
use_elbow_angle = True
batch_size = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 数据集准备
train_data = AmassData.load_data(folder_path='E:\DATA\processed_AMASS', use_elbow_angle=use_elbow_angle, add_noise=True)

data_train = AmassData(x=train_data['x_s1'],
                       y=train_data['pose_all'],
                       y2=train_data['joint_upper_body'],
                       seq_len=seq_len)

# model
model = Poser().to(device)

AE_model = DiffusionVAE(feat_dim=12+24, h_dim=32).to(device)
AE_model.restore(checkpoint_path='./checkpoint/DiffusionVAE_best.pth')
AE_model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
criterion = nn.MSELoss()


trainner = PoserTrainer(model=model, data=data_train, optimizer=optimizer, batch_size=batch_size, loss_func=criterion,
                        AE=AE_model)

model_name = f'Poser_Diffusion'
folder_path=f'./checkpoint/{model_name}'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

for i in range(10):
    trainner.run(epoch=1, evaluator=None, data_shuffle=True)
    trainner.save(folder_path=folder_path, model_name=model_name)
    trainner.log_export(f'./log/{model_name}.xlsx')

