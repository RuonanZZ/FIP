from data import *
from trainer import *
from model import *

def trainVAE(model, data_train, data_test, num_epochs=10):
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-3)
    model.encoder.train()
    model.decoder.train()
    model.diffusion.eval()  # 冻结 diffusion model

    trainer = VAETrainer(model=model, data=data_train, optimizer=optimizer, batch_size=512)
    evaluator = VAEEvaluator.from_trainner(trainer, data_eval=data_test)

    model_name='VAE'
    folder_path=f'./checkpoint/{model_name}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    for i in range(num_epochs):
        trainer.run(epoch=1, evaluator=evaluator, data_shuffle=True)
        trainer.save(folder_path=folder_path, model_name=model_name)
        trainer.log_export(f'./log/{model_name}.xlsx')

def trainDiffusion(model, data_train, data_test, criterion, num_epochs=10):
    optimizer = torch.optim.Adam(model.diffusion.parameters(), lr=1e-3)
    model.encoder.eval()  # 冻结 encoder
    model.decoder.eval()  # 冻结 decoder
    model.diffusion.train()

    trainer = DiffusionTrainer(model=model, data=data_train, optimizer=optimizer, batch_size=512, loss_func=criterion)
    evaluator = DiffusionEvaluator.from_trainner(trainer, data_eval=data_test)

    model_name='DiffusionVAE'
    folder_path=f'./checkpoint/{model_name}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    trainer.restore('./checkpoint/VAE_best.pth', load_optimizer=False, load_log_manager=False)

    for i in range(num_epochs):
        trainer.run(epoch=1, evaluator=evaluator, data_shuffle=True)
        trainer.save(folder_path=folder_path, model_name=model_name)
        trainer.log_export(f'./log/{model_name}.xlsx')


stage = 2

train_split, test_split = 0.95, 0.05

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = SynPairedIMUData.load_data(folder_path='E:\DATA\processed_AMASS', shuffle=False, clothes_imu_calibration=True)
data_len = len(dataset['data_mesh'])
train_size, test_size = int(data_len*train_split), int(data_len*test_split)

model = DiffusionVAE(feat_dim=12+24, h_dim=32, z_dim=12).to(device)

data_train = SynPairedIMUData(x=dataset['data_garment'][:train_size], y=dataset['data_mesh'][:train_size], shuffle=True)
data_test  = SynPairedIMUData(x=dataset['data_garment'][-test_size:], y=dataset['data_mesh'][-test_size:], shuffle=False)

criterion = nn.MSELoss()

if stage==1:
    trainVAE(model=model, data_train=data_train, data_test=data_test)
else:
    trainDiffusion(model=model, data_train=data_train, data_test=data_test, criterion=criterion)