import torch
from Aplus.models import *
from diffusers import UNet1DModel
from diffusers import DDIMScheduler
from diffusers import DDIMPipeline
from articulate.math.general import normalize_tensor, linear_interpolation_batch
from articulate.math.angular import rotation_matrix_to_axis_angle_torch

class Poser(BaseModel):
    def __init__(self):
        super(Poser, self).__init__()
        self.lstm = EasyLSTM(n_input=36, n_hidden=128, n_output=33, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2)
        self.elbow_predictor = EasyLSTM(n_input=36+4+33, n_hidden=128, n_output=12, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2)
        self.all_predictor = EasyLSTM(n_input=36+33, n_hidden=128, n_output=60, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2)

    def forward(self, x, *args):
        if len(args) > 0:
            h_s1, c_s1, h_s2, c_s2, h_s3, c_s3 = args
            joint, h_s1, c_s1 = self.lstm(x[:, :, :36], h_s1, c_s1)
            elbow, h_s2, c_s2 = self.elbow_predictor(torch.cat([x, joint.detach()], dim=-1), h_s2, c_s2)
            all, h_s3, c_s3 = self.all_predictor(torch.cat([x[:, :, :36], joint], dim=-1), h_s3, c_s3)

        else:
            joint = self.lstm(x[:, :, :36])
            elbow = self.elbow_predictor(torch.cat([x, joint.detach()], dim=-1))
            all = self.all_predictor(torch.cat([x[:, :, :36], joint], dim=-1))

        if len(args) > 0:
            return joint, elbow, all, h_s1, c_s1, h_s2, c_s2, h_s3, c_s3
        return joint, elbow, all

class Poser_without_flex(BaseModel):
    def __init__(self):
        super(Poser_without_flex, self).__init__()
        self.lstm = EasyLSTM(n_input=36, n_hidden=128, n_output=33, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2)
        self.all_predictor = EasyLSTM(n_input=36+33, n_hidden=128, n_output=60, n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0.2)

    def forward(self, x, *args):
        if len(args) > 0:
            h_s1, c_s1, h_s2, c_s2 = args
            joint, h_s1, c_s1 = self.lstm(x, h_s1, c_s1)
            all, h_s2, c_s2 = self.all_predictor(torch.cat([x, joint], dim=-1), h_s2, c_s2)

        else:
            joint = self.lstm(x)
            all = self.all_predictor(torch.cat([x, joint], dim=-1))

        if len(args) > 0:
            return joint, all, h_s1, c_s1, h_s2, c_s2
        return joint, all

class DiffusionModel(nn.Module):
    def __init__(self, encode_dim):
        super(DiffusionModel, self).__init__()
        self.model = UNet1DModel(sample_size=encode_dim, 
                                 in_channels=1, 
                                 out_channels=1,
                                 down_block_types=("DownBlock1D",),
                                 up_block_types=("UpBlock1D",),
                                 block_out_channels=(32,))
        self.scheduler = DDIMScheduler(num_train_timesteps=1000)

    def forward(self, x, timestep, return_dict=True):
        return self.model(x, timestep, return_dict)
    
    def gen(self, x, eta=1):

        pipeline = DDIMPipeline(unet=self.model, scheduler=self.scheduler)
        recon_semo = pipeline(
        batch_size=x.shape[0],
        generator=torch.Generator(device='cuda').manual_seed(0),
        )
        recon_semo = recon_semo.repeat(1, x.shape[1], 1)
        std = recon_semo.std()

        recon_semo = recon_semo
        return recon_semo

class DiffusionVAE(BaseModel):
    def __init__(self, feat_dim, h_dim, z_dim=12):
        super(DiffusionVAE, self).__init__()
        act_fun = 'tanh'
        self.encoder = nn.Module()
        self.encoder.dnn = EasyDNN(n_input=feat_dim, n_hiddens=[128, 64], n_output=h_dim, act_func=act_fun)
        self.encoder.fc1 = nn.Linear(h_dim, z_dim)
        self.encoder.fc2 = nn.Linear(h_dim, z_dim)
        
        self.decoder = nn.Module()
        self.decoder.fc1 = nn.Linear(z_dim, h_dim)
        self.decoder.dnn = EasyDNN(n_input=h_dim, n_hiddens=[64, 128], n_output=feat_dim, act_func=act_fun)
        self.diffusion = DiffusionModel(z_dim)

    def encode(self, x):
        x = self.encoder.dnn(x)
        mu = self.encoder.fc1(x)
        log_var = self.encoder.fc2(x)
        return mu, log_var
    
    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    
    def decode(self, x, norm=False):
        x = torch.relu(self.decoder.fc1(x))
        x = self.decoder.dnn(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decode(sampled_z)
        return x_hat, mu, log_var

    def _r6d_norm(self, x, rot_num=4):
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], rot_num, 6)
        result = []
        for i in range(rot_num):
            column0 = normalize_tensor(x[:, :, i, 0:3])
            column1 = normalize_tensor(x[:, :, i, 3:6] - (column0 * x[:, :, i, 3:6]).sum(dim=-1, keepdim=True) * column0)
            result.append(torch.cat([column0, column1], dim=-1))
        x = torch.cat(result, dim=-1)
        return x

    @torch.no_grad()
    def secondary_motion_gen(self, x, eta=1, acc_gen=True, mask=None):
        recon_semo = self.diffusion.gen(x=x, eta=eta)
        recon_semo = self.decode(recon_semo)

        if x.shape[1]!=1:
            bias_shift_1 = torch.randn(size=x[:, 0, :].shape).float().to(x.device)
            bias_shift_2 = torch.randn(size=x[:, 0, :].shape).float().to(x.device)
            bias_shift = linear_interpolation_batch(bias_shift_1, bias_shift_2, target_length=x.shape[1])
            recon_semo = (recon_semo + bias_shift) * eta

        return x + recon_semo
