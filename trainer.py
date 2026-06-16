from Aplus.tools.annotations import timing
from Aplus.runner import *
from articulate.evaluator import RotationErrorEvaluator, PerJointRotationErrorEvaluator
from articulate.math.angular import RotationRepresentation, r6d_to_rotation_matrix
import os
import numpy as np
import torch
from articulate.evaluator import mean_vector_length
import articulate as art
from tqdm import tqdm
from Aplus.tools.functions import pose_caculate_elbow_angle
import random
from Aplus.tools.data_visualize import *


ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
SMPL_MODEL_PATH = os.path.join(ASSETS_DIR, 'SMPL_MALE.pkl')


def r6d_global_y_rot(r, angle):
    sin_x = np.sin(angle)
    cos_x = np.cos(angle)
    r = r.reshape(-1, 6)
    r = torch.cat([
        cos_x * r[:, [0]] + sin_x * r[:, [2]],
        r[:, [1]],
        -sin_x * r[:, [0]] + cos_x * r[:, [2]],
        cos_x * r[:, [3]] + sin_x * r[:, [5]],
        r[:, [4]],
        -sin_x * r[:, [3]] + cos_x * r[:, [5]],
    ], dim=-1)
    return r


def VAE_loss_function(x_hat, x, mu, log_var, kld_a=0):
    mse_loss = nn.MSELoss()
    mse = mse_loss(x_hat, x)
    kld = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1.0 - log_var)
    loss = mse + kld_a * kld
    return loss, mse, kld


class PoserTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, data, optimizer, batch_size, loss_func, initializer=None, AE=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.data = data
        self.epoch = 0
        self.batch_size = batch_size
        self.log_manager = LogManager(items=['epoch', 'loss_elbow', 'loss_all', 'loss_joint', 'elbow_err', 'all_err'])
        self.checkpoint = None
        self.AE = AE
        self.loss_func = nn.MSELoss()
        self.loss_func_elbow = nn.L1Loss()

    @timing
    def run(self, epoch, data_shuffle=True, evaluator=None):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle, drop_last=False)
        device = self.get_model_device()
        avg_loss_elbow = DataMeter()
        avg_loss_angular = DataMeter()
        avg_loss_joint = DataMeter()

        for e in range(epoch):
            avg_loss_elbow.reset()
            avg_loss_angular.reset()
            self.model.train()

            for i, data in enumerate(data_loader):
                if i > int(len(data_loader) / 10):
                    break
                self.optimizer.zero_grad()

                x, y, y2 = data
                seq_len = x.shape[1]
                x = x.to(device)
                y = y.to(device)
                y2 = y2.to(device)
                angle = pose_caculate_elbow_angle(y[:, seq_len // 4:], False).to(device)

                if self.AE is not None:
                    x[:, :, :36] = self.AE.secondary_motion_gen(x[:, :, :36], eta=0.5)
                joint, elbow, all = self.model(x)
                pose_hat = torch.cat([all.detach()[:, :, :48], elbow.clone()], dim=-1)
                angle_hat = pose_caculate_elbow_angle(pose_hat[:, seq_len // 4:].clone(), False).to(device)

                loss_joint = self.loss_func(joint[:, seq_len // 4:], y2[:, seq_len // 4:])
                loss_angle = self.loss_func_elbow(angle_hat, angle)
                loss_axis = self.loss_func(elbow[:, seq_len // 4:], y[:, seq_len // 4:, 48:])
                loss_all = self.loss_func(all[:, seq_len // 4:], y[:, seq_len // 4:])
                a = 0.1
                loss_ang = loss_axis + loss_all
                loss = a * loss_angle + loss_ang + loss_joint * 4
                loss.backward()

                self.optimizer.step()

                avg_loss_elbow.update(value=loss_angle.item(), n_sample=len(y))
                avg_loss_angular.update(value=loss_ang.item(), n_sample=len(y))
                avg_loss_joint.update(value=loss_joint.item(), n_sample=len(y))

                print(
                    f'iter {i} | {len(self.data) // self.batch_size} \t '
                    f'loss_angle:{loss_angle} \t loss_axis:{loss_axis} \t '
                    f'loss_pos:{loss_joint} \t loss_other:{loss_all}',
                    end='\n',
                )

            loss_elbow = avg_loss_elbow.get_avg()
            loss_ang = avg_loss_angular.get_avg()
            loss_joint = avg_loss_joint.get_avg()
            self.epoch += 1
            print('')

            if evaluator is not None:
                elbow_err, all_err = evaluator.run()
            else:
                elbow_err, all_err = -1, -1

            self.log_manager.update({
                'epoch': self.epoch,
                'loss_elbow': loss_elbow,
                'loss_all': loss_ang,
                'loss_joint': loss_joint,
                'elbow_err': elbow_err,
                'all_err': all_err,
            })
            self.log_manager.print_latest()


class PoseEvaluatorWithStd:
    def __init__(self, rot_type='r6d', index_joint=[3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21],
                 index_pose=[0, 3, 6, 9, 13, 14, 16, 17, 18, 19]):
        self.index_joint = index_joint
        self.index_pose = index_pose
        self.body_model = art.ParametricModel(SMPL_MODEL_PATH)

        if rot_type == 'r6d':
            rep = RotationRepresentation.R6D
        elif rot_type == 'axis_angle':
            rep = RotationRepresentation.AXIS_ANGLE
        self.rot_type = rot_type
        self.rot_err_evaluator = RotationErrorEvaluator(rep=rep)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=rep)

    @torch.no_grad()
    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        p = p.cpu()
        t = t.cpu()
        p_all = p
        t_all = t

        per_joint_err = []
        position_err = []
        per_position_err = []
        joints = []

        for i in tqdm(range(len(p_all))):
            p = p_all[i]
            t = t_all[i]
            joint_num = len(self.index_pose)
            mpjre = self.per_joint_rot_err_evaluator(p.unsqueeze(0), t.unsqueeze(0), joint_num=joint_num)
            per_joint_err.append(mpjre.unsqueeze(0))
            if self.rot_type == 'r6d':
                p = p.reshape(-1, 6)
                t = t.reshape(-1, 6)

                p = r6d_to_rotation_matrix(p).reshape(-1, joint_num, 3, 3)
                t = r6d_to_rotation_matrix(t).reshape(-1, joint_num, 3, 3)

                p_full_body = torch.eye(3).reshape(1, 1, 3, 3).repeat(len(p), 24, 1, 1)
                p_full_body[:, self.index_pose] = p

                t_full_body = torch.eye(3).reshape(1, 1, 3, 3).repeat(len(p), 24, 1, 1)
                t_full_body[:, self.index_pose] = t

            shape = torch.zeros(10)
            tran = torch.zeros(len(p_full_body), 3)

            _, p_joint = self.body_model.forward_kinematics(p_full_body, shape, tran, calc_mesh=False)
            _, t_joint = self.body_model.forward_kinematics(t_full_body, shape, tran, calc_mesh=False)

            p_joint = p_joint[:, self.index_joint]
            t_joint = t_joint[:, self.index_joint]
            joints.append(p_joint)

            mjpe = torch.cat([
                mean_vector_length(p_joint[:, i, :] - t_joint[:, i, :]).unsqueeze(0)
                for i in range(len(self.index_joint))
            ], dim=0)

            position_err.append(mjpe.mean().detach())
            per_position_err.append(mjpe.detach().cpu().unsqueeze(0))

        per_joint_err = torch.cat(per_joint_err, dim=0)
        position_err = np.array(position_err, dtype=float) * 100
        per_position_err = torch.cat(per_position_err, dim=0) * 100
        joints = torch.cat(joints, dim=0)
        jitter = ((joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3])).norm(dim=2) * 100

        print(f"Angular Error: {format(per_joint_err.norm(dim=-1).mean(), '.2f')} ± {format(per_joint_err.norm(dim=-1).mean(dim=-1).std(), '.2f')}")
        print(f"Elbow Angular Error: {format(per_joint_err[:, -2:,].norm(dim=-1).mean(), '.2f')} ± {format(per_joint_err[:, -2:,].norm(dim=-1).std(), '.2f')}")
        print(f"Positional Error: {format(position_err.mean(), '.2f')} ± {format(position_err.std(), '.2f')}")
        print(f"Jitter: {format(jitter.mean(), '.2f')}")
        return per_joint_err.norm(dim=-1).mean(dim=-1), per_joint_err[:, -2:,].norm(dim=-1).mean(dim=-1), position_err, jitter.mean(dim=-1)


class VAETrainer(BaseTrainer):
    def __init__(self, model: nn.Module, data, optimizer, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.epoch = 0
        self.batch_size = batch_size
        self.log_manager = LogManager(items=['epoch', 'loss_train', 'loss_eval'])
        self.checkpoint = None
        self.loss_func = None

    @timing
    def run(self, epoch, data_shuffle=True, evaluator=None):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle, drop_last=False)
        device = self.get_model_device()
        avg_meter_gap = DataMeter()

        for e in range(epoch):
            optimizer = self.optimizer
            avg_meter_gap.reset()
            self.model.train()

            for i, data in enumerate(tqdm(data_loader)):
                optimizer.zero_grad()
                loose_data, tight_data = data
                loose_data = loose_data.to(device)
                tight_data = tight_data.to(device)

                loose_acc, loose_rot = loose_data[:, :12], loose_data[:, 12:]
                tight_acc, tight_rot = tight_data[:, :12], tight_data[:, 12:]

                x = random.uniform(-np.pi / 2, np.pi / 2)
                loose_rot = torch.cat([loose_rot[:, :-6], r6d_global_y_rot(r=loose_rot[:, -6:], angle=x)], dim=-1)
                tight_rot = torch.cat([tight_rot[:, :-6], r6d_global_y_rot(r=tight_rot[:, -6:], angle=x)], dim=-1)

                loose_data = torch.cat([loose_acc, loose_rot], dim=-1)
                tight_data = torch.cat([tight_acc, tight_rot], dim=-1)
                loose_gap = loose_data - tight_data

                x_hat, mu, log_var = self.model(loose_gap)
                loss, _, _ = VAE_loss_function(x_hat=x_hat, x=loose_gap, mu=mu, log_var=log_var, kld_a=1e-8)
                loss.backward()
                optimizer.step()

                avg_meter_gap.update(loss, n_sample=len(loose_data))

            loss_train = avg_meter_gap.get_avg()
            self.epoch += 1

            if evaluator is not None:
                loss_eval = evaluator.run(epoch=self.epoch)
            else:
                loss_eval = -1

            self.log_manager.update({'epoch': self.epoch, 'loss_train': loss_train, 'loss_eval': loss_eval})
            self.log_manager.print_latest()


class VAEEvaluator(BaseEvaluator):
    def __init__(self, model, data, loss_func, batch_size):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.batch_size = batch_size

    @torch.no_grad()
    def run(self, device=None, noise_eta=None, epoch=0):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=False, drop_last=False)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        avg_meter_gap = DataMeter()
        self.model.to(device)
        self.model.eval()
        for i, data in enumerate(tqdm(data_loader)):
            loose_data, tight_data = data
            loose_data = loose_data.to(device)
            tight_data = tight_data.to(device)

            loose_gap = loose_data - tight_data
            x_hat, mu, log_var = self.model(loose_gap)

            loss_func = nn.L1Loss()
            loss = loss_func(x_hat, loose_gap)
            avg_meter_gap.update(loss, n_sample=len(loose_data))

        return avg_meter_gap.get_avg()


class DiffusionTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, data, optimizer, batch_size, loss_func):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.data = data
        self.epoch = 0
        self.batch_size = batch_size
        self.log_manager = LogManager(items=['epoch', 'loss_train', 'loss_eval'])
        self.checkpoint = None

    @timing
    def run(self, epoch, data_shuffle=True, evaluator=None):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle, drop_last=False)
        device = self.get_model_device()
        avg_meter_diffusion = DataMeter()

        for e in range(epoch):
            optimizer = self.optimizer
            avg_meter_diffusion.reset()
            self.model.train()

            for i, data in enumerate(tqdm(data_loader)):
                loose_data, tight_data = data
                loose_data = loose_data.to(device)
                tight_data = tight_data.to(device)

                loose_acc, loose_rot = loose_data[:, :12], loose_data[:, 12:]
                tight_acc, tight_rot = tight_data[:, :12], tight_data[:, 12:]

                x = random.uniform(-np.pi / 2, np.pi / 2)
                loose_rot = torch.cat([loose_rot[:, :-6], r6d_global_y_rot(r=loose_rot[:, -6:], angle=x)], dim=-1)
                tight_rot = torch.cat([tight_rot[:, :-6], r6d_global_y_rot(r=tight_rot[:, -6:], angle=x)], dim=-1)

                loose_data = torch.cat([loose_acc, loose_rot], dim=-1)
                tight_data = torch.cat([tight_acc, tight_rot], dim=-1)
                loose_gap = loose_data - tight_data

                with torch.no_grad():
                    mu, log_var = self.model.encode(loose_gap)
                    sampled_z = self.model.reparameterization(mu, log_var)

                optimizer.zero_grad()

                sampled_z = sampled_z.unsqueeze(1)
                noise_scheduler = self.model.diffusion.scheduler
                noise = torch.randn(sampled_z.shape, device=device)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (sampled_z.shape[0],), device=device).long()
                noisy_encoded = noise_scheduler.add_noise(sampled_z, noise, timesteps)
                pred_noise = self.model.diffusion(noisy_encoded, timesteps).sample
                loss = self.loss_func(pred_noise, noise)

                loss.backward()
                optimizer.step()
                avg_meter_diffusion.update(value=loss.cpu(), n_sample=len(loose_data))

            loss = avg_meter_diffusion.get_avg()
            self.epoch += 1

            if evaluator is not None:
                loss_eval = evaluator.run(epoch=self.epoch)
            else:
                loss_eval = -1

            self.log_manager.update({'epoch': self.epoch, 'loss_train': loss, 'loss_eval': loss_eval})
            self.log_manager.print_latest()


class DiffusionEvaluator(BaseEvaluator):
    def __init__(self, model, data, loss_func, batch_size):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.batch_size = batch_size

    @torch.no_grad()
    def run(self, device=None, epoch=0):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        avg_meter_diffusion = DataMeter()
        self.model.eval()
        tight_data_visualize, loose_data_visualize = None, None
        for i, data in enumerate(tqdm(data_loader)):
            loose_data, tight_data = data
            loose_data = loose_data.to(device)
            tight_data = tight_data.to(device)

            if i == 6:
                tight_data_visualize = tight_data[:128]
                loose_data_visualize = loose_data[:128]

            loose_gap = loose_data - tight_data

            with torch.no_grad():
                mu, log_var = self.model.encode(loose_gap)
                sampled_z = self.model.reparameterization(mu, log_var)

            sampled_z = sampled_z.unsqueeze(1)
            noise_scheduler = self.model.diffusion.scheduler
            noise = torch.randn_like(sampled_z)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (sampled_z.shape[0],), device=device).long()
            noisy_encoded = noise_scheduler.add_noise(sampled_z, noise, timesteps)
            pred_noise = self.model.diffusion(noisy_encoded, timesteps, return_dict=False)[0]
            loss = self.loss_func(pred_noise, noise)
            avg_meter_diffusion.update(loss, n_sample=len(loose_data))

        loss_eval = avg_meter_diffusion.get_avg()

        dimensionReducer = DimensionReducer(36, 2)
        with torch.no_grad():
            tight_data_visualize = tight_data_visualize.unsqueeze(0)
            loose_gap = loose_data_visualize - tight_data_visualize
            loose_gap_recon, _, _ = self.model(loose_gap)
            loose_data_recon = tight_data_visualize + loose_gap_recon
            loose_gen_1 = self.model.secondary_motion_gen(tight_data_visualize, eta=1)
            loose_gen_2 = self.model.secondary_motion_gen(tight_data_visualize, eta=2)
            loose_gen_3 = self.model.secondary_motion_gen(tight_data_visualize, eta=3)

        data_dict = {
            "loose": dimensionReducer.fit_transform(loose_data_visualize),
            "tight": dimensionReducer.fit_transform(tight_data_visualize.squeeze(0)),
            "gen_1": dimensionReducer.fit_transform(loose_gen_1.squeeze(0)),
            "gen_2": dimensionReducer.fit_transform(loose_gen_2.squeeze(0)),
            "gen_3": dimensionReducer.fit_transform(loose_gen_3.squeeze(0)),
            "loose_recon": dimensionReducer.fit_transform(loose_data_recon.squeeze(0)),
        }
        plot_scatter_2d_from_dict(data_dict=data_dict, epoch=epoch)

        return loss_eval
