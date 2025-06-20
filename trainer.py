from Aplus.tools.annotations import timing
from Aplus.runner import *
from articulate.evaluator import RotationErrorEvaluator, PerJointRotationErrorEvaluator, PerJointAccErrorEvaluator
from articulate.math.angular import RotationRepresentation, quaternion_to_rotation_matrix, r6d_to_rotation_matrix, rotation_matrix_to_axis_angle
import numpy as np
import torch
from articulate.evaluator import mean_vector_length
import articulate as art
from tqdm import tqdm
from Aplus.tools.functions import pose_caculate_elbow_angle
import random
from Aplus.tools.data_visualize import *

def r6d_global_y_rot(r, angle):
    sin_x = np.sin(angle)
    cos_x = np.cos(angle)
    r = r.reshape(-1, 6)
    r = torch.cat([cos_x*r[:, [0]]+sin_x*r[:, [2]], r[:, [1]], -sin_x*r[:, [0]]+cos_x*r[:, [2]],
                   cos_x*r[:, [3]]+sin_x*r[:, [5]], r[:, [4]], -sin_x*r[:, [3]]+cos_x*r[:, [5]]], dim=-1)

    # rot = [[ cos_x, sin_x, 0],
    #        [-sin_x, cos_x, 0],
    #        [0,      0,     1]]
    # rot = torch.tensor(rot).float()
    # print(r.shape)
    return r

def VAE_loss_function(x_hat, x, mu, log_var, kld_a=0):
        """
        Calculate the loss. Note that the loss includes two parts.
        :param x_hat:
        :param x:
        :param mu:
        :param log_var:
        :return: total loss, BCE and KLD of our model
        """
        # 1. the reconstruction loss.
        # We regard the MNIST as binary classification
        mse_loss = nn.MSELoss()
        mse = mse_loss(x_hat, x)

        # 2. KL-divergence
        # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
        kld = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

        # 3. total loss
        loss = mse + kld_a * kld
        return loss, mse, kld

class PoserEvaluator(BaseEvaluator):
    def __init__(self, model, data, loss_func, batch_size, rot_type='r6d'):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.batch_size = batch_size
        if rot_type == 'r6d':
            rep = RotationRepresentation.R6D
        elif rot_type == 'axis_angle':
            rep = RotationRepresentation.AXIS_ANGLE
        self.rot_err_evaluator = RotationErrorEvaluator(rep=rep)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=rep)
        self.loss_func = nn.MSELoss()

    @torch.no_grad()
    def run(self, device=None):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=False,
                                       drop_last=False)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        avg_elbow_err_left = DataMeter()
        avg_elbow_err_right = DataMeter()
        avg_all_err = DataMeter()
        self.model.eval()
        for i, data in enumerate(data_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            seq_len = x.shape[1]
            _, elbow, all = self.model(x)

            # 计算角度误差
            elbow_err_left = self.rot_err_evaluator(p=elbow[:, -1, :6], t=y[:, -1, -12:-6]).cpu()
            elbow_err_right = self.rot_err_evaluator(p=elbow[:, -1, 6:], t=y[:, -1, -6:]).cpu()
            all_err = self.rot_err_evaluator(p=torch.cat([all[:, -1, :48],elbow[:, -1]], dim=-1), t=y[:, -1]).cpu()
            avg_elbow_err_left.update(value=elbow_err_left, n_sample=len(y))
            avg_elbow_err_right.update(value=elbow_err_right, n_sample=len(y))
            avg_all_err.update(value=all_err, n_sample=len(y))


        elbow_err_left = avg_elbow_err_left.get_avg()
        elbow_err_right = avg_elbow_err_right.get_avg()
        all_err = avg_all_err.get_avg()

        return (elbow_err_left.norm(dim=-1)+elbow_err_right.norm(dim=-1))/2, all_err.norm(dim=-1)

    @classmethod
    def from_trainner(cls, trainner, data_eval, rot_type='r6d'):
        return cls(model=trainner.model, loss_func=trainner.loss_func, batch_size=trainner.batch_size, data=data_eval, rot_type=rot_type)

class PoserTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, data, optimizer, batch_size, loss_func, initializer=None, AE=None):
        """
        Used for manage training process.
        Args:
            model: Your model.
            data: Dataset object. You can build dataset via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
            batch_size: /
            loss_func: /
        """
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

        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle,
                                 drop_last=False)

        # 获取当前模型所在device
        device = self.get_model_device()

        # AverageMeter用于计算整个epoch的loss
        avg_loss_elbow = DataMeter()
        avg_loss_angular = DataMeter()
        avg_loss_joint = DataMeter()

        for e in range(epoch):

            # AverageMeter需要在每个epoch开始时置0
            avg_loss_elbow.reset()
            avg_loss_angular.reset()

            self.model.train()

            for i, data in enumerate(data_loader):
            # for i, data in enumerate(tqdm(data_loader)):
                if i > int(len(data_loader)/10):
                    break
                self.optimizer.zero_grad()

                x, y, y2 = data
                batch_size, seq_len = x.shape[0], x.shape[1]
                x = x.to(device)
                y = y.to(device)
                y2 = y2.to(device)
                angle = pose_caculate_elbow_angle(y[:, seq_len//4:], False).to(device)

                if self.AE is not None:
                    x[:,:,:36] = self.AE.secondary_motion_gen(x[:,:,:36], eta=0.5)
                joint, elbow, all = self.model(x)
                pose_hat = torch.cat([all.detach()[:,:,:48], elbow.clone()], dim=-1)
                angle_hat = pose_caculate_elbow_angle(pose_hat[:, seq_len//4:].clone(), False).to(device)

                loss_joint = self.loss_func(joint[:, seq_len//4:], y2[:, seq_len//4:])

                loss_angle = self.loss_func_elbow(angle_hat, angle)
                loss_axis = self.loss_func(elbow[:, seq_len//4:], y[:, seq_len//4:, 48:])
                loss_all = self.loss_func(all[:, seq_len//4:], y[:, seq_len//4:])
                a = 0.1
                # loss_elbow = a*loss_angle + loss_axis
                loss_ang = loss_axis+loss_all
                # loss_elbow.backward()
                # loss_all.backward()
                loss = a*loss_angle + loss_ang + loss_joint *4
                loss.backward()

                self.optimizer.step()

                # 每个batch记录一次
                avg_loss_elbow.update(value=loss_angle.item(), n_sample=len(y))
                avg_loss_angular.update(value=loss_ang.item(), n_sample=len(y))
                avg_loss_joint.update(value=loss_joint.item(), n_sample=len(y))

                # print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')
                print(f'iter {i} | {len(self.data) // self.batch_size} \t loss_angle:{loss_angle} \t loss_axis:{loss_axis} \t loss_pos:{loss_joint} \t loss_other:{loss_all}', end='\n')

            # 获取整个epoch的loss
            loss_elbow = avg_loss_elbow.get_avg()
            loss_ang = avg_loss_angular.get_avg()
            loss_joint = avg_loss_joint.get_avg()
            self.epoch += 1
            print('')

            if evaluator is not None:
                elbow_err, all_err = evaluator.run()
            else:
                elbow_err, all_err = -1, -1

            # 记录当前epoch的训练集 & 验证集loss
            self.log_manager.update(
                {'epoch': self.epoch, 'loss_elbow': loss_elbow, 'loss_all': loss_ang, 'loss_joint': loss_joint, 'elbow_err': elbow_err, 'all_err': all_err})

            # 打印最新一个epoch的训练记录
            self.log_manager.print_latest()

class PoseEvaluatorWithStd:
    def __init__(self, rot_type='r6d', index_joint=[3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21],
                 index_pose=[0, 3, 6, 9, 13, 14, 16, 17, 18, 19]):
        self.index_joint = index_joint
        self.index_pose = index_pose
        self.body_model = art.ParametricModel('E:\H+\Leizu4.1+数据集\smpl\smpl/SMPL_MALE.pkl')

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

        joint_err = []
        per_joint_err = []
        position_err = []
        per_position_err = []
        joints = []

        for i in tqdm(range(len(p_all))):
            p = p_all[i]
            t = t_all[i]
            joint_num = len(self.index_pose)
            mjre = self.rot_err_evaluator(p, t)
            joint_err.append(mjre)
            mpjre = self.per_joint_rot_err_evaluator(p.unsqueeze(0), t.unsqueeze(0), joint_num=joint_num)
            per_joint_err.append(mpjre.unsqueeze(0))
            if self.rot_type == 'r6d':
                p = p.reshape(-1, 6)
                t = t.reshape(-1, 6)

                p = r6d_to_rotation_matrix(p).reshape(-1, joint_num, 3, 3)
                # p = rotation_matrix_to_axis_angle(p).reshape(-1, joint_num, 3)

                t = r6d_to_rotation_matrix(t).reshape(-1, joint_num, 3, 3)
                # t = rotation_matrix_to_axis_angle(t).reshape(-1, joint_num, 3)

                p_full_body = torch.eye(3).reshape(1,1,3,3).repeat(len(p), 24, 1, 1)
                p_full_body[:, self.index_pose] = p

                t_full_body = torch.eye(3).reshape(1, 1, 3, 3).repeat(len(p), 24, 1, 1)
                t_full_body[:, self.index_pose] = t


            shape = torch.zeros(10)
            tran = torch.zeros(len(p_full_body), 3)

            p_grot, p_joint = self.body_model.forward_kinematics(p_full_body, shape, tran, calc_mesh=False)
            t_grot, t_joint = self.body_model.forward_kinematics(t_full_body, shape, tran, calc_mesh=False)

            p_joint = p_joint[:, self.index_joint]
            t_joint = t_joint[:, self.index_joint]
            joints.append(p_joint)

            mjpe = torch.cat([mean_vector_length(p_joint[:, i, :] - t_joint[:, i, :]).unsqueeze(0) for i in range(len(self.index_joint))], dim=0)
            # print(mjpe.mean())

            position_err.append(mjpe.mean().detach())
            per_position_err.append(mjpe.detach().cpu().unsqueeze(0))


        # joint_err = np.array(joint_err, dtype=float)
        joint_err = torch.cat(joint_err, dim=0)
        per_joint_err = torch.cat(per_joint_err, dim=0)
        position_err = np.array(position_err, dtype=float) *100
        # print(per_position_err)
        per_position_err = torch.cat(per_position_err, dim=0) *100
        joints = torch.cat(joints, dim=0)
        # print(joints.shape)
        jitter = ((joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3])).norm(dim=2)*100

        # print(per_position_err.shape)
        print(f'肘部各肘误差: {per_joint_err[:,-2:,].mean(dim=0)}')
        print(f'肘部各肘误差: {per_joint_err[:,-2:,].mean(dim=0).mean(dim=0)}')
        print(f'各关节角度误差: {per_joint_err.norm(dim=-1).mean(dim=0)}')
        print(f'各关节位置误差: {per_position_err.mean(dim=0)}')
        print(f"平均角度误差: {format(per_joint_err.norm(dim=-1).mean(), '.2f')} ± {format(per_joint_err.norm(dim=-1).mean(dim=-1).std(), '.2f')}")
        print(f"平均肘部误差: {format(per_joint_err[:,-2:,].norm(dim=-1).mean(), '.2f')} ± {format(per_joint_err[:,-2:,].norm(dim=-1).std(), '.2f')}")
        print(f"平均关节位置误差: {format(position_err.mean(), '.2f')} ± {format(position_err.std(), '.2f')}")
        print(f"Jitters: {format(jitter.mean(), '.2f')}")
        return per_joint_err.norm(dim=-1).mean(dim=-1), per_joint_err[:,-2:,].norm(dim=-1).mean(dim=-1), position_err, jitter.mean(dim=-1)

class VAETrainer(BaseTrainer):
    def __init__(self, model:nn.Module, data, optimizer, batch_size):
        """
        Used for manage training process.
        Args:
            model: Your model.
            data: Dataset object. You can build dataset via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
            batch_size: /
            loss_func: /
        """
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.epoch = 0
        self.batch_size = batch_size
        self.log_manager = LogManager(items=['epoch', 'loss_train', 'loss_eval'])
        self.checkpoint = None
        self.rot_err_evaluator = RotationErrorEvaluator(rep=RotationRepresentation.R6D)
        self.loss_func = None

    

    @timing
    def run(self, epoch, data_shuffle=True, evaluator=None):

        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle,
                                       drop_last=False)

        # 获取当前模型所在device
        device = self.get_model_device()

        # AverageMeter用于计算整个epoch的loss
        avg_meter_gap = DataMeter()
        avg_meter_loose = DataMeter()


        for e in range(epoch):
            optimizer = self.optimizer

            # AverageMeter需要在每个epoch开始时置0
            avg_meter_gap.reset()

            self.model.train()

            for i, data in enumerate(tqdm(data_loader)):
                optimizer.zero_grad()
                loose_data, tight_data = data
                loose_data = loose_data.to(device)
                tight_data = tight_data.to(device)

                loose_acc, loose_rot = loose_data[:, :12], loose_data[:, 12:]
                tight_acc, tight_rot = tight_data[:, :12], tight_data[:, 12:]

                # 根节点绕y(up)轴随机旋转
                x = random.uniform(-np.pi/2, np.pi/2)

                loose_rot = torch.cat([loose_rot[:, :-6], r6d_global_y_rot(r=loose_rot[:, -6:], angle=x)], dim=-1)
                tight_rot = torch.cat([tight_rot[:, :-6], r6d_global_y_rot(r=tight_rot[:, -6:], angle=x)], dim=-1)

                loose_data = torch.cat([loose_acc, loose_rot], dim=-1)
                tight_data = torch.cat([tight_acc, tight_rot], dim=-1)

                loose_gap = loose_data - tight_data

                x_hat, mu, log_var = self.model(loose_gap)

                # print(f"mean:{loose_gap.mean()}, std:{loose_gap.std()}, mean:{x_hat.mean()}, std:{x_hat.std()}")

                loss, mse, kld = VAE_loss_function(x_hat=x_hat, x=loose_gap, mu=mu, log_var=log_var, kld_a=1e-8)

                loss.backward()

                optimizer.step()

                # 每个batch记录一次
                avg_meter_gap.update(loss, n_sample=len(loose_data))
                # print(f'iter {i} | {len(self.data) // self.batch_size}, mse:{mse}, kld:{kld}', end='')


                # print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')

            # 获取整个epoch的loss
            loss_train = avg_meter_gap.get_avg()
            self.epoch += 1
            # print(loss_tight, loss_loose, loss_distribution)

            if evaluator is not None:
                loss_eval = evaluator.run(epoch=self.epoch)
            else:
                loss_eval = -1

            # 记录当前epoch的训练集 & 验证集loss
            self.log_manager.update({'epoch': self.epoch, 'loss_train': loss_train, 'loss_eval': loss_eval})


            self.log_manager.print_latest()

class VAEEvaluator(BaseEvaluator):
    def __init__(self, model, data, loss_func, batch_size):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.rot_err_evaluator = RotationErrorEvaluator(rep=RotationRepresentation.R6D)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=RotationRepresentation.R6D)
        self.per_joint_acc_err_evaluator = PerJointAccErrorEvaluator()

    @torch.no_grad()
    def run(self, device=None, noise_eta=None,  epoch=0):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=False,
                                       drop_last=False)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        # AverageMeter用于计算整个epoch的loss
        avg_meter_gap = DataMeter()
        self.model.eval()
        # self.model.dis_normalizer.eval()
        for i, data in enumerate(tqdm(data_loader)):

            loose_data, tight_data = data
            loose_data = loose_data.to(device)
            tight_data = tight_data.to(device)

            loose_gap = loose_data - tight_data

            x_hat, mu, log_var = self.model(loose_gap)

            loss_func = nn.L1Loss()
            loss = loss_func(x_hat, loose_gap)
            # loss, mse, kld = VAE_loss_function(x_hat=x_hat, x=loose_gap, mu=mu, log_var=log_var, kld_a=1e-5)

            # 每个batch记录一次
            avg_meter_gap.update(loss, n_sample=len(loose_data))

            if i == 6:
                tight_data_visualize = tight_data[:128]
                loose_data_visualize = loose_data[:128]
                loose_data_recon = (tight_data + x_hat)[:128]

            # print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')
            # print(f'iter {i} | {len(self.data) // self.batch_size}, mse:{mse}, kld:{kld}', end='')

            
        # print(f'\riter {i} | {len(self.data) // self.batch_size}, mse:{mse}, kld:{kld}, loose_gap:{loose_gap.mean()}, x_hat:{x_hat.mean()}', end='')

        # 获取整个epoch的loss
        loss_eval = avg_meter_gap.get_avg()

        # dimensionReducer = DimensionReducer(36, 2, method='tsne')
        # data_dict = {
        #     "loose": dimensionReducer.fit_transform(loose_data_visualize),
        #     "tight": dimensionReducer.fit_transform(tight_data_visualize),
        #     "loose_recon": dimensionReducer.fit_transform(loose_data_recon),
        # }
        # plot_scatter_2d_from_dict(data_dict=data_dict, epoch=f"VAE_{epoch}")

        return loss_eval

class DiffusionTrainer(BaseTrainer):
    def __init__(self, model:nn.Module, data, optimizer, batch_size, loss_func):
        """
        Used for manage training process.
        Args:
            model: Your model.
            data: Dataset object. You can build dataset via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
            batch_size: /
            loss_func: /
        """
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

        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle,
                                       drop_last=False)

        # 获取当前模型所在device
        device = self.get_model_device()

        # AverageMeter用于计算整个epoch的loss
        avg_meter_diffusion = DataMeter()

        for e in range(epoch):
            optimizer = self.optimizer

            # AverageMeter需要在每个epoch开始时置0
            avg_meter_diffusion.reset()

            self.model.train()

            for i, data in enumerate(tqdm(data_loader)):
                
                loose_data, tight_data = data
                loose_data = loose_data.to(device)
                tight_data = tight_data.to(device)

                loose_acc, loose_rot = loose_data[:, :12], loose_data[:, 12:]
                tight_acc, tight_rot = tight_data[:, :12], tight_data[:, 12:]

                # 根节点绕y(up)轴随机旋转
                x = random.uniform(-np.pi/2, np.pi/2)

                loose_rot = torch.cat([loose_rot[:, :-6], r6d_global_y_rot(r=loose_rot[:, -6:], angle=x)], dim=-1)
                tight_rot = torch.cat([tight_rot[:, :-6], r6d_global_y_rot(r=tight_rot[:, -6:], angle=x)], dim=-1)

                loose_data = torch.cat([loose_acc, loose_rot], dim=-1)
                tight_data = torch.cat([tight_acc, tight_rot], dim=-1)

                loose_gap = loose_data - tight_data

                with torch.no_grad():
                    # encoder
                    mu, log_var = self.model.encode(loose_gap)
                    # reparameterization trick
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

                # 每个batch记录一次
                avg_meter_diffusion.update(value=loss.cpu(), n_sample=len(loose_data))

                # print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')

            # 获取整个epoch的loss
            loss = avg_meter_diffusion.get_avg()
            self.epoch += 1
            # print(loss_tight, loss_loose, loss_distribution)

            if evaluator is not None:
                loss_eval = evaluator.run(epoch=self.epoch)
            else:
                loss_eval = -1

            # 记录当前epoch的训练集 & 验证集loss
            self.log_manager.update({'epoch': self.epoch, 'loss_train': loss, 'loss_eval':loss_eval})


            self.log_manager.print_latest()

class DiffusionEvaluator(BaseEvaluator):
    def __init__(self, model, data, loss_func, batch_size):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.rot_err_evaluator = RotationErrorEvaluator(rep=RotationRepresentation.R6D)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=RotationRepresentation.R6D)
        self.per_joint_acc_err_evaluator = PerJointAccErrorEvaluator()

    @torch.no_grad()
    def run(self, device=None, epoch=0):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=False,
                                       drop_last=True)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        avg_meter_diffusion = DataMeter()
        self.model.eval()
        tight_data_visualize, loose_data_visualize = None, None
        # self.model.dis_normalizer.eval()
        for i, data in enumerate(tqdm(data_loader)):

            loose_data, tight_data = data
            loose_data = loose_data.to(device)
            tight_data = tight_data.to(device)

            if i == 6:
                tight_data_visualize = tight_data[:128]
                loose_data_visualize = loose_data[:128]

            loose_gap = loose_data - tight_data

            with torch.no_grad():
                # encoder
                mu, log_var = self.model.encode(loose_gap)
                # reparameterization trick
                sampled_z = self.model.reparameterization(mu, log_var)
            
            sampled_z = sampled_z.unsqueeze(1)
            noise_scheduler = self.model.diffusion.scheduler
            noise = torch.randn_like(sampled_z)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (sampled_z.shape[0],), device=device).long()
            
            noisy_encoded = noise_scheduler.add_noise(sampled_z, noise, timesteps)

            pred_noise = self.model.diffusion(noisy_encoded, timesteps, return_dict=False)[0]

            loss = self.loss_func(pred_noise, noise)

            # 每个batch记录一次
            avg_meter_diffusion.update(loss, n_sample=len(loose_data))

            # print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')

        # 获取整个epoch的loss
        loss_eval = avg_meter_diffusion.get_avg()

        # 可视化生成结果
        dimensionReducer = DimensionReducer(36, 2)
        with torch.no_grad():
            tight_data_visualize = tight_data_visualize.unsqueeze(0)
            loose_data_visualize = loose_data_visualize
            loose_gap = loose_data_visualize - tight_data_visualize
            loose_gap_recon, _, _ = self.model(loose_gap)
            loose_data_recon = tight_data_visualize + loose_gap_recon
            loose_gen_1 = self.model.secondary_motion_gen(tight_data_visualize, eta=1)
            loose_gen_2 = self.model.secondary_motion_gen(tight_data_visualize, eta=2)
            loose_gen_3 = self.model.secondary_motion_gen(tight_data_visualize, eta=3)
            # print(f"{tight_data_visualize.mean()}, {tight_data_visualize.std()}, {loose_data_visualize.mean()}, {loose_data_visualize.std()}, {loose_gen_1.mean()}, {loose_gen_1.std()}, {loose_gen_2.mean()}, {loose_gen_2.std()}, {loose_gen_3.mean()}, {loose_gen_3.std()}")
            # tight_data_recon = self.model.decode(self.model.encode(tight_data_visualize))

        # print(loose_gen_3.shape)
        data_dict = {
            "loose": dimensionReducer.fit_transform(loose_data_visualize),
            "tight": dimensionReducer.fit_transform(tight_data_visualize.squeeze(0)),
            "gen_1": dimensionReducer.fit_transform(loose_gen_1.squeeze(0)),
            "gen_2": dimensionReducer.fit_transform(loose_gen_2.squeeze(0)),
            "gen_3": dimensionReducer.fit_transform(loose_gen_3.squeeze(0)),
            "loose_recon": dimensionReducer.fit_transform(loose_data_recon.squeeze(0)),
            # "tight_recon": dimensionReducer.fit_transform(tight_data_recon.squeeze(0)),
        }
        plot_scatter_2d_from_dict(data_dict=data_dict, epoch=epoch)

        return loss_eval

class PoserWithoutFlexTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, data, optimizer, batch_size, loss_func, initializer=None, AE=None):
        """
        Used for manage training process.
        Args:
            model: Your model.
            data: Dataset object. You can build dataset via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
            batch_size: /
            loss_func: /
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.data = data
        self.epoch = 0
        self.batch_size = batch_size
        self.log_manager = LogManager(items=['epoch', 'loss_ang', 'loss_joint', 'ang_err'])
        self.checkpoint = None
        self.AE = AE
        self.loss_func = nn.MSELoss()
        self.loss_func_elbow = nn.L1Loss()

    @timing
    def run(self, epoch, data_shuffle=True, evaluator=None):

        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle,
                                 drop_last=False)

        # 获取当前模型所在device
        device = self.get_model_device()

        # AverageMeter用于计算整个epoch的loss
        avg_loss_angular = DataMeter()
        avg_loss_joint = DataMeter()

        for e in range(epoch):

            # AverageMeter需要在每个epoch开始时置0
            avg_loss_joint.reset()
            avg_loss_angular.reset()

            self.model.train()

            for i, data in enumerate(data_loader):
            # for i, data in enumerate(tqdm(data_loader)):
                if i > int(len(data_loader)/10):
                    break
                self.optimizer.zero_grad()

                x, y, y2 = data
                batch_size, seq_len = x.shape[0], x.shape[1]
                x = x.to(device)
                y = y.to(device)
                y2 = y2.to(device)

                if self.AE is not None:
                    x[:,:,:36] = self.AE.secondary_motion_gen(x[:,:,:36], eta=0.5)
                joint, pose = self.model(x)

                loss_joint = self.loss_func(joint[:, seq_len//4:], y2[:, seq_len//4:])

                loss_pose = self.loss_func(pose[:, seq_len//4:], y[:, seq_len//4:])

                loss = loss_pose + loss_joint *4
                loss.backward()

                self.optimizer.step()

                # 每个batch记录一次
                avg_loss_angular.update(value=loss_pose.item(), n_sample=len(y))
                avg_loss_joint.update(value=loss_joint.item(), n_sample=len(y))

                # print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')
                # print(f'iter {i} | {len(self.data) // self.batch_size} \t loss_angle:{loss_angle} \t loss_axis:{loss_axis} \t loss_pos:{loss_joint} \t loss_other:{loss_all}', end='\n')

            # 获取整个epoch的loss
            loss_ang = avg_loss_angular.get_avg()
            loss_joint = avg_loss_joint.get_avg()
            self.epoch += 1
            print('')

            if evaluator is not None:
                ang_err = evaluator.run()
            else:
                ang_err = -1, -1

            # 记录当前epoch的训练集 & 验证集loss
            self.log_manager.update(
                {'epoch': self.epoch, 'loss_ang': loss_ang, 'loss_joint': loss_joint, 'ang_err': ang_err})

            # 打印最新一个epoch的训练记录
            self.log_manager.print_latest()

class PoserWithoutFlexEvaluator(BaseEvaluator):
    def __init__(self, model, data, loss_func, batch_size, rot_type='r6d'):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.batch_size = batch_size
        if rot_type == 'r6d':
            rep = RotationRepresentation.R6D
        elif rot_type == 'axis_angle':
            rep = RotationRepresentation.AXIS_ANGLE
        self.rot_err_evaluator = RotationErrorEvaluator(rep=rep)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=rep)
        self.loss_func = nn.MSELoss()

    @torch.no_grad()
    def run(self, device=None):
        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=False,
                                       drop_last=False)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(device)
        avg_ang_err = DataMeter()
        self.model.eval()
        for i, data in enumerate(data_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            seq_len = x.shape[1]
            _, pose = self.model(x)

            # 计算角度误差
            ang_err = self.rot_err_evaluator(p=pose[:, -1], t=y[:, -1]).cpu()
            avg_ang_err.update(value=ang_err, n_sample=len(y))


        ang_err = avg_ang_err.get_avg()

        return ang_err.norm(dim=-1)

    @classmethod
    def from_trainner(cls, trainner, data_eval, rot_type='r6d'):
        return cls(model=trainner.model, loss_func=trainner.loss_func, batch_size=trainner.batch_size, data=data_eval, rot_type=rot_type)


