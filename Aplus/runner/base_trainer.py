from Aplus.utils import LogManager
from Aplus.utils import CheckPoint
import torch
from torch import nn
from torch.utils.data import DataLoader
from Aplus.utils import DataMeter
import matplotlib.pyplot as plt


class BaseTrainer:
    def __init__(self, model: nn.Module, data, optimizer, batch_size, loss_func):
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

    def save(self, folder_path, model_name=None):
        if self.checkpoint is None:
            self.checkpoint = CheckPoint(model=self.model, optimizer=self.optimizer,
                                         log_manager=self.log_manager)
        print(f'saving checkpoint ...', end='')
        self.checkpoint.save(save_folder_path=folder_path, epoch=self.epoch, model_name=model_name)
        print('done')

    def restore(self, checkpoint_path, load_optimizer=True, load_log_manager=True):
        checkpoint_dict = CheckPoint.load(file_path=checkpoint_path)
        self.model.load_state_dict(checkpoint_dict['model'])
        if load_optimizer:
            # Restore each optimizer separately when multiple optimizers are stored.
            if isinstance(checkpoint_dict['optimizer'], list):
                for i, optim in self.optimizer:
                    optim.load_state_dict(checkpoint_dict['optimizer'][i])
            else:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        print(f'training continue from epoch {self.epoch}')
        if load_log_manager:
            self.epoch = checkpoint_dict['epoch']
            self.log_manager.load_data(data=checkpoint_dict['log'])
            self.log_manager.print_latest()


    def log_export(self, path):
        """
        Export training log.
        :param path: e.g. './log.xlsx'
        :return: None
        """
        self.log_manager.to_excel(path)

    def get_model_device(self):
        return next(self.model.parameters()).device

    def run(self, epoch, data_shuffle=True, evaluator=None, verbose=False):

        data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle,
                                       drop_last=False)

        # Get the device where the current model lives.
        device = self.get_model_device()

        # Track the average loss over the full epoch.
        avg_meter_loss = DataMeter()

        for e in range(epoch):

            # Reset the meter at the beginning of each epoch.
            avg_meter_loss.reset()
            self.model.train()
            for i, data in enumerate(data_loader):
                if isinstance(self.optimizer, list):
                    for optim in self.optimizer:
                        optim.zero_grad()
                else:
                    self.optimizer.zero_grad()

                x, y = data
                x = x.to(device)
                y = y.to(device)
                y_hat = self.model(x)

                loss = self.loss_func(y_hat, y)
                loss.backward()

                if isinstance(self.optimizer, list):
                    for optim in self.optimizer:
                        optim.step()
                else:
                    self.optimizer.step()

                # Update the running loss once per batch.
                avg_meter_loss.update(value=loss.item(), n_sample=len(y))
                if verbose:
                    print(f'\riter {i} | {len(self.data) // self.batch_size}', end='')

            # Compute the average training loss for this epoch.
            loss_train = avg_meter_loss.get_avg()
            self.epoch += 1

            if evaluator is not None:
                loss_eval = evaluator.run()
            else:
                loss_eval = -1

            # Record the training and validation loss of this epoch.
            self.log_manager.update({'epoch': self.epoch, 'loss_train': loss_train, 'loss_eval': loss_eval})

            # Print the latest training record.
            self.log_manager.print_latest()








