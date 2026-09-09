import os
import torch
from util.logger import get_logger


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


class Config:
    def __init__(self, args):
        # about experiment
        self.seed = args.sd
        self.mode = args.m
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # about data
        self.dataset_path = '../../../Datasets'
        self.dataset_name = args.dn
        self.source = {'d': args.s}
        if self.mode == 'adapt':
            self.target = {'d': args.t}
        self.num_classes = 6

        # about model
        self.backbone = args.b

        # about training
        self.lr = args.l
        self.batch_size = args.bs
        if self.mode == 'train':
            self.train_epochs = args.e
        else:
            self.adapt_steps = 1

        # about save
        self.root_dir = makedirs("./result/{}_{}".format(self.dataset_name, self.backbone))
        if self.mode == 'train':
            self.work_dir = makedirs(
                "{}/train/day{}".format(self.root_dir, self.source['d']))
        else:
            self.work_dir = makedirs(
                "{}/adapt/etta/day{}_day{}".format(self.root_dir, self.source['d'], self.target['d']))
        self.logger = get_logger(self.mode, self.work_dir)

    def print_config(self):
        self.logger.info('=' * 100)
        self.logger.info('Config')
        self.logger.info('-' * 100)
        for key, value in self.__dict__.items():
            self.logger.info(f"{key} : {value}")
        self.logger.info('=' * 100 + "\n\n")
