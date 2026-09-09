import os
import sys
import torch
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from util.logger import get_logger
# from thop import profile, clever_format
from data.dataloader_wifi import get_data, get_dataloader
from config.config_wifi import Config
from scipy.io import savemat

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='etta',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', type=str, default='train', help='Training mode.')
    parser.add_argument('-e', type=int, default='500', help='Training epochs')
    parser.add_argument('-b', type=str, default='ours', help='Backbone network.')
    parser.add_argument('-s', type=int, nargs=2, required=True, help='Source domain info: S, ft.')
    parser.add_argument('-t', type=int, nargs=2, default=None, help='Target domain info: S, ft.')
    parser.add_argument('-l', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('-bs', type=int, default=128, help='Batch size.')
    parser.add_argument('-d', type=int, default=0, help='GUP device number.')
    args = parser.parse_args()
    return args


def train(model, dataloader, optimizer, scheduler):
    model.train()
    losses = 0.
    correct = 0
    energys = 0.
    for data, target in dataloader:
        if torch.cuda.is_available():
            data = data.float().to(device)
            target = target.long().to(device)

        output = model(data)
        energy = -(output[1].logsumexp(1).mean())
        logits = F.log_softmax(output[1], dim=-1)
        pred = torch.argmax(logits, dim=-1, keepdim=True)

        loss = F.nll_loss(logits, target)
        losses += loss.item()
        energys += energy.item()
        correct += pred.eq(target.view_as(pred)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss_mean = losses / len(dataloader)
    energy_mean = energys / len(dataloader)
    acc = (correct / len(dataloader.dataset)) * 100
    logger.info(f"Train | Loss : {loss_mean:.6f} | Energy : {energy_mean:.6f} | ACC : {correct}/{len(dataloader.dataset)} ({acc:.3f}%)")


def valid(model, dataloader):
    model.eval()
    losses = 0.
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            if torch.cuda.is_available():
                data = data.float().to(device)
                target = target.long().to(device)

            output = model(data)
            logits = F.log_softmax(output[1], dim=-1)
            loss = F.nll_loss(logits, target)
            pred = torch.argmax(logits, dim=-1, keepdim=True)

            losses += loss.item()
            correct += pred.eq(target.view_as(pred)).sum().item()
        loss_mean = losses / len(dataloader)
        acc = (correct / len(dataloader.dataset)) * 100
    logger.info(f"Valid | Loss : {loss_mean:.6f} | ACC : {correct}/{len(dataloader.dataset)} ({acc:.3f}%)")
    return loss_mean


def test(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            if torch.cuda.is_available():
                data = data.float().to(device)
                target = target.long().to(device)

            output = model(data)
            logits = F.log_softmax(output[1], dim=-1)
            pred = torch.argmax(logits, dim=-1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = (correct / len(dataloader.dataset)) * 100
    return acc, correct


def main():
    # dataloader
    x_train, x_valid, x_test, y_train, y_valid, y_test = get_data(config.dataset_path, config.source['s'],
                                                                  config.source['ft'])
    train_loader = get_dataloader(config.mode, x_train, y_train, config.batch_size)
    valid_loader = get_dataloader(config.mode, x_valid, y_valid, config.batch_size)
    test_loader = get_dataloader(config.mode, x_test, y_test, config.batch_size)

    # model
    if config.backbone == 'ours':
        from model.Ours_model import OursModel as Model
    elif config.backbone == 'cnn':
        from model.CNN_model import CNN_outdoor as Model
    elif config.backbone == 'oracle':
        from model.ORACLE_CNN_model import ORACLE_CNN as Model
    elif config.backbone == 'light':
        from model.LightCNN_model import LightCNN as Model
    elif config.backbone == 'cvcnn':
        from model.CVCNN_model import CVCNN as Model
    elif config.backbone == 'cvsrn':
        from model.CVSRN_model import CVSRN as Model
    elif config.backbone == 'xcept':
        from model.XceptionTime_model import XceptionTime as Model
    elif config.backbone == 'drsn':
        from model.DRSN_model import DRSN_CS34 as Model
    elif config.backbone == 'mscnn':
        from model.MSCNN_model import MSCNN as Model
    else:
        logger.info('Backbone not supported.')
        sys.exit()
    model = Model(2, config.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # training
    logger.info("Start training ...")
    logger.info(f"No. train samples: {len(train_loader.dataset)}")
    logger.info(f"No. valid samples: {len(valid_loader.dataset)}")
    logger.info("-" * 100 + '\n')

    save_info = {
        'best_epoch': -1,
        'best_loss': float('inf'),
    }
    for epoch in range(1, config.train_epochs + 1):
        logger.info(f"Current Epoch: {epoch} / {config.train_epochs} | Learning Rate : {optimizer.param_groups[0]['lr']}")
        train(model, train_loader, optimizer, scheduler)
        valid_loss = valid(model, valid_loader)
        if valid_loss < save_info['best_loss']:
            torch.save(model.state_dict(), config.work_dir + "/ckpt.pth")
            logger.info(
                f"      | Loss dropped from {save_info['best_loss']:.6f} to {valid_loss:.6f}, model has been saved!\n"
            )
            save_info['best_epoch'] = epoch
            save_info['best_loss'] = valid_loss
        else:
            logger.info("      | Loss didn't drop!\n")
        logger.info(
            f"best epoch: {save_info['best_epoch']}, best loss: {save_info['best_loss']:.6f}"
        )
        logger.info("-" * 100 + "\n")

    logger.info(
        f"Training finished! Best Epoch : {save_info['best_epoch']} | Best Loss : {save_info['best_loss']:.6f}\n\n"
    )

    # test
    logger.info("Start testing ...")
    logger.info(f"No. test samples: {len(test_loader.dataset)}")
    logger.info("-" * 100 + '\n')

    # load model weight
    ckpt = torch.load(config.work_dir + '/ckpt.pth')
    model.load_state_dict(ckpt, strict=True)

    test_acc, correct = test(model, test_loader)
    logger.info(f"Testing finished! ACC : {correct}/{len(test_loader.dataset)} ({test_acc:.3f}%) *****\n\n")
    logger.disabled = True


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.d}"
    config = Config(args)
    config.print_config()

    set_seed(config.seed)
    logger = config.logger
    device = config.device

    main()
