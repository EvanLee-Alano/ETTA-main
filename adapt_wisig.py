import os
import sys
import copy
import time
import torch
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from util.logger import get_logger
from util.etta import config_model, collect_params, set_random_weights
from data.dataloader_wisig import get_data, get_dataloader
from config.config_wisig import Config
from util.visualization import visualize
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
    parser.add_argument('-m', type=str, default='adapt', help='Training Mode.')
    parser.add_argument('-sd', type=int, required=True, help='Random seed.')
    parser.add_argument('-b', type=str, default='ours', help='Backbone network.')
    parser.add_argument('-s', type=int, required=True, help='Source domain info: day.')
    parser.add_argument('-t', type=int, required=True, help='Target domain info: day.')
    parser.add_argument('-l', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('-bs', type=int, default=128, help='Batch size.')
    parser.add_argument('-d', type=int, default=0, help='GUP device number.')
    args = parser.parse_args()
    return args


def adapt(model, dataloader, optimizer):
    st = time.time()
    model.train()

    losses = 0.
    correct = 0
    for data, target in dataloader:
        if torch.cuda.is_available():
            data = data.float().to(device)
            target = target.long().to(device)

        output = model(data)
        # energy-based
        loss = -(output[1].logsumexp(1).mean())
        logits = F.log_softmax(output[1], dim=-1)
        pred = torch.argmax(logits, dim=-1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # entropy-based
        #logits = output[1]
        #prob = F.softmax(logits, dim=1)
        #log_prob = F.log_softmax(logits, dim=1)
        #loss = -(prob * log_prob).sum(1).mean()
        #pred = torch.argmax(prob, dim=1, keepdim=True)
        #correct += pred.eq(target.view_as(pred)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
    loss_mean = losses / len(dataloader)
    acc = (correct / len(dataloader.dataset)) * 100
    et = time.time()
    return loss_mean, correct, acc, et - st


def test(model, dataloader):
    model.eval()
    correct = 0
    energys = 0.
    with torch.no_grad():
        for data, target in dataloader:
            if torch.cuda.is_available():
                data = data.float().to(device)
                target = target.long().to(device)

            output = model(data)
            energys += -(output[1].logsumexp(1).mean()).item()
            logits = F.log_softmax(output[1], dim=-1)
            pred = torch.argmax(logits, dim=-1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = (correct / len(dataloader.dataset)) * 100
        energy = energys / len(dataloader)
    return correct, acc, energy


def main():
    # dataloader
    x_train, x_valid, x_test, y_train, y_valid, y_test = get_data(config.dataset_path, config.target['d'])
    x_test = np.concatenate([x_train, x_valid, x_test], axis=0)
    y_test = np.concatenate([y_train, y_valid, y_test], axis=0)
    dataloader = get_dataloader(config.mode, x_test, y_test, config.batch_size)

    # model
    if config.backbone == 'scnn':
        from model.SCNN_model import SCNN as Model
    elif config.backbone == 'cnn':
        from model.CNN_model import CNN_lab as Model
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

    ckpt = torch.load(f"{config.root_dir}/train/day{config.source['d']}/ckpt.pth")
    model.load_state_dict(ckpt, strict=True)

    # primary test
    logger.info("Testing before TTA ...")
    logger.info(f"No. test samples: {len(dataloader.dataset)}")
    logger.info("-" * 100 + '\n')
    # visualize(model, dataloader, config.num_classes, device, logger, config.work_dir, "before_tta")
    test_correct, test_acc, test_energy = test(model, dataloader)
    logger.info(f"Testing finished! Energy : {test_energy:.4f} | ACC : {test_correct}/{len(dataloader.dataset)} ({test_acc:.3f}%)\n")
    logger.info("-" * 100 + '\n\n')

    # config model
    model = config_model(model).to(device)
    model_orig = copy.deepcopy(model)
    params, param_names = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=config.lr, betas=(0.9, 0.999), weight_decay=0.)

    # training
    logger.info("Start TTA ...")
    logger.info(f"No. test samples: {len(dataloader.dataset)}")
    logger.info("-" * 100 + '\n')
    total_time = 0.
    for step in range(1, config.adapt_steps + 1):
        logger.info(f"Current step: {step} / {config.adapt_steps}")
        adapt_loss, adapt_correct, adapt_acc, t = adapt(model, dataloader, optimizer)
        total_time += t
        logger.info(f"Adapting time : {t:.2f} seconds | Loss : {adapt_loss:.4f} | ACC : {adapt_correct}/{len(dataloader.dataset)} ({adapt_acc:.3f}%)")
        set_random_weights(model, model_orig, percentage=0.01, device=device)
        logger.info("Parameters randomly reset.\n")
        # visualize(model, dataloader, config.num_classes, device, logger, config.work_dir, f"after_step_{step}")
        logger.info("-" * 100 + "\n")
    avg_time = total_time / config.adapt_steps
    logger.info(f"Total {config.adapt_steps} steps, average time consumption : {avg_time:.2f} seconds.")
    logger.disabled = True



if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.d}"
    config = Config(args)
    config.print_config()

    set_seed(config.seed)
    device = config.device
    logger = config.logger

    main()
