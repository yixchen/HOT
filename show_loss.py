# System libs
import argparse
from glob import glob
import pickle
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
# Our libs
from hot.config import cfg
from hot.utils import setup_logger


STAT_FILE = 'data/stats.pickle'
with open(STAT_FILE, 'rb') as f:
    stats = pickle.load(f)
number_by_cate = stats['contact_number'][1:]
ratio_by_cate = number_by_cate / np.sum(number_by_cate)


def main(cfg):
    print(cfg.DIR)
    train_history = sorted(glob(cfg.DIR + '/history*.pth'))
    val_history = sorted(glob(cfg.DIR + '/validation*.pkl'))
    for train_rec, val_rec in zip(train_history, val_history):
        epoch = train_rec.split('_')[-1].split('.')[0]
        with open(val_rec, 'rb') as f:
            val_data = pickle.load(f)
        train_data = torch.load(train_rec)
        train_acc = train_data['train']['acc']
        print(len(train_acc))
        # avg_acc = np.mean(np.array(train_acc[(int(epoch)-1)*100: int(epoch)*100]))
        avg_acc = np.mean(np.array(train_acc))
        train_loss = train_data['train']['loss']
        avg_total_loss = np.mean(np.array(train_loss))

        print('Epoch: [{}], Train_Acc: {:4.2f}, Train_Loss: {:.6f}, '
              'Val_Acc: {:.2f}, Val_iou: {:.4f}'
                .format(epoch, avg_acc, avg_total_loss,
                        val_data['acc'], val_data['mean_iou']))


def train_res(cfg):
    print(cfg.DIR)
    train_history = sorted(glob(cfg.DIR + '/history*.pth'))
    for train_rec in train_history:
        epoch = train_rec.split('_')[-1].split('.')[0]
        train_data = torch.load(train_rec)
        train_acc = train_data['train']['acc']
        avg_acc = np.mean(np.array(train_acc))
        train_loss = train_data['train']['loss']
        avg_total_loss = np.mean(np.array(train_loss))

        print('Epoch: [{}], Train_Acc: {:4.2f}, Train_Loss: {:.6f}, '
                .format(epoch, avg_acc, avg_total_loss))


def valid_res(cfg):
    print(cfg.DIR)
    val_history = sorted(glob(cfg.DIR + '/validation*.pkl'))
    for val_rec in val_history:
        epoch = val_rec.split('_')[-1].split('.')[0]
        with open(val_rec, 'rb') as f:
            val_data = pickle.load(f)
        # print(len(val_data['iou']))
        ious = val_data['iou']
        # print(len(ious))
        mean_iou = np.mean(ious)
        weighted_iou = np.sum(ious*ratio_by_cate)

        # exit() val_data['mean_iou']
        print('Epoch: [{}], Val_Acc: {:.2f}, Val_C_Acc: {:.2f}, Val_iou: {:.4f}, Val_weighted_iou: {:.4f}'
                .format(epoch, val_data['acc'], val_data['acc_b'], mean_iou, weighted_iou))  # 


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # main(cfg)
    train_res(cfg)
    valid_res(cfg)
