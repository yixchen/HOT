# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
# Our libs
from hot.config import cfg
from hot.dataset import TestDataset
from hot.models import ModelBuilder, SegmentationModule
from hot.utils import colorEncode, setup_logger
from hot.lib.nn import user_scattered_collate, async_copy_to
from hot.lib.utils import as_numpy


with open('data/colors.npy', 'rb') as f:
    colors = np.load(f)

def visualize_result(data, pred, dir_result):
    (img, _, info) = data

    img_name = info.split('/')[-1]
    img_im = Image.fromarray(img)

    # prediction
    pred_color = colorEncode(pred, colors)
    pred_color_im = Image.fromarray(pred_color)
    comp_pred = Image.blend(img_im, pred_color_im, 0.7)
    comp_pred = np.array(comp_pred)

    im_vis=np.concatenate((img, comp_pred),
                           axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name))


def evaluate(segmentation_module, loader, cfg, gpu, epoch):
    print('evaluating epoch:', epoch)

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for idx, batch_data in enumerate(loader):
        info = batch_data[0]['info']
        img_name = info.split('/')[-1]

        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        print(segSize)
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp, scores_part_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
                del feed_dict

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()

        # visualization
        if cfg.TEST.visualize:
            visualize_result(
                (batch_data['img_ori'], None, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result_epoch_'+epoch),
            )

        pbar.update(1)



def main(cfg, gpu, epoch):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        cfg=cfg,
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_val = TestDataset(
        cfg.DATASET.list_test,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu, epoch)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/hot-resnet50dilated-c1.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
    )
    parser.add_argument(
        "--epoch",
        default=0,
        help="which epoch"
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
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        # cfg.DIR, 'encoder_epoch_' + args.epoch + '.pth')
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        # cfg.DIR, 'decoder_epoch_' + args.epoch + '.pth')
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"
    if not os.path.isdir(os.path.join(cfg.DIR, "result_"+cfg.TEST.checkpoint.split('.')[0])):
        os.makedirs(os.path.join(cfg.DIR, "result_"+cfg.TEST.checkpoint.split('.')[0]))

    main(cfg, args.gpu, cfg.TEST.checkpoint.split('.')[0].split('_')[-1])
