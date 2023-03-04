import os
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my import
from dataset_all import TrainLabeled, TrainUnlabeled, ValLabeled
from model import AIMnet
from utils import *
from trainer import Trainer


def main(gpu, args):
    args.local_rank = gpu
    # random seed
    setup_seed(2022)
    # load data
    train_folder = args.data_dir
    paired_dataset = TrainLabeled(dataroot=train_folder, phase='labeled1', finesize=args.crop_size)
    unpaired_dataset = TrainUnlabeled(dataroot=train_folder, phase='unlabeled', finesize=args.crop_size)
    val_dataset = ValLabeled(dataroot=train_folder, phase='val', finesize=args.crop_size)
    paired_sampler = None
    unpaired_sampler = None
    val_sampler = None
    paired_loader = DataLoader(paired_dataset, batch_size=args.train_batchsize, sampler=paired_sampler)
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=args.train_batchsize, sampler=unpaired_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batchsize, sampler=val_sampler)
    print('there are total %s batches for train' % (len(paired_loader)))
    print('there are total %s batches for val' % (len(val_loader)))
    # create model
    net = AIMnet()
    ema_net = AIMnet()
    ema_net = create_emamodel(ema_net)
    print('student model params: %d' % count_parameters(net))
    # tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    trainer = Trainer(model=net, tmodel=ema_net, args=args, supervised_loader=paired_loader,
                      unsupervised_loader=unpaired_loader,
                      val_loader=val_loader, iter_per_epoch=len(unpaired_loader), writer=writer)

    trainer.train()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-g', '--gpus', default=2, type=int, metavar='N')
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--train_batchsize', default=8, type=int, help='train batchsize')
    parser.add_argument('--val_batchsize', default=4, type=int, help='val batchsize')
    parser.add_argument('--crop_size', default=256, type=int, help='crop size')
    parser.add_argument('--resume', default='False', type=str, help='if resume')
    parser.add_argument('--resume_path', default='/path/to/your/net.pth', type=str, help='if resume')
    parser.add_argument('--use_pretain', default='False', type=str, help='use pretained model')
    parser.add_argument('--pretrained_path', default='/path/to/pretained/net.pth', type=str, help='if pretrained')
    parser.add_argument('--data_dir', default='./data', type=str, help='data root path')
    parser.add_argument('--save_path', default='./model/ckpt/', type=str)
    parser.add_argument('--log_dir', default='./model/log', type=str)

    args = parser.parse_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    main(-1, args)
