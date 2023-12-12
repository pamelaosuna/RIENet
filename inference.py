#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import sys
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import RIENET
from util import npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import yaml
from easydict import EasyDict
from common.math import se3
from common.math_torch import se3

from data_modelnet40 import ModelNet40
from data_synapse import SynapseDataset
from data_icl import TrainData, TestData
from data import dataset

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    if not args.eval:
        os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
        os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
        os.system('cp utils.py checkpoints' + '/' + args.exp_name + '/' + 'utils.py.backup')

def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0
    rotations_ab_pred = []
    translations_ab_pred = []
    
    print(len(test_loader))
    for src, target in tqdm(test_loader):
        print(src.shape, target.shape)
        src = src.cuda()
        target = target.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        
        rotation_ab_pred, translation_ab_pred,\
            loss1, loss2, loss3 = net(src, target) #  global_alignment_loss, consensus_loss, spatial_consistency_loss
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        loss = loss1.sum() + loss2.sum() + loss3.sum()
        total_loss += loss.item()

    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations_ab_pred, translations_ab_pred

def test(args, net, test_loader, boardio, textio):
    with torch.no_grad():
        test_loss, test_rotations_ab_pred, \
        test_translations_ab_pred = test_one_epoch(args, net, test_loader)

    pred_transforms = torch.from_numpy(np.concatenate([test_rotations_ab_pred,test_translations_ab_pred.reshape(-1,3,1)], axis=-1))

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    # from sklearn.metrics import r2_score
    # r_ab_r2_score = r2_score(test_eulers_ab, test_rotations_ab_pred_euler)
    # t_ab_r2_score = r2_score(test_translations_ab, test_translations_ab_pred)

def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r',encoding='utf-8') as fd:
        args = yaml.safe_load(fd)
        args = EasyDict(d=args)
    return args

def main():
    args = parse_args_from_yaml(sys.argv[1])
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'synapse':
        test_loader = DataLoader(SynapseDataset(num_points=args.n_points,
                                            num_subsampled_points=args.n_subsampled_points,
                                            partition='ZSeries-11132023', gaussian_noise=args.gaussian_noise,
                                            unseen=args.unseen, rot_factor=args.rot_factor),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=6)

    elif args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                             num_subsampled_points=args.n_subsampled_points,
                                             partition='train', gaussian_noise=args.gaussian_noise,
                                             unseen=args.unseen, rot_factor=args.rot_factor),
                                  batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6)
        test_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                            num_subsampled_points=args.n_subsampled_points,
                                            partition='test', gaussian_noise=args.gaussian_noise,
                                            unseen=args.unseen, rot_factor=args.rot_factor),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=6)

    elif args.dataset == '7scenes':
        trainset, testset = dataset.get_datasets(args)
        test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    elif args.dataset == 'icl_nuim':
        train_data = TrainData(args.data_file, args)
        train_loader = DataLoader(train_data, args.batch_size, drop_last=True, shuffle=True)
        test_data = TestData(args.data_file_test, args)
        test_loader = DataLoader(test_data, args.batch_size)

    elif args.dataset == 'kitti':
        from data.kitti_data import KittiDataset
        train_seqs = ['00','01','02','03','04','05']
        train_dataset = KittiDataset(args.root, train_seqs, args.n_points, args.voxel_size, args.data_list, 'Train', args.augment)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)

        test_seqs = ['08','09','10']
        test_dataset = KittiDataset(args.root, test_seqs, args.n_points, args.voxel_size, args.data_list,'Test', args.augment)

        test_loader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

    else:
        raise Exception("not implemented")

    if args.model == 'RIENET':
        net = RIENET(args).cuda()
        if args.eval:
            if args.model_path == '':
                model_path = 'pretrained' + '/' + args.exp_name + '/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            
            checkpoint = torch.load(model_path)
            print(checkpoint['epoch'],checkpoint['best_result'])
            net.load_state_dict(checkpoint['model'], strict=False)
            textio.cprint("end resume from checkpoint!!!!!!!!!!!!!!")
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')
    if args.eval:
        test(args, net, test_loader, boardio, textio)
    # else:
    #     train(args, net, train_loader, test_loader, boardio, textio)


    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    main()
