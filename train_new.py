

from __future__ import division
import os.path as osp
import os
import sys
import time
import random
import argparse
from tqdm import tqdm
import PIL.Image

from datetime import datetime
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
# from dataloader import get_train_loader, get_val_loader
# from network import Network, Network_UNet, SingleUNet, Single_IBNUNet, Single_contrast_UNet

# from dataloader import XCAD
from utils.init_func import init_weight, group_weight
from utils.data_disturb import noise_input
from engine.lr_policy import WarmUpPolyLR, CosinLR
from utils.evaluation_metric import computeF1, compute_allRetinal
from Datasetloader.dataset import CSDataset
from common.logger import Logger
from common.logger import AverageMeter
import csv
from utils.loss_function import DiceLoss, Contrastloss, ContrastRegionloss, ContrastRegionloss_noedge, \
    ContrastRegionloss_supunsup, ContrastRegionloss_NCE, ContrastRegionloss_AllNCE, ContrastRegionloss_quaryrepeatNCE, Triplet
import copy
from base_model.discriminator import PredictDiscriminator, PredictDiscriminator_affinity
from base_model.feature_memory import FeatureMemory_TWODomain_NC
import numpy as np
from tensorboardX import SummaryWriter

torch.cuda.set_device(0)

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class linear_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(linear_block, self).__init__()

        self.lnearconv = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.lnearconv(x)
        return x


def mask2edge(seg):
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=seg.device).reshape(1, 1, 3, 3).requires_grad_(False)
    # print("seg",torch.unique(seg))
    edge_targets = F.conv2d(seg, laplacian_kernel, padding=1)
    edge_targets = edge_targets.clamp(min=0)
    edge_targets[edge_targets > 0.1] = 1
    edge_targets[edge_targets <= 0.1] = 0
    return edge_targets


def get_query_keys_eval(cams):
    """
        Input
            cams: Tensor, cuda, Nx1x28x28

        Here, when performing evaluation, only cams are provided for all categories, including base and novel.
    """
    return_result = dict()
    cams = cams.squeeze(1).cpu()
    cams = normalize_zero_to_one(cams)  # tensor  shape:N,28,28, 0-1

    # we only need queries
    query_pos_sets = torch.where(cams > 0.92, 1.0, 0.0).to(dtype=torch.bool)
    query_neg_sets = torch.where(cams < 0.08, 1.0, 0.0).to(dtype=torch.bool)

    return_result['query_pos_sets'] = query_pos_sets
    return_result['query_neg_sets'] = query_neg_sets

    return return_result


def normalize_zero_to_one(imgs):
    if isinstance(imgs, torch.Tensor):
        bs, h, w = imgs.shape
        imgs_mins = getattr(imgs.view(bs, -1).min(1), 'values').view(bs, 1, 1)
        imgs_maxs = getattr(imgs.view(bs, -1).max(1), 'values').view(bs, 1, 1)
        return (imgs - imgs_mins) / (imgs_maxs - imgs_mins)
    else:
        raise TypeError(f'Only tensor is supported!')


def get_pixel_sets_N_myself(src_sets, select_num):
    return_ = []
    if isinstance(src_sets, torch.Tensor):
        bs, c, h, w = src_sets.shape
        flag = True
        if torch.where(src_sets > 0.5, 1, 0).shape[0] == 0:
            flag = False
            return src_sets, False
        keeps_all = torch.where(src_sets > 0.5, 1, 0).reshape(bs, -1)  # get the right area point
        for idx, keeps in enumerate(keeps_all):
            keeps_init = np.zeros_like(keeps.cpu())  # For 1204
            src_set_index = np.arange(len(keeps))
            src_set_index_keeps = src_set_index[keeps.cpu().numpy().astype(bool)]  # For 1204
            select_num[idx] = int(select_num[idx]) if int(select_num[idx]) < 500 else 500
            resultList = random.sample(range(0, len(src_set_index_keeps)), int(select_num[idx]))
            src_set_index_keeps_select = src_set_index_keeps[resultList]
            keeps_init[src_set_index_keeps_select] = 1
            return_.append(torch.tensor(keeps_init).reshape(1, h, w))
    else:
        raise ValueError(f'only tensor is supported!')
    return_ = [aa.tolist() for aa in return_]  # For 1207
    return torch.tensor(return_) * src_sets, flag


def get_query_keys_myself(
        edges,
        masks=None,
        thred_u=0.1,
        scale_u=1.0,
        percent=0.3,
        fake=True):
    """
        Input
            edges: Tensor, cuda, Nx28x28
            masks: Tensor, cuda, Nx28x28 is the mask or prediction 0-1 0r [0,1]
    """
    #######################################################
    # ---------- some pre-processing -----------------------
    #######################################################
    masks = masks.cpu()  # to cpu, Nx28x28
    #######################################################
    # ---------- get query mask for each proposal ----------#
    #######################################################
    if fake:
        # write_tensormap(masks, "mask.png")
        query_pos_sets = masks.to(dtype=torch.bool)  # here, pos=foreground area  neg=background area
        query_neg_sets = torch.logical_not(query_pos_sets)  # the background
        edges = edges.cpu()  # to cpu, Nx28x28 # 8 1 256 256
    else:
        pos_masks = torch.where(masks > (1 - thred_u), 1.0, 0.0).to(dtype=torch.bool)  # greater 0.9
        neg_mask = torch.where(masks < thred_u, 1.0, 0.0).to(dtype=torch.bool)  # less than 0.1
        query_pos_sets = pos_masks.to(dtype=torch.bool)  # here, pos=foreground area  neg=background area
        query_neg_sets = neg_mask.to(dtype=torch.bool)  # 8 1 256 256
    #######################################################
    # ----------- get different types of keys -------------
    #######################################################
    # different sets, you can refer to the figure in https://blog.huiserwang.site/2022-03/Project-ContrastMask/ to easily understand.
    if fake:  # fakedata  with mask
        # for all region
        label_positive_sets = torch.where(masks > (1.0 - thred_u * scale_u), 1.0,
                                           0.0)  # scale_u can adjust the threshold, it is not used in our paper.
        label_negative_sets = torch.where(masks < (thred_u * scale_u), 1.0, 0.0)
        easy_positive_sets = label_positive_sets
        easy_negative_sets = label_negative_sets
        hard_positive_sets = label_positive_sets
        hard_negative_sets = label_negative_sets

    else:
        # for novel(unseen), get keys according to cam, hard and easy are both sampled in the same sets, replace original sets
        unseen_positive_sets = torch.where(masks > (1.0 - thred_u * scale_u), 1.0,
                                           0.0)  # scale_u can adjust the threshold, it is not used in our paper.
        unseen_negative_sets = torch.where(masks < (thred_u * scale_u), 1.0, 0.0)
        easy_positive_sets = unseen_positive_sets
        easy_negative_sets = unseen_negative_sets
        hard_positive_sets = unseen_positive_sets
        hard_negative_sets = unseen_negative_sets
    #######################################################
    # --------- determine the number of sampling ----------
    #######################################################
    # how many points can be sampled for all proposals for each type of sets
    num_Epos_ = easy_positive_sets.sum(dim=[2, 3])  # H, W to count points numbers E=easy, H=hard
    num_Hpos_ = hard_positive_sets.sum(dim=[2, 3])
    num_Eneg_ = easy_negative_sets.sum(dim=[2, 3])
    num_Hneg_ = hard_negative_sets.sum(dim=[2, 3])
    # if available points are less then 5 for each type, this proposal will be dropped out.
    available_num = torch.cat([num_Epos_, num_Eneg_, num_Hpos_, num_Hneg_])
    abandon_inds = torch.where(available_num < 5, 1, 0).reshape(4, -1)
    keeps = torch.logical_not(abandon_inds.sum(0).to(dtype=torch.bool))
    if True not in keeps:  # all proposals do not have enough points that can be sample. This is a extreme situation.
        # set the points number of all types sets to 2
        # sometimes, there would still raise an error. I will fix it later.
        sample_num_Epos = torch.ones_like(num_Epos_) * 2
        sample_num_Hpos = torch.ones_like(num_Hpos_) * 2
        sample_num_Eneg = torch.ones_like(num_Eneg_) * 2
        sample_num_Hneg = torch.ones_like(num_Hneg_) * 2.
    else:
        sample_num_Epos = (percent * num_Epos_[keeps]).ceil()  # percent is the sigma in our paper
        sample_num_Hpos = (percent * num_Hpos_[keeps]).ceil()
        sample_num_Eneg = (percent * num_Eneg_[keeps]).ceil()
        sample_num_Hneg = (percent * num_Hneg_[keeps]).ceil()

    #######################################################
    # ----------------- sample points ---------------------
    #######################################################
    empty_dict = {}
    easy_positive_sets_N, flag0 = get_pixel_sets_N_myself(easy_positive_sets[keeps], sample_num_Epos)
    if not flag0:
        return empty_dict, False
    easy_negative_sets_N, flag1 = get_pixel_sets_N_myself(easy_negative_sets[keeps], sample_num_Eneg)
    if not flag1:
        return empty_dict, False
    hard_positive_sets_N, flag2 = get_pixel_sets_N_myself(hard_positive_sets[keeps], sample_num_Hpos)
    if not flag2:
        return empty_dict, False
    hard_negative_sets_N, flag3 = get_pixel_sets_N_myself(hard_negative_sets[keeps], sample_num_Hneg)
    if not flag3:
        return empty_dict, False

    # Record points number
    num_per_type = dict()
    num_per_type['Epos_num_'] = sample_num_Epos
    num_per_type['Hpos_num_'] = sample_num_Hpos
    num_per_type['Eneg_num_'] = sample_num_Eneg
    num_per_type['Hneg_num_'] = sample_num_Hneg

    #######################################################
    # ------------------- return data ---------------------
    #######################################################
    return_result = dict()
    return_result['keeps'] = keeps  # which proposal is preserved
    return_result['num_per_type'] = num_per_type
    return_result['query_pos_sets'] = query_pos_sets[keeps]  # query area for foreground
    return_result['query_neg_sets'] = query_neg_sets[keeps]  # query area for background
    return_result['easy_positive_sets_N'] = easy_positive_sets_N.to(dtype=torch.bool)
    return_result['easy_negative_sets_N'] = easy_negative_sets_N.to(dtype=torch.bool)
    return_result['hard_positive_sets_N'] = hard_positive_sets_N.to(dtype=torch.bool)
    return_result['hard_negative_sets_N'] = hard_negative_sets_N.to(dtype=torch.bool)
    return return_result, True


class ContrastiveHead_myself(nn.Module):

    def __init__(self,
                 num_convs=1,
                 num_projectfc=2,
                 in_channels=64,
                 conv_out_channels=64,
                 fc_out_channels=64,
                 thred_u=0.1,
                 scale_u=1.0,
                 percent=0.3):
        super(ContrastiveHead_myself, self).__init__()
        self.num_convs = num_convs  # layer of encoder
        self.num_projectfc = num_projectfc  # layers of projector
        self.in_channels = in_channels  # channels number
        self.conv_out_channels = conv_out_channels  # out put channels numbers
        self.fc_out_channels = fc_out_channels
        self.thred_u = thred_u
        self.scale_u = scale_u
        self.percent = percent
        self.fake = True

        # build encoder module
        self.encoder = nn.ModuleList()  # make a list to upconv
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            self.encoder.append(
                conv_block(in_channels, self.conv_out_channels))
            last_layer_dim = self.conv_out_channels

        # build projecter module
        self.projector = nn.ModuleList()
        for j in range(self.num_projectfc - 1):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            self.projector.append(
                conv_block(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        self.projector.append(linear_block(in_ch=last_layer_dim, out_ch=self.fc_out_channels))

    def forward(self, x, masks, trained, faked):
        # mask for supervised  and prdict for unsupervised
        """
        We get average foreground pixel and background pixel for Quary pixel feature (by mask and thrshold for prdiction)
        easy by bounary on the boundary and less than
        """
        self.fake = faked
        sample_sets = dict()
        if self.fake:
            edges = mask2edge(masks)
        else:
            edges = None
        # 1. get query and keys
        if trained:  # training phase
            sample_results, flag = get_query_keys_myself(edges, masks, thred_u=self.thred_u, scale_u=self.scale_u,
                                                         percent=self.percent, fake=self.fake)
            if flag == False:
                return x, sample_results, flag
            keeps_ = sample_results['keeps']
            keeps = keeps_.reshape(-1, 1, 1)
            keeps = keeps.expand(keeps.shape[0], x.shape[2],
                                 x.shape[3])  # expand the flag(numbers of batch level) for the whole feature
            keeps_all = keeps.reshape(-1)
        else:  # evaluation phase
            sample_results = get_query_keys_eval(masks)

        # 2. forward
        for conv in self.encoder:
            x = conv(x)
        x_pro = self.projector[0](x)
        for i in range(1, len(self.projector) - 1):
            x_pro = self.projector[i](x_pro)
        n, c, h, w = x_pro.shape
        x_pro = x_pro.permute(0, 2, 3, 1).reshape(-1, c)  # n,c,h,w -> n,h,w,c -> (nhw),c
        x_pro = self.projector[-1](x_pro)  # (nhw),c

        # 3. get vectors for queries and keys so that we can calculate contrastive loss
        if trained:
            query_pos_num = sample_results['query_pos_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[2, 3])
            query_neg_num = sample_results['query_neg_sets'].to(device=x_pro.device, dtype=x_pro.dtype).sum(dim=[2, 3])
            sample_easy_pos = x_pro[keeps_all][sample_results['easy_positive_sets_N'].reshape(-1), :]  # *, 256
            sample_easy_neg = x_pro[keeps_all][sample_results['easy_negative_sets_N'].reshape(-1), :]  # *, 256
            sample_hard_pos = x_pro[keeps_all][sample_results['hard_positive_sets_N'].reshape(-1), :]  # *, 256
            sample_hard_neg = x_pro[keeps_all][sample_results['hard_negative_sets_N'].reshape(-1),
                              :]  # *, 256 choose the True postion feature as sample [oint
            squeeze_sampletresult = sample_results['query_pos_sets'].squeeze(1)  # 5 256 256 to get the whole pos map
            query_pos = (x_pro[keeps_all].reshape(-1, 256, 256, 64) * squeeze_sampletresult.to(
                device=x_pro[keeps_all].device).unsqueeze(3)).sum(dim=[1, 2]) / query_pos_num
            query_pos_set = x_pro[keeps_all][sample_results['query_pos_sets'].reshape(-1), :]
            squeeze_negsampletresult = sample_results['query_neg_sets'].squeeze(1)  # 5 256 256
            query_neg = (x_pro[keeps_all].reshape(-1, 256, 256, 64) * squeeze_negsampletresult.to(
                device=x_pro[keeps_all].device).unsqueeze(3)).sum(dim=[1, 2]) / query_neg_num
            query_neg_set = x_pro[keeps_all][sample_results['query_neg_sets'].reshape(-1), :]
            # sample sets are used to calculate loss
            sample_sets['keeps_proposal'] = keeps_
            sample_sets['query_pos'] = query_pos.unsqueeze(1)  # N,HW,C 1,1,64
            sample_sets['query_neg'] = query_neg.unsqueeze(1)
            sample_sets['query_pos_set'] = query_pos_set  # N,dims
            sample_sets['query_neg_set'] = query_neg_set  # N,dims
            sample_sets['num_per_type'] = sample_results['num_per_type']
            sample_sets['sample_easy_pos'] = sample_easy_pos  # N,64
            sample_sets['sample_easy_neg'] = sample_easy_neg
            sample_sets['sample_hard_pos'] = sample_hard_pos
            sample_sets['sample_hard_neg'] = sample_hard_neg
        return x, sample_sets, True


class UNet_contrast(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels, n_classes):
        super(UNet_contrast, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()
        #self.contrast = ContrastiveHead_torch(num_convs=1,num_projectfc=2,thred_u=0.1,scale_u=1.0,percent=0.3) #init the contrast head to conv 8;
        self.contrast = ContrastiveHead_myself(num_convs=1,num_projectfc=2,thred_u=0.1,scale_u=1.0,percent=0.3)

    def cat_(self,xe,xd):
        diffY = xe.size()[2] - xd.size()[2]
        diffX = xe.size()[3] - xd.size()[3]
        xd = F.pad(xd, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        xd = torch.cat((xe, xd), dim=1)
        return xd

    def forward(self, x, mask, trained,fake):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = self.cat_(e4,d5)
        #d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.cat_(e3, d4)
        #d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.cat_(e2, d3)
        #d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.cat_(e1, d2)
        #d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        #d2 = d2 + contrast_tensor0
        out = self.Conv(d2)
        d1 = self.active(out)
        if trained and fake:
            contrast_tensor0, sample_sets, flag = self.contrast(d2,mask,trained,fake)
        elif trained and fake==False:
            contrast_tensor0, sample_sets, flag = self.contrast(d2, d1, trained, fake)
        else:
            contrast_tensor0, sample_sets, flag = self.contrast(d2,d1,trained,fake)

        return d1, sample_sets, flag


class Single_contrast_UNet(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(Single_contrast_UNet, self).__init__()
        self.backbone = UNet_contrast(n_channels=n_channels, n_classes=num_classes)
        self.business_layer = []
        self.business_layer.append(self.backbone)

    def forward(self, data, mask=None, trained=True, fake=True):
        pred, sample_set, flag = self.backbone(data, mask, trained, fake)
        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        return pred, sample_set, flag

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

# def get_parser():
#     parser = argparse.ArgumentParser(description='JTFN for Curvilinear Structure Segmentation')
#     parser.add_argument('--config', type=str, default='config/UNet_DRIVE.yaml', help='Model config file')
#     args = parser.parse_args()
#     assert args.config is not None
#     cfg = config.load_cfg_from_cfg_file(args.config)
#     return cfg

def create_csv(path, csv_head):
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        # csv_head = ["good","bad"]
        csv_write.writerow(csv_head)

def write_csv(path, data_row):
    # path  = "aa.csv"
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        # data_row = ["1","2"]
        csv_write.writerow(data_row)

def check_feature(sample_set_sup, sample_set_unsup):
    """
    feature N,dims，Has bug or debuff because of zeros
    """
    flag = True
    Single = False
    queue_len = 500
    # sample_set_sup['sample_easy_pos'], sample_set_sup['sample_easy_neg'], sample_set_unsup['sample_easy_pos'], sample_set_unsup['sample_easy_neg']
    with torch.no_grad():
        if 'sample_easy_pos' not in sample_set_sup.keys() or 'sample_easy_neg' not in sample_set_unsup.keys() or 'sample_easy_pos' not in sample_set_unsup.keys():
            flag = False
            quary_feature = None
            pos_feature = None
            neg_feature = None
        else:
            quary_feature = sample_set_sup['sample_easy_pos']
            pos_feature = sample_set_unsup['sample_easy_pos']
            neg_feature = sample_set_unsup['sample_easy_neg']
            flag = True

        if 'sample_easy_neg' in sample_set_sup.keys() and 'sample_easy_neg' in sample_set_unsup.keys():
            neg_unlabel = sample_set_unsup['sample_easy_neg']
            neg_label = sample_set_sup['sample_easy_neg']
            neg_feature = torch.cat((neg_unlabel[:min(queue_len // 2, neg_unlabel.shape[0]), :],
                                     neg_label[:min(queue_len // 2, neg_label.shape[0]), :]), dim=0)
    return quary_feature, pos_feature, neg_feature, flag

def train(epoch, Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
          optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
          average_negregion):
    # if torch.cuda.device_count() > 1:
    #     Segment_model.module.train()
    #     predict_Discriminator_model.module.train()
    # else:
    
    print("start_model_train")
    Segment_model.train()
    predict_Discriminator_model.train()
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
    dataloader = iter(dataloader_supervised)
    unsupervised_dataloader = iter(dataloader_unsupervised)
    bce_loss = nn.BCELoss()
    sum_loss_seg = 0
    sum_adv_loss = 0
    sum_Dsrc_loss = 0
    sum_Dtar_loss = 0
    sum_totalloss = 0
    sum_contrastloss = 0
    sum_celoss = 0
    sum_diceloss = 0
    source_label = 0
    target_label = 1
    criterion_contrast = ContrastRegionloss_quaryrepeatNCE()
    print('begin train')
    ''' supervised part '''
    for idx in pbar:
        current_idx = epoch * config.niters_per_epoch + idx
        damping = (1 - current_idx / total_iteration)
        start_time = time.time()
        optimizer_l.zero_grad()
        optimizer_D.zero_grad()
        try:
            minibatch = next(dataloader)
        except StopIteration:
            dataloader = iter(dataloader_supervised)
            minibatch = next(dataloader)

        imgs = minibatch['img']
        gts = minibatch['anno_mask']
        imgs = imgs.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)
        with torch.no_grad():
            weight_mask = gts.clone().detach()
            weight_mask[weight_mask == 0] = 0.1
            weight_mask[weight_mask == 1] = 1
            criterion_bce = nn.BCELoss(weight=weight_mask)
        try:
            unsup_minibatch = next(unsupervised_dataloader)
        except StopIteration:
            unsupervised_dataloader = iter(dataloader_unsupervised)
            unsup_minibatch = next(unsupervised_dataloader)

        unsup_imgs = unsup_minibatch['img']
        unsup_imgs = unsup_imgs.cuda(non_blocking=True)
        # unsup_imgs = unsup_imgs.cuda()
        
        # Start train fake vessel
        for param in predict_Discriminator_model.parameters():
            param.requires_grad = False
        pred_sup_l, sample_set_sup, flag_sup = Segment_model(imgs, mask=gts, trained=True, fake=True)
        loss_ce = 0.1 * criterion_bce(pred_sup_l, gts)  # For retinal :5 For XCAD:0.1 5 for crack
        loss_dice = criterion(pred_sup_l, gts)
        pred_target, sample_set_unsup, flag_un = Segment_model(unsup_imgs, mask=None, trained=True, fake=False)
        D_seg_target_out = predict_Discriminator_model(pred_target)
        loss_adv_target = bce_loss(F.sigmoid(D_seg_target_out),
                                   torch.FloatTensor(D_seg_target_out.data.size()).fill_(source_label).cuda())
        quary_feature, pos_feature, neg_feature, flag = check_feature(sample_set_sup, sample_set_unsup)

        if flag:
            loss_contrast = criterion_contrast(quary_feature, pos_feature, neg_feature)
        else:
            loss_contrast = 0
        weight_contrast = 0.04  # 0.04 for NCE allpixel/0.01maybe same as dice
        loss_seg = loss_dice + loss_ce
        sum_loss_seg += loss_seg.item()
        loss_contrast_sum = weight_contrast * (loss_contrast)
        sum_contrastloss += loss_contrast_sum

        loss_adv = (loss_adv_target * damping) / 4 + loss_dice + loss_ce + weight_contrast * (loss_contrast)
        loss_adv.backward(retain_graph=False)
        loss_adv_sum = (loss_adv_target * damping) / 4
        sum_adv_loss += loss_adv_sum.item()

        sum_celoss += loss_ce
        sum_diceloss += loss_dice.item()
        for param in predict_Discriminator_model.parameters():
            param.requires_grad = True
        pred_sup_l = pred_sup_l.detach()
        D_out_src = predict_Discriminator_model(pred_sup_l)

        loss_D_src = bce_loss(F.sigmoid(D_out_src), torch.FloatTensor(
            D_out_src.data.size()).fill_(source_label).cuda())
        loss_D_src = loss_D_src / 8
        loss_D_src.backward(retain_graph=False)
        sum_Dsrc_loss += loss_D_src.item()

        pred_target = pred_target.detach()
        D_out_tar = predict_Discriminator_model(pred_target)

        loss_D_tar = bce_loss(F.sigmoid(D_out_tar), torch.FloatTensor(
            D_out_tar.data.size()).fill_(target_label).cuda())
        loss_D_tar = loss_D_tar / 8  # bias
        loss_D_tar.backward(retain_graph=False)
        sum_Dtar_loss += loss_D_tar.item()
        optimizer_l.step()
        optimizer_D.step()

        lr = lr_policy.get_lr(current_idx)  # lr change
        optimizer_l.param_groups[0]['lr'] = lr
        optimizer_l.param_groups[1]['lr'] = lr
        for i in range(2, len(optimizer_l.param_groups)):
            optimizer_l.param_groups[i]['lr'] = lr

        Lr_D = lrD_policy.get_lr(current_idx)
        optimizer_D.param_groups[0]['lr'] = Lr_D
        for i in range(2, len(optimizer_D.param_groups)):
            optimizer_D.param_groups[i]['lr'] = Lr_D

        sum_contrastloss += loss_contrast_sum
        print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                    + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' lr=%.2e' % lr \
                    + ' loss_seg=%.4f' % loss_seg.item() \
                    + ' loss_D_tar=%.4f' % loss_D_tar.item() \
                    + ' loss_D_src=%.4f' % loss_D_src.item() \
                    + ' loss_adv=%.4f' % loss_adv.item() \
                    + ' loss_ce=%.4f' % loss_ce \
                    + ' loss_dice=%.4f' % loss_dice.item() \
                    + ' loss_contrast=%.4f' % loss_contrast_sum

        sum_totalloss = sum_totalloss + sum_Dtar_loss + sum_Dsrc_loss + sum_adv_loss + sum_loss_seg + sum_contrastloss
        pbar.set_description(print_str, refresh=False)

        end_time = time.time()

    if (epoch + 1) % 20 == 0:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"/workspace/liuye/FreeCOS/output/output_{current_time}"
        os.makedirs(output_dir, exist_ok=True)
        save_image(pred_sup_l, os.path.join(output_dir, f"pred_sup_l_epoch_{epoch+1}.png"))
        save_image(pred_target, os.path.join(output_dir, f"pred_target_epoch_{epoch+1}.png"))

    train_loss_seg = sum_loss_seg / len(pbar)
    train_loss_Dtar = sum_Dtar_loss / len(pbar)
    train_loss_Dsrc = sum_Dsrc_loss / len(pbar)
    train_loss_adv = sum_adv_loss / len(pbar)
    train_loss_ce = sum_celoss / len(pbar)
    train_loss_dice = sum_diceloss / len(pbar)
    train_loss_contrast = sum_contrastloss / len(pbar)
    train_total_loss = train_loss_seg + train_loss_Dtar + train_loss_Dsrc + train_loss_adv + train_loss_contrast
    return train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion


# evaluate(epoch, model, dataloader_val,criterion,criterion_consist)
def evaluate(epoch, Segment_model, predict_Discriminator_model, val_target_loader, criterion):
    # if torch.cuda.device_count() > 1:
    #     Segment_model.module.eval()
    #     predict_Discriminator_model.module.eval()
    # else:
    Segment_model.eval()
    predict_Discriminator_model.eval()
    with torch.no_grad():
        val_sum_loss_sup = 0
        val_sum_f1 = 0
        val_sum_pr = 0
        val_sum_re = 0
        val_sum_sp = 0
        val_sum_acc = 0
        val_sum_jc = 0
        val_sum_AUC = 0
        F1_best = 0
        print('begin eval')
        ''' supervised part '''
        for val_idx, minibatch in enumerate(val_target_loader):
            start_time = time.time()
            val_imgs = minibatch['img']
            val_gts = minibatch['anno_mask']
            val_imgs = val_imgs.cuda(non_blocking=True)
            val_gts = val_gts.cuda(non_blocking=True)
            # NCHW
            val_pred_sup_l, sample_set_unsup, _ = Segment_model(val_imgs, mask=None, trained=False, fake=False)

            max_l = torch.where(val_pred_sup_l >= 0.5, 1, 0)
            val_max_l = max_l.float()
            val_loss_sup = criterion(val_pred_sup_l, val_gts)

            current_validx = epoch * config.niters_per_epoch + val_idx
            val_loss = val_loss_sup


            val_f1, val_precision, val_recall, val_Sp, val_Acc, val_jc, val_AUC = compute_allRetinal(val_max_l,
                                                                                                     val_pred_sup_l,
                                                                                                     val_gts)
            val_sum_loss_sup += val_loss_sup.item()
            val_sum_f1 += val_f1
            val_sum_pr += val_precision
            val_sum_re += val_recall
            val_sum_AUC += val_AUC
            val_sum_sp += val_Sp
            val_sum_acc += val_Acc
            val_sum_jc += val_jc
            end_time = time.time()
        val_mean_f1 = val_sum_f1 / len(val_target_loader)
        val_mean_pr = val_sum_pr / len(val_target_loader)
        val_mean_re = val_sum_re / len(val_target_loader)
        val_mean_AUC = val_sum_AUC / len(val_target_loader)
        val_mean_acc = val_sum_acc / len(val_target_loader)
        val_mean_sp = val_sum_sp / len(val_target_loader)
        val_mean_jc = val_sum_jc / len(val_target_loader)
        val_loss_sup = val_sum_loss_sup / len(val_target_loader)
        return val_mean_f1, val_mean_AUC, val_mean_pr, val_mean_re, val_mean_acc, val_mean_sp, val_mean_jc, val_loss_sup


def main():
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    tbd_writer = SummaryWriter('/workspace/liuye/FreeCOS/yy_log/output'+ current_time)


    if os.getenv('debug') is not None:
        is_debug = os.environ['debug']
    else:
        is_debug = False
    parser = argparse.ArgumentParser()
    os.environ['MASTER_PORT'] = '169711'

    args = parser.parse_args()
    cudnn.benchmark = True
    # set seed
    seed = config.seed  # 12345
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("Begin Dataloader.....")

    CSDataset.initialize(datapath=config.datapath)
    dataloader_supervised = CSDataset.build_dataloader(config.benchmark,  # XCAD
                                                       config.batch_size,
                                                       config.nworker,
                                                       'train',
                                                       config.img_mode,
                                                       config.img_size,
                                                       'supervised')  # FDA sys
    dataloader_unsupervised = CSDataset.build_dataloader(config.benchmark,  # XCAD
                                                         config.batch_size,
                                                         config.nworker,
                                                         'train',
                                                         config.img_mode,
                                                         config.img_size,
                                                         'unsupervised')

    dataloader_val = CSDataset.build_dataloader(config.benchmark,
                                                config.batch_size_val,
                                                config.nworker,
                                                'val',
                                                'same',
                                                None,
                                                'supervised')
    print("Dataloader.....")
    criterion = DiceLoss()  # try both loss BCE and DICE

    # define and init the model
    # Single or not single
    BatchNorm2d = nn.BatchNorm2d
    Segment_model = Single_contrast_UNet(4, config.num_classes)

    init_weight(Segment_model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    
    # define the learning rate
    base_lr = config.lr  # 0.04
    base_lr_D = config.lr_D  # 0.04

    params_list_l = []
    params_list_l = group_weight(params_list_l, Segment_model.backbone,
                                 BatchNorm2d, base_lr)
    # optimizer for segmentation_L
    print("config.weight_decay", config.weight_decay)
    optimizer_l = torch.optim.SGD(params_list_l,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    predict_Discriminator_model = PredictDiscriminator(num_classes=1)
    init_weight(predict_Discriminator_model, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    optimizer_D = torch.optim.Adam(predict_Discriminator_model.parameters(),
                                   lr=base_lr_D, betas=(0.9, 0.99))

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch  # nepochs=137  niters=C.max_samples // C.batch_size
    print("total_iteration", total_iteration)
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    lrD_policy = WarmUpPolyLR(base_lr_D, config.lr_power, total_iteration,
                              config.niters_per_epoch * config.warm_up_epoch)

    average_posregion = torch.zeros((1, 128))
    average_negregion = torch.zeros((1, 128))
    # if torch.cuda.device_count() > 1:
    #     Segment_model = Segment_model.cuda()
    #     Segment_model = nn.DataParallel(Segment_model)
    #     average_posregion.cuda()
    #     average_negregion.cuda()
    #     predict_Discriminator_model = predict_Discriminator_model.cuda()
    #     predict_Discriminator_model = nn.DataParallel(predict_Discriminator_model)
    #     # Logger.info('Use GPU Parallel.')
    if torch.cuda.is_available():
        print("cuda_is available")
        Segment_model = Segment_model.cuda()
        average_posregion.cuda()
        average_negregion.cuda()
        predict_Discriminator_model = predict_Discriminator_model.cuda()
    else:
        Segment_model = Segment_model
        predict_Discriminator_model = predict_Discriminator_model

    best_val_f1 = 0
    best_val_AUC = 0
    Logger.initialize(config, training=True)
    val_score_path = os.path.join('logs', config.logname + '.log') + '/' + 'val_train_f1.csv'
    csv_head = ["epoch", "total_loss", "f1", "AUC", "pr", "recall", "Acc", "Sp", "JC"]
    create_csv(val_score_path, csv_head)

    for epoch in range(config.state_epoch, config.nepochs):
        # train_loss_sup, train_loss_consis, train_total_loss, 
        train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion = train(
            epoch, Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
            optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
            average_negregion)
        
        print(
            "train_seg_loss:{},train_loss_Dtar:{},train_loss_Dsrc:{},train_loss_adv:{},train_total_loss:{},\
            train_loss_contrast:{}".format(
                train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss,
                train_loss_contrast))
        print("train_loss_dice:{},train_loss_ce:{}".format(train_loss_dice, train_loss_ce))
        # val_mean_f1, val_mean_pr, val_mean_re, val_mean_f1, val_mean_pr, val_mean_re,val_loss_sup
        val_mean_f1, val_mean_AUC, val_mean_pr, val_mean_re, val_mean_acc, val_mean_sp, val_mean_jc, val_loss_sup = evaluate(
            epoch, Segment_model, predict_Discriminator_model, dataloader_val,
            criterion)  # evaluate(epoch, model, val_target_loader,criterion, criterion_cps)
        # val_mean_f1, val_mean_AUC, val_mean_pr, val_mean_re,val_mean_acc, val_mean_sp, val_mean_jc, val_loss_sup
        data_row_f1score = [str(epoch), str(train_total_loss), str(val_mean_f1.item()), str(val_mean_AUC),
                            str(val_mean_pr.item()), str(val_mean_re.item()), str(val_mean_acc), str(val_mean_sp),
                            str(val_mean_jc)]
        print("val_mean_f1", val_mean_f1.item())
        print("val_mean_AUC", val_mean_AUC)
        print("val_mean_pr", val_mean_pr.item())
        print("val_mean_re", val_mean_re.item())
        print("val_mean_acc", val_mean_acc.item())
        write_csv(val_score_path, data_row_f1score)
        if val_mean_f1 > best_val_f1:
            best_val_f1 = val_mean_f1
            Logger.save_model_f1_S(Segment_model, epoch, val_mean_f1, optimizer_l)
            Logger.save_model_f1_T(predict_Discriminator_model, epoch, val_mean_f1, optimizer_D)


        tbd_writer.add_scalar('f1',val_mean_f1.item(),epoch)
        tbd_writer.add_scalar('AUC',val_mean_AUC,epoch)
        tbd_writer.add_scalar('PR',val_mean_pr.item(),epoch)
        tbd_writer.add_scalar('RE',val_mean_re.item(),epoch)
        tbd_writer.add_scalar('ACC',val_mean_acc.item(),epoch)
        tbd_writer.add_scalar('Loss',train_total_loss,epoch)
        '''
            logger = Logger()
            logger.write_result("Train", epoch)
            logger.write_result("Validation", epoch)
        '''

        # if val_mean_AUC > best_val_AUC:
        #     best_val_AUC = val_mean_AUC
        #     Logger.save_model_f1_S(Segment_model, epoch, val_mean_AUC, optimizer_l)
        #     Logger.save_model_f1_T(predict_Discriminator_model, epoch, val_mean_AUC, optimizer_D)




if __name__ == '__main__':
    main()