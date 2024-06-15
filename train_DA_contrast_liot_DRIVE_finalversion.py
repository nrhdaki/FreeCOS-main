from __future__ import division
import os.path as osp
import os
import sys
import time
import math
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

#自定义模块部分 
from config import config #参数配置
from dataloader import get_train_loader, get_val_loader 
from network import Network, Network_UNet, SingleUNet, Single_IBNUNet, Single_contrast_UNet
from dataloader import XCAD
from utils.init_func import init_weight, group_weight #初始化模型的权重、对权重进行分组
from utils.data_disturb import noise_input 
#数据扰动（一种数据增强办法）：加噪声、裁剪、平移等。
#这可以增加训练数据的多样性，提高模型的鲁棒性和泛化性能
from engine.lr_policy import WarmUpPolyLR, CosinLR #学习率调度
from utils.evaluation_metric import computeF1, compute_allRetinal #计算二分类的F1值（精准率和召回率的调和平均数）、计算交集比并集的值
from Datasetloader.dataset import CSDataset
from common.logger import Logger #日志记录器，方便调试代码
import csv
#导入损失函数的类 dice损失函数、对比损失函数、基于对比关系的区域损失函数、不考虑边缘信息的ContrastRegionloss
#结合监督学习和无监督学习的对比区域损失函数、基于噪声对比估计的对比区域损失函数、类似前者但使用了所有样本对进行对比学习的函数
#在 ContrastRegionloss_AllNCE 的基础上，使用了重复样本和查询样本之间的对比学习的函数
from utils.loss_function import DiceLoss, Contrastloss, ContrastRegionloss, ContrastRegionloss_noedge, \
    ContrastRegionloss_supunsup, ContrastRegionloss_NCE, ContrastRegionloss_AllNCE, ContrastRegionloss_quaryrepeatNCE, Triplet
import copy
from base_model.discriminator import PredictDiscriminator, PredictDiscriminator_affinity #判别器模型
#一个特征记忆模块的实现，用于处理两个域的无监督学习任务
from base_model.feature_memory import FeatureMemory_TWODomain_NC

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

#sample_set_sup：有监督样本字典、sample_set_unsup：无监督样本字典
def check_feature(sample_set_sup, sample_set_unsup):
    """
    feature N,dims,Has bug or debuff because of zeros
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
#防止提取特征值为0的数据

def train(epoch, Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
          optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
          average_negregion):
    if torch.cuda.device_count() > 1:
        Segment_model.module.train()
        predict_Discriminator_model.module.train()
    else:
        print("start_model_train")
        Segment_model.train()
        predict_Discriminator_model.train()
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'#进度条格式
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
    for idx in pbar:#加载有监督的数据批次
        current_idx = epoch * config.niters_per_epoch + idx
        damping = (1 - current_idx / total_iteration)
        start_time = time.time()
        optimizer_l.zero_grad()
        optimizer_D.zero_grad()
        try:
            minibatch = dataloader.next()
        except StopIteration:
            dataloader = iter(dataloader_supervised)
            minibatch = dataloader.next()

        imgs = minibatch['img']#图像数据
        gts = minibatch['anno_mask']#注释掩码数据
        imgs = imgs.cuda(non_blocking=True)#.cuda 移动到GPU上计算
        gts = gts.cuda(non_blocking=True)
        with torch.no_grad():
            weight_mask = gts.clone().detach()
            weight_mask[weight_mask == 0] = 0.1
            weight_mask[weight_mask == 1] = 12
            criterion_bce = nn.BCELoss(weight=weight_mask)
        try:
            unsup_minibatch = unsupervised_dataloader.next()
        except StopIteration:
            unsupervised_dataloader = iter(dataloader_unsupervised)
            unsup_minibatch = unsupervised_dataloader.next()
#代码使用torch.no_grad()上下文管理器，该上下文管理器用于禁用梯度计算。
#在此上下文中，代码生成weight_mask变量，它是gts的克隆，并且不再与原始张量关联。
#然后，根据weight_mask的值，将其中值为0的元素替换为0.1，将值为1的元素替换为12。
#这样做的目的是为了在计算二元交叉熵损失时，给gts中的不同类别赋予不同的权重。
#然后，通过nn.BCELoss函数创建criterion_bce损失函数，并传入weight_mask作为权重参数。

        unsup_imgs = unsup_minibatch['img']
        unsup_imgs = unsup_imgs.cuda(non_blocking=True)

        # Start train fake vessel 训练伪血管网络
        # 冻结参数 不进行梯度更新
        for param in predict_Discriminator_model.parameters():
            param.requires_grad = False
        #有监督训练
        pred_sup_l, sample_set_sup, flag_sup = Segment_model(imgs, mask=gts, trained=True, fake=True)
        #计算交叉熵损失函数
        loss_ce = 2 * criterion_bce(pred_sup_l, gts)  # For retinal :5 For XCAD:0.5
        loss_dice = criterion(pred_sup_l, gts)
        #无监督训练 得到预测结果和样本特征
        pred_target, sample_set_unsup, flag_un = Segment_model(unsup_imgs, mask=None, trained=True, fake=False)
        #判别结果和对抗损失
        D_seg_target_out = predict_Discriminator_model(pred_target)
        loss_adv_target = bce_loss(F.sigmoid(D_seg_target_out),
                                   torch.FloatTensor(D_seg_target_out.data.size()).fill_(source_label).cuda())
        #提取样本的特征变量（qf：特征向量；pf：正样本特征向量：nf：负样本特征向量；flag:标志变量
        quary_feature, pos_feature, neg_feature, flag = check_feature(sample_set_sup, sample_set_unsup)

        if flag:#flag为真
            #计算特征向量之间的对比损失，度量差异
            loss_contrast = criterion_contrast(quary_feature, pos_feature, neg_feature)
        else:#flag为假
            loss_contrast = 0 #不计算对比损失
        weight_contrast = 0.04  # 0.2 for 0.1bcesim  0.5 for 3_debug_losssimi 0.5for labelpos->unlabposneg 2 for triplet 0.04 for NCE allpixel/0.01maybe same as dice
        #控制对比损失的权重

        #seg：总损失=相似度损失+交叉熵损失
        loss_seg = loss_dice + loss_ce
        sum_loss_seg += loss_seg.item()
        loss_contrast_sum = weight_contrast * (loss_contrast)#对比损失
        sum_contrastloss += loss_contrast_sum

        #对抗训练的总损失：目标损失、loss_dice、loss_ce、对比损失
        loss_adv = (loss_adv_target * damping) / 8 + loss_dice + loss_ce + weight_contrast * (loss_contrast)
        loss_adv.backward(retain_graph=False) #反向传播计算梯度，释放计算图
        loss_adv_sum = (loss_adv_target * damping) / 8 # bias /4XCAD /8 STARE #仅包含目标损失部分的值
        sum_adv_loss += loss_adv_sum.item()#累加
        sum_celoss += loss_ce.item()
        sum_diceloss += loss_dice.item()
        # train predict_Discriminator_model
        for param in predict_Discriminator_model.parameters():

            param.requires_grad = True #可梯度更新
        #处理预测结果
        pred_sup_l = pred_sup_l.detach() #对来自源域的预测结果进行截断
        D_out_src = predict_Discriminator_model(pred_sup_l) #对截断后的进行预测
        #计算源域损失和目标域损失，使用了二分类交叉熵损失函数bce_loss，
        #将D_out_src和D_out_tar通过F.sigmoid函数进行sigmoid激活，并计算其与目标标签之间的交叉熵损失
        loss_D_src = bce_loss(F.sigmoid(D_out_src), torch.FloatTensor(
            D_out_src.data.size()).fill_(source_label).cuda())
        loss_D_src = loss_D_src / 16 # bias /8XCAD /16 STARE #平衡源域和目标域的样本数量差异
        loss_D_src.backward(retain_graph=False) #反向传播计算梯度
        sum_Dsrc_loss += loss_D_src.item()

        pred_target = pred_target.detach() #目标域
        D_out_tar = predict_Discriminator_model(pred_target)
        loss_D_tar = bce_loss(F.sigmoid(D_out_tar), torch.FloatTensor(
            D_out_tar.data.size()).fill_(target_label).cuda())
        loss_D_tar = loss_D_tar/16 # bias /8XCAD /16 STARE
        loss_D_tar.backward(retain_graph=False)
        sum_Dtar_loss += loss_D_tar.item()
        optimizer_l.step() #更新优化器参数
        optimizer_D.step()

        #根据lr_policy和lrD_policy获取当前迭代步数current_idx对应的学习率lr和Lr_D
        #Learning Rate Policy：学习率调整策略：控制模型参数在每次迭代中的更新幅度。合理调整，可以使模型更快地收敛或更好地适应数据。
        #Learning Rate Decay Policy：学习率衰减策略：在训练的早期阶段较大幅度地更新模型参数，帮助模型更快地收敛；
        #而在训练的后期阶段逐渐降低学习率，使得模型能够更精细地调整参数，以达到更好的性能
        lr = lr_policy.get_lr(current_idx)  # lr change

        #通过修改优化器中的学习率参数来更新模型的学习率。
        #将学习率lr分别赋值给param_groups中的第1个和第2个参数组（一般对应于主干网络和辅助分支，分别属于模型的不同部分），
        #并且遍历从第3个参数组开始的所有参数组，将学习率同样赋值为lr
        optimizer_l.param_groups[0]['lr'] = lr
        optimizer_l.param_groups[1]['lr'] = lr
        for i in range(2, len(optimizer_l.param_groups)):
            optimizer_l.param_groups[i]['lr'] = lr

        Lr_D = lrD_policy.get_lr(current_idx)
        optimizer_D.param_groups[0]['lr'] = Lr_D
        for i in range(2, len(optimizer_D.param_groups)):
            optimizer_D.param_groups[i]['lr'] = Lr_D

        #计算各个损失项的总和，打印训练信息

        sum_contrastloss += loss_contrast_sum
        print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                    + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' lr=%.2e' % lr \
                    + ' loss_seg=%.4f' % loss_seg.item() \
                    + ' loss_D_tar=%.4f' % loss_D_tar.item() \
                    + ' loss_D_src=%.4f' % loss_D_src.item() \
                    + ' loss_adv=%.4f' % loss_adv.item() \
                    + 'loss_ce=%.4f' % loss_ce.item() \
                    + 'loss_dice=%.4f' % loss_dice.item() \
                    + 'loss_contrast=%.4f' % loss_contrast_sum

        sum_totalloss = sum_totalloss + sum_Dtar_loss + sum_Dsrc_loss + sum_adv_loss + sum_loss_seg + sum_contrastloss
        pbar.set_description(print_str, refresh=False)
        end_time = time.time()

    train_loss_seg = sum_loss_seg / len(pbar) #计算各个损失项的均值
    train_loss_Dtar = sum_Dtar_loss / len(pbar)
    train_loss_Dsrc = sum_Dsrc_loss / len(pbar)
    train_loss_adv = sum_adv_loss / len(pbar)
    train_loss_ce = sum_celoss / len(pbar)
    train_loss_dice = sum_diceloss / len(pbar)
    train_loss_contrast = sum_contrastloss / len(pbar)
    #加和得到总体的训练损失
    train_total_loss = train_loss_seg + train_loss_Dtar + train_loss_Dsrc + train_loss_adv + train_loss_contrast
    #将每个损失项的平均值、总体训练损失和其他指标作为返回值
    return train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion


# evaluate(epoch, model, dataloader_val,criterion,criterion_consist)
#                    语义分割模型              判别器模型             验证集加载器   损失函数
def evaluate(epoch, Segment_model, predict_Discriminator_model, val_target_loader, criterion):
    #判断是否需要多GPU评估，用eval()方法进入模型评估模式
    if torch.cuda.device_count() > 1:
        Segment_model.module.eval()
        predict_Discriminator_model.module.eval()
    else:
        Segment_model.eval()
        predict_Discriminator_model.eval()
    with torch.no_grad():
        # val_dataloader = iter(val_target_loader)
        val_sum_loss_sup = 0 #验证集上的监督学习损失总和
        val_sum_loss_sup_r = 0 #验证集上的监督学习重建损失总和
        val_sum_cps_loss = 0 #验证集上的CPS损失总和
        val_sum_f1 = 0 #表示验证集上的F1值、精确率、召回率、特异度、准确率和Jaccard系数总和
        val_sum_pr = 0
        val_sum_re = 0
        val_sum_sp = 0
        val_sum_acc = 0
        val_sum_jc = 0
        val_sum_AUC = 0 #验证集上每个批次的平均AUC值之和
        F1_best = 0 #精确率和召回率的调和平均数（对于不平衡数据集的评估更加稳）
        print('begin eval')
        ''' supervised part '''
        for val_idx, minibatch in enumerate(val_target_loader): #遍历验证集的每个数据批次，并进行评估
            start_time = time.time()
            val_imgs = minibatch['img'] #获取图像
            val_gts = minibatch['anno_mask'] #获取标签
            val_imgs = val_imgs.cuda(non_blocking=True) #转移到GPU上处理
            val_gts = val_gts.cuda(non_blocking=True)
            # NCHW
            #对验证图像进行预测
            val_pred_sup_l, sample_set_unsup, _ = Segment_model(val_imgs, mask=None, trained=False, fake=False)
            max_l = torch.where(val_pred_sup_l >= 0.5, 1, 0) #将val_pred_sup_l中大于等于0.5的值置为1，否则为0
            val_max_l = max_l.float() #转换为浮点数
            val_loss_sup = criterion(val_pred_sup_l, val_gts) #使用交叉熵损失函数计算验证损失

            current_validx = epoch * config.niters_per_epoch + val_idx
            # unsup_weight = config.unsup_weight
            val_loss = val_loss_sup
            #传入预测结果
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
        # print("len(val_target_loader)",len(val_target_loader))#126
        #计算平均值
        val_mean_f1 = val_sum_f1 / len(val_target_loader)
        val_mean_pr = val_sum_pr / len(val_target_loader)
        val_mean_re = val_sum_re / len(val_target_loader)
        val_mean_AUC = val_sum_AUC / len(val_target_loader)
        val_mean_acc = val_sum_acc / len(val_target_loader)
        val_mean_sp = val_sum_sp / len(val_target_loader)
        val_mean_jc = val_sum_jc / len(val_target_loader)
        val_loss_sup = val_sum_loss_sup / len(val_target_loader)
        return val_mean_f1, val_mean_AUC, val_mean_pr, val_mean_re, val_mean_acc, val_mean_sp, val_mean_jc, val_loss_sup

#定义主函数
def main():
    if os.getenv('debug') is not None:
        is_debug = os.environ['debug']
    else:
        is_debug = False
    parser = argparse.ArgumentParser() #用于解析命令行参数
    os.environ['MASTER_PORT'] = '169711' #设置环境变量的值（在分布式训练中用于指定不同进程之间的通信端口）

    args = parser.parse_args() #将结果保存在变量args中
    cudnn.benchmark = True #启用cuDNN的自动调整功能，可以提高训练速度
    # set seed（设置随机数种子）
    seed = config.seed  # 12345
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("Begin Dataloader.....")


    CSDataset.initialize(datapath=config.datapath) #初始化数据集路径信息
    #带标注的有监督训练数据加载器
    dataloader_supervised = CSDataset.build_dataloader(config.benchmark,  # XCAD
                                                       config.batch_size,
                                                       config.nworker,
                                                       'train',
                                                       config.img_mode,
                                                       config.img_size,
                                                       'supervised')  # FDA sys #只返回带标注的数据
    #不带标注的无监督训练数据加载器
    dataloader_unsupervised = CSDataset.build_dataloader(config.benchmark,  # XCAD
                                                         config.batch_size,
                                                         config.nworker,
                                                         'train',
                                                         config.img_mode,
                                                         config.img_size,
                                                         'unsupervised') #只返回不带标注的数据
    #验证数据加载器
    dataloader_val = CSDataset.build_dataloader(config.benchmark,
                                                config.batch_size_val,
                                                config.nworker,
                                                'val',
                                                'same',
                                                None,
                                                'supervised') #只返回带标注的数据
    print("Dataloader.....")
    criterion = DiceLoss()  # try both loss BCE and DICE

    # define and init the model
    # Single or not single
    BatchNorm2d = nn.BatchNorm2d #构建批归一化层：对数据归一化，使得每个特征的均值为0、方差为1
                                 #然后，批归一化层将归一化后的特征进行线性变换和平移，得到拉伸和偏移参数
                                 #加速模型的训练和提高模型的泛化能力
    Segment_model = Single_contrast_UNet(4, config.num_classes) #单输入，多输出的对比模型，通道为4，类别数为config.num_classes

    #初始化权重
    init_weight(Segment_model.business_layer, nn.init.kaiming_normal_, #Kaiming初始化
                BatchNorm2d, config.bn_eps, config.bn_momentum, #参数初始化
                mode='fan_in', nonlinearity='relu')
    # define the learning rate
    base_lr = config.lr  # 0.04 #基础学习率和判别器学习率
    base_lr_D = config.lr_D  # 0.04

    params_list_l = [] #需要进行优化的列表
    #group_weight：将指定的参数添加到参数列表中，并为每个参数设置不同的学习率
    params_list_l = group_weight(params_list_l, Segment_model.backbone, 
                                 BatchNorm2d, base_lr)
    # optimizer for segmentation_L
    print("config.weight_decay", config.weight_decay)
    #SGD：最基本的随机梯度下降（易陷入局部最小值）
    optimizer_l = torch.optim.SGD(params_list_l,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)
    
    #用于二分类的预测模型，使用了kaiming_normal_方法对其参数进行初始化
    #optimizer_D为优化器，采用了Adam算法，并设置了不同的学习率（base_lr_D）和动量（betas）
    predict_Discriminator_model = PredictDiscriminator(num_classes=1)
    init_weight(predict_Discriminator_model, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    #Adam：改进后的随机梯度下降，使用了动态调整的学习率（易过拟合）
    optimizer_D = torch.optim.Adam(predict_Discriminator_model.parameters(),
                                   lr=base_lr_D, betas=(0.9, 0.99))

    # config lr policy #配置学习率策略
    #总的迭代次数=训练轮数*每轮迭代次数
    total_iteration = config.nepochs * config.niters_per_epoch  # nepochs=137  niters=C.max_samples // C.batch_size
    print("total_iteration", total_iteration)
    #学习率策略对象          基础学习率       学习率幂次方    总迭代次数         渐变上升期（开始学习时学习率增加）的迭代次数
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    # lr_policy = CosinLR(base_lr, Te=5, total_iters=total_iteration, niters_per_epoch=config.niters_per_epoch)
    #针对判别器的学习率的学习率策略对象
    lrD_policy = WarmUpPolyLR(base_lr_D, config.lr_power, total_iteration,
                              config.niters_per_epoch * config.warm_up_epoch)
    # CosinLR:(start_lr, Te, total_iters,,niters_per_epoch)
    average_posregion = torch.zeros((1, 128))
    average_negregion = torch.zeros((1, 128))
    if torch.cuda.device_count() > 1:
        Segment_model = Segment_model.cuda()
        Segment_model = nn.DataParallel(Segment_model)
        average_posregion.cuda()
        average_negregion.cuda()
        predict_Discriminator_model = predict_Discriminator_model.cuda()
        predict_Discriminator_model = nn.DataParallel(predict_Discriminator_model)
        # Logger.info('Use GPU Parallel.')
    elif torch.cuda.is_available():
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
    #初始化日志记录器
    Logger.initialize(config, training=True)
    # os.path.join('logs', logname + '.log')
    val_score_path = os.path.join('logs', config.logname + '.log') + '/' + 'val_train_f1.csv'
    csv_head = ["epoch", "total_loss", "f1", "AUC", "pr", "recall", "Acc", "Sp", "JC"]

    #训练模型、记录训练损失、评估模型、记录评估指标
    create_csv(val_score_path, csv_head)
    for epoch in range(config.state_epoch, config.nepochs):
        # train_loss_sup, train_loss_consis, train_total_loss
        train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast, average_posregion, average_negregion = train(
            epoch, Segment_model, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
            optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration, average_posregion,
            average_negregion)
        print(
            "train_seg_loss:{},train_loss_Dtar:{},train_loss_Dsrc:{},train_loss_adv:{},train_total_loss:{},train_loss_contrast:{}".format(
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
        # 更新表现最好的验证集
        if val_mean_AUC > best_val_AUC:
            best_val_AUC = val_mean_AUC
            Logger.save_model_f1_S(Segment_model, epoch, best_val_AUC, optimizer_l)
            Logger.save_model_f1_T(predict_Discriminator_model, epoch, best_val_AUC, optimizer_D)


if __name__ == '__main__':
    main()
