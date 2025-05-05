import argparse
import json
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 假设 dataset 和 utils 在你的项目路径下
from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0

# <<<--- [辅助函数] 分布式数据收集 (简化版占位符) --->>>
# 注意：这需要根据你的 torch.distributed 设置进行调整或确认其可用性
def distributed_gather(tensor, args, world_size):
    """ 收集来自所有进程的张量 (需要处理不同长度) """
    if world_size <= 1:
        return tensor, [tensor.shape[0]] # 返回张量和大小列表

    # 1. 获取每个进程的张量大小
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=args.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(all_sizes, local_size)
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)

    # 如果所有进程大小相同，可以直接 all_gather
    if max_size == min(all_sizes):
        gathered_tensors = [torch.zeros_like(tensor).repeat(max_size // tensor.shape[0] if tensor.shape[0] > 0 else 1, *([1]*(tensor.dim()-1))) for _ in range(world_size)]
        if tensor.shape[0] > 0 : # 只有当本地张量不为空时才收集
             torch.distributed.all_gather(gathered_tensors, tensor)
        gathered_tensor = torch.cat(gathered_tensors, dim=0)
    else:
        # 2. 填充张量到最大尺寸
        padded_tensor = F.pad(tensor, (0,)*(2*(tensor.dim()-1)) + (0, max_size - tensor.shape[0])) # 假设填充最后一维

        # 3. 收集填充后的张量
        gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors, padded_tensor)

        # 4. 在 rank 0 上裁剪掉填充部分
        gathered_tensor = torch.cat(gathered_tensors, dim=0)
        # (裁剪逻辑在后面更新统计信息时处理，这里先返回聚合的包含填充的张量)

    return gathered_tensor, all_sizes # 返回聚合后的张量和原始大小列表

def distributed_broadcast(tensor, args, src=0):
    """ 将张量从 src 进程广播到所有进程 """
    if args.world_size > 1:
        torch.distributed.broadcast(tensor, src=src)
    return tensor
# <<<--------------------------------------------->>>

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    """ 保存模型检查点 """
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    """ 设置随机种子以保证可复现性 """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    """ 创建一个带有预热的余弦学习率调度器 """
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    """ 交错操作，用于合并弱增强和强增强的 batch """
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    """ 反交错操作，用于分离 logits """
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training with Adaptive Thresholding')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='加载数据的进程数')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='数据集名称')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='有标签数据量')
    parser.add_argument("--expand-labels", action="store_true",
                        help="扩展标签以适应评估步骤 (如果使用了这个，需要确保数据集实现支持)")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='模型架构')
    parser.add_argument('--total-steps', default=204800, type=int,
                        help='总训练步数')
    parser.add_argument('--eval-step', default=512, type=int,
                        help='每多少步评估一次')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='起始 epoch (用于断点续训)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='训练批次大小 (有标签)')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='初始学习率')
    parser.add_argument('--warmup', default=0, type=float,
                        help='学习率预热步数 (基于总步数)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='权重衰减')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='使用 Nesterov 动量')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='使用指数移动平均模型 (EMA)')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA 衰减率')
    parser.add_argument('--mu', default=7, type=int,
                        help='无标签数据批次大小相对于有标签数据的倍数')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='无标签损失的系数')
    parser.add_argument('--T', default=1, type=float,
                        help='伪标签生成的温度系数')
    # parser.add_argument('--threshold', default=0.95, type=float, # <<<--- [注释掉] 不再使用固定的全局阈值
    #                      help='伪标签置信度阈值')
    parser.add_argument('--out', default='result',
                        help='结果输出目录')
    parser.add_argument('--resume', default='', type=str,
                        help='从检查点恢复的路径 (默认: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="随机种子")
    parser.add_argument("--amp", action="store_true",
                        help="使用 Apex AMP 进行混合精度训练")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="Apex AMP 优化等级")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="分布式训练的本地进程排名")
    parser.add_argument('--no-progress', action='store_true',
                        help="不显示进度条")

    # <<<--- [新增] 自适应阈值相关参数 --->>>
    parser.add_argument('--global-thresh-base', default=0.7, type=float,
                        help='[自适应阈值] 全局基础阈值')
    parser.add_argument('--class-conf-factor', default=0.1, type=float,
                        help='[自适应阈值] 类别平均置信度权重因子')
    parser.add_argument('--confusion-factor', default=0.05, type=float,
                        help='[自适应阈值] 类别混乱度(标准差)权重因子')
    parser.add_argument('--initial-class-threshold', default=0.9, type=float,
                        help='[自适应阈值] 类别平均置信度的初始值')
    parser.add_argument('--ema-decay-stats', default=0.9, type=float,
                        help='[自适应阈值] 类别统计信息 EMA 更新的衰减率')
    # <<<----------------------------------->>>

    args = parser.parse_args()
    global best_acc

    # --- 分布式训练和设备设置 ---
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    args.device = device

    # --- 日志设置 ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        f"进程排名: {args.local_rank}, 设备: {args.device}, n_gpu: {args.n_gpu}, "
        f"分布式训练: {bool(args.local_rank != -1)}, 16位训练: {args.amp}")
    logger.info(dict(args._get_kwargs()))

    # --- 设置随机种子 ---
    if args.seed is not None:
        set_seed(args)

    # --- 创建输出目录和 TensorBoard Writer ---
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    # --- 数据集和模型特定设置 ---
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext': # 添加 ResNeXt 的例子 (如果需要)
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8 # CIFAR100 通常用更宽的模型
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    # --- 模型创建函数 ---
    # (放到这里，因为它依赖 num_classes 等参数)
    def create_model(args):
        """ 根据参数创建模型 """
        if args.arch == 'wideresnet':
            import models.wideresnet as models # 确保你的项目中有这个文件
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0, # FixMatch 通常不用 dropout
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models # 确保你的项目中有这个文件
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        else:
             raise ValueError(f"未知的模型架构: {args.arch}")
        logger.info("总参数量: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    # --- 数据加载 ---
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier() # 等待主进程完成数据集下载等操作
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data') # 确保 DATASET_GETTERS 定义在 dataset/cifar.py 或类似文件中
    if args.local_rank == 0:
        torch.distributed.barrier() # 主进程完成，通知其他进程

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu, # 无标签批次大小
        num_workers=args.num_workers,
        drop_last=True)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset), # 测试时使用顺序采样器
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # --- 模型初始化 ---
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    model = create_model(args)
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)

    # --- 优化器和学习率调度器 ---
    no_decay = ['bias', 'bn'] # 通常不对偏置和 BatchNorm 的 gamma/beta 应用权重衰减
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    # 计算预热步数
    warmup_steps = args.warmup * args.eval_step if args.warmup < 1 else args.warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, args.total_steps)

    # --- EMA 模型 ---
    if args.use_ema:
        from models.ema import ModelEMA # 确保你的项目中有这个文件
        ema_model = ModelEMA(args, model, args.ema_decay)
    else:
        ema_model = None # 如果不使用 EMA，设为 None

    args.start_epoch = 0
    # <<<--- [修改] 初始化移到 main 函数，以便传递给 train ---
    # <<<--- [新增] 初始化自适应阈值状态 --->>>
    class_avg_conf = torch.ones(args.num_classes, device=args.device) * args.initial_class_threshold
    class_confusion = torch.zeros(args.num_classes, device=args.device)
    # <<<----------------------------------->>>

    # --- 从 Checkpoint 恢复 ---
    if args.resume:
        # ... (恢复逻辑，确保也恢复 class_avg_conf 和 class_confusion, 同上一版) ...
        logger.info("==> 从检查点恢复..")
        assert os.path.isfile(args.resume), f"错误: 找不到检查点文件 {args.resume}!"
        args.out = os.path.dirname(args.resume)  # 恢复时，输出目录应与检查点一致
        checkpoint = torch.load(args.resume, map_location=args.device)  # 加载到当前设备
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        # 加载模型状态 (处理分布式和非分布式情况)
        model_state_dict = checkpoint['state_dict']
        if args.local_rank != -1 and not hasattr(model, 'module'):
             # 如果是分布式训练但保存的模型不是 DDP 包装的
             from collections import OrderedDict
             new_state_dict = OrderedDict()
             for k, v in model_state_dict.items():
                 name = 'module.' + k # 添加 'module.' 前缀
                 new_state_dict[name] = v
             model_state_dict = new_state_dict
        elif args.local_rank == -1 and hasattr(model, 'module'):
             # 如果是非分布式训练但保存的模型是 DDP 包装的
             from collections import OrderedDict
             new_state_dict = OrderedDict()
             for k, v in model_state_dict.items():
                 name = k[7:] # 移除 'module.' 前缀
                 new_state_dict[name] = v
             model_state_dict = new_state_dict

        model.load_state_dict(model_state_dict)

        if args.use_ema and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
            # 加载 EMA 模型状态 (同样需要处理 'module.' 前缀)
            ema_state_dict = checkpoint['ema_state_dict']
            if hasattr(ema_model.ema, 'module') and not list(ema_state_dict.keys())[0].startswith('module.'):
                 ema_state_dict = OrderedDict(('module.' + k, v) for k, v in ema_state_dict.items())
            elif not hasattr(ema_model.ema, 'module') and list(ema_state_dict.keys())[0].startswith('module.'):
                 ema_state_dict = OrderedDict((k[7:], v) for k, v in ema_state_dict.items())
            ema_model.ema.load_state_dict(ema_state_dict)

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # <<<--- [修改] 确保检查点恢复加载 class_avg_conf 和 class_confusion --->>>
        if 'class_avg_conf' in checkpoint:
            class_avg_conf = checkpoint['class_avg_conf'].to(args.device)
            logger.info("从检查点加载 class_avg_conf")
        if 'class_confusion' in checkpoint:
            class_confusion = checkpoint['class_confusion'].to(args.device)
            logger.info("从检查点加载 class_confusion")
        # <<<-------------------------------------------------------------------->>>
        logger.info(f"成功从 epoch {args.start_epoch} 恢复")


    # --- AMP 混合精度训练设置 ---
    if args.amp:
        # 检查 Apex 是否安装
        try:
            from apex import amp
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.opt_level)
            logger.info(f"使用 Apex AMP, 优化等级: {args.opt_level}")
        except ImportError:
            logger.error("未找到 Apex. 请安装 Apex (https://www.github.com/nvidia/apex) 来使用 AMP.")
            args.amp = False # 禁用 AMP

    # --- 分布式模型包装 ---
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True) # find_unused_parameters 可能需要根据模型调整

    # --- 开始训练 ---
    logger.info("***** 开始训练 *****")
    logger.info(f"  任务 = {args.dataset}@{args.num_labeled}")
    logger.info(f"  总 Epoch 数 = {args.epochs}")
    logger.info(f"  每个 GPU 的批次大小 = {args.batch_size}")
    logger.info(f"  总训练批次大小 = {args.batch_size * args.world_size * (1 + args.mu)}") # L + U
    logger.info(f"  总优化步数 = {args.total_steps}")

    model.zero_grad()
    # 将自适应阈值状态传入 train 函数
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, class_avg_conf, class_confusion)

# 生成带时间的文件名，例如：dynamic_thresholds_20250430_142305.jsonl
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"dynamic_thresholds_{timestamp}.jsonl"

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, class_avg_conf, class_confusion):
    """ 训练函数 - 实现按类别动态阈值 """
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    # class_avg_conf 和 class_confusion 作为参数传入，并在 epoch 结束时更新

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        # set_epoch 在 epoch 循环开始时设置

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        dynamic_threshold_meter = AverageMeter() # 仍然记录应用于 batch 的阈值的平均值

        epoch_pseudo_probs = []
        epoch_pseudo_targets = []

        if args.world_size > 1:
            labeled_trainloader.sampler.set_epoch(epoch)
            unlabeled_trainloader.sampler.set_epoch(epoch)

        # <<<--- [修改] 在 Epoch 开始时计算当前 Epoch 使用的类别阈值 --->>>
        # 注意：这里使用的是上一个 epoch 结束时更新的 class_avg_conf 和 class_confusion
        # 在 rank 0 计算，然后广播
        if args.local_rank in [-1, 0]:
            epoch_dynamic_thresholds = args.global_thresh_base + \
                                     args.class_conf_factor * class_avg_conf + \
                                     args.confusion_factor * class_confusion
            # 将阈值限制在 [0, 1] 范围内
            epoch_dynamic_thresholds.clamp_(0.0, 1.0)
        else:
            # 其他进程初始化一个同样大小的 placeholder
            epoch_dynamic_thresholds = torch.zeros(args.num_classes, device=args.device)

        # 广播计算好的类别阈值向量
        epoch_dynamic_thresholds = distributed_broadcast(epoch_dynamic_thresholds, args, src=0)
        # <<<------------------------------------------------------------>>>
        # print(epoch_dynamic_thresholds)
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])

        # --- Batch 循环 ---
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(epoch + labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except StopIteration:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(epoch + unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            u_batch_size = inputs_u_w.shape[0]
            inputs_x, targets_x = inputs_x.to(args.device), targets_x.to(args.device)
            inputs_u_w, inputs_u_s = inputs_u_w.to(args.device), inputs_u_s.to(args.device)

            # --- 前向传播 ---
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 1 + args.mu + args.mu) # 调整这里的 size
            logits = model(inputs)
            logits = de_interleave(logits, 1 + args.mu + args.mu) # 调整这里的 size
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            # --- 有标签损失 ---
            Lx = F.cross_entropy(logits_x, targets_x.to(torch.int64), reduction='mean')

            # --- 无标签损失 ---
            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)

            # <<<--- [修改] 使用预先计算好的类别阈值 --->>>
            if u_batch_size > 0:
                # 1. 根据伪标签类别，从 epoch_dynamic_thresholds 中查找对应的阈值
                thresholds_for_batch = epoch_dynamic_thresholds[targets_u] # shape: [u_batch_size]

                # 2. 使用查找到的阈值生成 mask
                mask = max_probs.ge(thresholds_for_batch).float() # shape: [u_batch_size]
            else:
                mask = torch.zeros(0, device=args.device) # 空 mask
                thresholds_for_batch = torch.zeros(0, device=args.device) # 空阈值
            # <<<--------------------------------------->>>

            # --- 收集伪标签信息 (逻辑不变) ---
            if u_batch_size > 0:
                epoch_pseudo_probs.append(max_probs.detach().cpu())
                epoch_pseudo_targets.append(targets_u.detach().cpu())

            # --- 计算无标签损失 (逻辑不变) ---
            if u_batch_size > 0:
                targets_u_long = targets_u.to(torch.int64)
                Lu = (F.cross_entropy(logits_u_s, targets_u_long, reduction='none') * mask).sum() / (mask.sum() + 1e-5)
            else:
                Lu = torch.tensor(0.0, device=args.device)

            # --- 总损失 ---
            loss = Lx + args.lambda_u * Lu

            # --- 反向传播和优化 (逻辑不变) ---
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            if u_batch_size > 0:
                losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            # --- 更新时间和统计信息 ---
            batch_time.update(time.time() - end)
            end = time.time()
            if u_batch_size > 0:
                mask_probs.update(mask.mean().item())
                dynamic_threshold_meter.update(thresholds_for_batch.mean().item()) # 记录实际应用的阈值均值
            else:
                mask_probs.update(0.0)
                # 如果没有无标签数据，记录一个默认值或基础阈值
                dynamic_threshold_meter.update(args.global_thresh_base)

            # --- 更新进度条 (显示信息不变) ---
            if not args.no_progress:
                p_bar.set_description(
                   "训练 Epoch: {epoch}/{epochs:4}. 迭代: {batch:4}/{iter:4}. LR: {lr:.4f}. "
                   "数据: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. "
                   "Mask: {mask:.2f}. DynThr: {dyn_thr:.3f}".format( # DynThr 显示的是 batch 中阈值的平均值
                        epoch=epoch + 1, epochs=args.epochs, batch=batch_idx + 1, iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0], data=data_time.avg, bt=batch_time.avg,
                        loss=losses.avg, loss_x=losses_x.avg, loss_u=losses_u.avg,
                        mask=mask_probs.avg, dyn_thr=dynamic_threshold_meter.avg
                    )
                )
                p_bar.update()

        # --- Epoch 结束 ---
        if not args.no_progress:
            p_bar.close()







        # <<<--- [新增/修改] Epoch 结束时更新类别统计信息 (这部分逻辑不变, 但很重要) --->>>
        # (这部分代码与上一个版本相同，负责收集、聚合、计算新统计量并 EMA 更新 class_avg_conf, class_confusion)
        if len(epoch_pseudo_probs) > 0:
            all_probs = torch.cat(epoch_pseudo_probs)
            all_targets = torch.cat(epoch_pseudo_targets)

            if args.world_size > 1:
                # 分布式收集 (使用占位符或实际实现)
                gathered_probs, prob_sizes = distributed_gather(all_probs.to(args.device), args, args.world_size)
                gathered_targets, target_sizes = distributed_gather(all_targets.to(args.device), args, args.world_size)
                if args.local_rank in [-1, 0]:
                    # 在 rank 0 处理聚合数据 (可能需要移除填充)
                    all_probs = gathered_probs.cpu()
                    all_targets = gathered_targets.cpu()
                    # logger.info(f"Epoch {epoch+1} Rank 0: 收集到 {all_targets.shape[0]} 个伪标签样本更新统计")
            else:
                pass # 非分布式

            if args.local_rank in [-1, 0] and all_targets.numel() > 0:
                new_class_avg_conf = torch.zeros(args.num_classes)
                new_class_confusion = torch.zeros(args.num_classes)
                counts = torch.zeros(args.num_classes, dtype=torch.long)

                for c in range(args.num_classes):
                    class_mask = (all_targets == c)
                    counts[c] = class_mask.sum()
                    if counts[c] > 0:
                        probs_c = all_probs[class_mask]
                        new_class_avg_conf[c] = probs_c.mean()
                        if counts[c] > 1:
                            new_class_confusion[c] = torch.std(probs_c, unbiased=True).clamp(0.0, 1.0)
                        else:
                            new_class_confusion[c] = 0.0

                new_class_avg_conf = new_class_avg_conf.to(args.device)
                new_class_confusion = new_class_confusion.to(args.device)
                counts = counts.to(args.device)
                has_samples_mask = counts > 0
                decay = args.ema_decay_stats

                class_avg_conf[has_samples_mask] = decay * class_avg_conf[has_samples_mask] + \
                                                   (1 - decay) * new_class_avg_conf[has_samples_mask]
                class_confusion[has_samples_mask] = decay * class_confusion[has_samples_mask] + \
                                                    (1 - decay) * new_class_confusion[has_samples_mask]

            if args.world_size > 1:
                class_avg_conf = distributed_broadcast(class_avg_conf, args, src=0)
                class_confusion = distributed_broadcast(class_confusion, args, src=0)

        del epoch_pseudo_probs, epoch_pseudo_targets

        # --- 模型评估 (逻辑不变) ---
        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            # --- TensorBoard 日志记录 (逻辑不变) ---
            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('train/5.avg_dynamic_threshold', dynamic_threshold_meter.avg, epoch) # 记录 batch 平均阈值
            # (可选) 记录 epoch 开始时计算的类别阈值
            # for c in range(args.num_classes):
            #     args.writer.add_scalar(f'thresholds/class_{c}', epoch_dynamic_thresholds[c].item(), epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            # --- 保存模型检查点 (逻辑不变, 但确保保存了 class_avg_conf, class_confusion) ---
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            model_to_save = model.module if hasattr(model, "module") else model
            ema_to_save = None
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'class_avg_conf': class_avg_conf.cpu(), # 保存更新后的状态
                'class_confusion': class_confusion.cpu(), # 保存更新后的状态
            }, is_best, args.out)
            
            save_epochs = [0, 50, 100, 200, 300]
            if epoch in save_epochs:
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'class_avg_conf': class_avg_conf.cpu(), # 保存更新后的状态
                'class_confusion': class_confusion.cpu(), # 保存更新后的状态
            }, is_best, args.out, filename=f"epoch_{epoch}_checkpoint.pth.tar")

            test_accs.append(test_acc)
            logger.info('Epoch: {}/{}'.format(epoch + 1, args.epochs))
            logger.info('当前 Epoch Top-1 Acc: {:.2f}'.format(test_acc))
            logger.info('最佳 Top-1 Acc: {:.2f}'.format(best_acc))
            logger.info('平均动态阈值 (Batch均值): {:.3f}'.format(dynamic_threshold_meter.avg)) # 这个值是 batch 阈值的均值
            logger.info('最近 20 个 Epoch 平均 Top-1 Acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))

            data = {
                "epoch": epoch,
                "loss": losses.avg,
                "loss_x": losses_x.avg,
                "loss_u": losses_u.avg,
                "Top-1 Acc":test_acc,
                "epoch_dynamic_thresholds": epoch_dynamic_thresholds.tolist()

            }

            # 写入 JSON Lines 文件（每次追加一行）
            with open(filename, "a") as f:
                f.write(json.dumps(data) + "\n")


    # --- 训练结束 ---
    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    """ 测试函数 """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # --- 准备测试 ---
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0]) # 只在主进程显示进度条

    with torch.no_grad(): # 测试时不需要计算梯度
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval() # 设置模型为评估模式

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            # --- 前向传播 ---
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets.to(torch.int64)) # 计算损失

            # --- 计算准确率 ---
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5)) # 使用 accuracy 工具函数
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            # --- 更新进度条 ---
            if not args.no_progress:
                test_loader.set_description("测试 迭代: {batch:4}/{iter:4}. 数据: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close() # 关闭进度条

    # --- 打印测试结果 ---
    logger.info(" * Top-1 Acc {top1.avg:.3f}".format(top1=top1))
    logger.info(" * Top-5 Acc {top5.avg:.3f}".format(top5=top5))
    return losses.avg, top1.avg # 返回平均损失和 Top-1 准确率


if __name__ == '__main__':
    main()
