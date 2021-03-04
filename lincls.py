#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import warnings

warnings.filterwarnings('ignore')
import argparse
import builtins
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# from Data_Processing.VOC_Dataset import PascalVOC_Dataset
# from Data_Processing.VOC_utils import get_ap_score
from data_processing.loader import GaussianBlur
from ops.os_operation import mkdir
from training.train_utils import accuracy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=10., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[15, 25, 30], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')  # default is for places205
parser.add_argument('--cos', type=int, default=1,
                    help='use cosine lr schedule')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', type=int, default=1,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--choose', type=str, default=None, help="choose gpu for training")
parser.add_argument("--use_swav", type=int, default=0, help="use swav model or not")
parser.add_argument("--dataset", type=str, default="ImageNet", help="which dataset is used to finetune")
parser.add_argument("--aug", type=int, default=0, help="use augmentation or not during fine tuning")
parser.add_argument("--size_crops", type=int, default=[224, 192, 160, 128, 96], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.2, 0.172, 0.143, 0.114, 0.086], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1.0, 0.86, 0.715, 0.571, 0.429], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--add_crop", type=int, default=0, help="use crop or not in our training dataset")
parser.add_argument("--strong", type=int, default=0, help="use strong augmentation or not")
parser.add_argument("--final_lr", type=float, default=0.01, help="ending learning rate for training")
parser.add_argument("--aug_type", type=int, default=0, help="augmentation type for our condition")
parser.add_argument('--save_path', default="", type=str, help="model and record save path")
parser.add_argument('--log_path', type=str, default="train_log", help="log path for saving models")
parser.add_argument("--nodes_num", type=int, default=1, help="number of nodes to use")
parser.add_argument("--ngpu", type=int, default=8, help="number of gpus per node")
parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="addr for master node")
parser.add_argument("--master_port", type=str, default="1234", help="port for master node")
parser.add_argument('--node_rank', type=int, default=0, help='rank of machine, 0 to nodes_num-1')
parser.add_argument("--final", default=0, type=int, help="use the final specified augment or not")
parser.add_argument("--avg_pool", default=1, type=int, help="average pool output size")
parser.add_argument("--crop_scale", type=float, default=[0.2, 1.0], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--train_strong", type=int, default=0, help="training use stronger augmentation or not")
parser.add_argument("--sgdr", type=int, default=0, help="training with warm up (1) or restart warm up (2)")
parser.add_argument("--sgdr_t0", type=int, default=10, help="sgdr t0")
parser.add_argument("--sgdr_t_mult", type=int, default=1, help="sgdr t mult")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout layer settings")
parser.add_argument("--randcrop", type=int, default=0, help="use random crop or not")
best_acc1 = 0


def main():
    args = parser.parse_args()
    choose = args.choose
    if choose is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = choose
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    params = vars(args)
    data_path = args.data  # the path stored
    args.data = data_path
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    params = vars(args)
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset == "VOC2007":
        num_classes = 20
    elif args.dataset == "Place205":
        num_classes = 205
    else:
        num_classes = 1000
    # import src.resnet50 as resnet_models
    # model = resnet_models.__dict__[args.arch](output_dim=num_classes, eval_mode=True)
    model = models.__dict__[args.arch](num_classes=num_classes)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # model.avgpool=nn.AdaptiveAvgPool2d((params['avg_pool'], params['avg_pool']))
    # model.fc=nn.Linear(2048*(params['avg_pool']**2), num_classes)
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:

        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))

            checkpoint = torch.load(args.pretrained, map_location="cpu")

            if args.use_swav:
                # remove prefixe "module."
                state_dict = checkpoint
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                for k, v in model.state_dict().items():
                    if k not in list(state_dict):
                        print('key "{}" could not be found in provided state dict'.format(k))
                    elif state_dict[k].shape != v.shape:
                        print('key "{}" is of different shape in model and provided state dict'.format(k))
                        state_dict[k] = v

            else:
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.dropout != 0.0:
        model.fc = nn.Sequential(nn.Dropout(args.dropout), model.fc)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    if args.dataset == "VOC2007":
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum').cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset == "ImageNet":
        data_path = args.data
        traindir = os.path.join(data_path, 'train')
        valdir = os.path.join(data_path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize
        # ])
        if args.train_strong:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        elif args.randcrop:
            transform_train = transforms.Compose([
                transforms.RandomCrop(224, pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize, ])

        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize, ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        # transform_test = transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(args.crop_scale[0], args.crop_scale[1])),
        #     #transforms.RandomApply([
        #     #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        #     #], p=0.8),
        #     #transforms.RandomGrayscale(p=0.2),
        #     #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize
        # ])
        transform_testfinal = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(args.crop_scale[0], args.crop_scale[1])),
            # transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        from data_processing.MultiCrop_Transform import Last_transform
        transform_testfinal = Last_transform(8, transform_testfinal)
        transform_testfinal2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(args.crop_scale[0], args.crop_scale[1])),
            # transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        from data_processing.MultiCrop_Transform import Last_transform
        transform_testfinal2 = Last_transform(8, transform_testfinal2)

        # transform_testfinal=TenCrop_transform(normalize)
        train_dataset = datasets.ImageFolder(traindir, transform_train)
        val_dataset = datasets.ImageFolder(valdir, transform_test)
        test_dataset = datasets.ImageFolder(valdir, transform_testfinal)
        test_dataset2 = datasets.ImageFolder(valdir, transform_testfinal2)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                          shuffle=True)  # different gpu forward individual based on its own statistics
            # val_sampler=None
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
            test_sampler2 = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None
            test_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, sampler=val_sampler,
            batch_size=args.batch_size, shuffle=(val_sampler is None),
            # different gpu forward is different, thus it's necessary
            num_workers=args.workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, sampler=test_sampler,
            batch_size=args.batch_size, shuffle=(test_sampler is None),
            num_workers=args.workers, pin_memory=True)

        test_loader2 = torch.utils.data.DataLoader(
            test_dataset2, sampler=test_sampler2,
            batch_size=args.batch_size, shuffle=(test_sampler2 is None),
            num_workers=args.workers, pin_memory=True)

    elif args.dataset == "Place205":
        from data_processing.Place205_Dataset import Places205
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if args.train_strong:
            if args.randcrop:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(224),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            if args.randcrop:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize, ])

            else:
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize, ])
        # waiting to add 10 crop
        transform_valid = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        if args.randcrop:
            transform_testfinal = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            transform_testfinal2 = transforms.Compose([
                transforms.RandomCrop(224),
                # transforms.RandomApply([
                #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                # ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform_testfinal = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(args.crop_scale[0], args.crop_scale[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            transform_testfinal2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(args.crop_scale[0], args.crop_scale[1])),
                # transforms.RandomApply([
                #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                # ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        from data_processing.MultiCrop_Transform import Last_transform
        transform_testfinal2 = Last_transform(8, transform_testfinal2)
        transform_testfinal = Last_transform(8, transform_testfinal)
        train_dataset = Places205(args.data, 'train', transform_train)
        valid_dataset = Places205(args.data, 'val', transform_valid)
        test_dataset = Places205(args.data, 'val', transform_testfinal)
        test_dataset2 = Places205(args.data, 'val', transform_testfinal2)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)
            #             val_sampler = None
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
            test_sampler2 = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            valid_dataset, sampler=val_sampler,
            batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, sampler=test_sampler,
            batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True)
        test_loader2 = torch.utils.data.DataLoader(
            test_dataset2, sampler=test_sampler2,
            batch_size=args.batch_size, shuffle=(test_sampler2 is None),
            num_workers=args.workers, pin_memory=True)

    else:
        print("your dataset %s is not supported for finetuning now" % args.dataset)
        exit()

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        testing(test_loader, model, criterion, args)
        testing2(test_loader2, model, criterion, args)
        return
    import datetime
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")

    save_path = os.path.join(args.save_path, args.log_path)
    log_path = os.path.join(save_path, 'Finetune_log')
    mkdir(log_path)
    log_path = os.path.join(log_path, formatted_today + now)
    mkdir(log_path)
    # model_path=os.path.join(log_path,'checkpoint.pth.tar')
    lr_scheduler = None
    if args.sgdr == 1:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 12)
    elif args.sgdr == 2:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.sgdr_t0, args.sgdr_t_mult)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.sgdr == 0:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        # if args.final:
        #    train2(train_loader, model, criterion, optimizer, epoch, args)
        # else:
        train(train_loader, model, criterion, optimizer, epoch, args, lr_scheduler)
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        if abs(args.epochs - epoch) <= 20 and args.dataset == "ImageNet":
            acc2 = testing(test_loader, model, criterion, args)
            print("###Testing acc %.5f###" % acc2)
        if abs(args.epochs - epoch) <= 10 and args.dataset == "Place205":
            acc2 = testing(test_loader, model, criterion, args)
            print("###Testing acc %.5f###" % acc2)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            # add timestamp
            tmp_save_path = os.path.join(log_path, 'checkpoint.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=tmp_save_path)

            if epoch == args.start_epoch and not args.use_swav:
                sanity_check(model.state_dict(), args.pretrained)
            
            if abs(args.epochs - epoch) <= 20:
                tmp_save_path = os.path.join(log_path, 'model_%d.pth.tar' % epoch)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, False, filename=tmp_save_path)
                


def train(train_loader, model, criterion, optimizer, epoch, args, lr_scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mAP = AverageMeter("mAP", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, mAP],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    batch_total = len(train_loader)
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # adjust_batch_learning_rate(optimizer, epoch, i, batch_total, args)

        if args.gpu is not None:
            # if args.add_crop :
            #     len_images = len(images)
            #     for k in range(len(images)):
            #         images[k] = images[k].cuda(args.gpu, non_blocking=True)
            #
            # else:
            images = images.cuda(args.gpu, non_blocking=True)

        # if args.add_crop:
        #     target = target.cuda(args.gpu, non_blocking=True)
        #     len_images = len(images)
        #     loss=0
        #     first_output=-1
        #
        #     for k in range(len_images):
        #         output=model(images[k])
        #         loss += criterion(output, target)
        #         if k==0:
        #             first_output=output
        #         if epoch == 0 and i == 0:
        #             print("%d/%d loss values %.5f" %(k,len_images, loss.item()))
        #     loss/=len_images
        #     images = images[0]
        #     output=first_output
        # else:
        #
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.sgdr != 0:
            lr_scheduler.step(epoch + i / batch_total)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train2(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mAP = AverageMeter("mAP", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, mAP],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            len_images = len(images)
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)
        len_images = len(images)

        first_output = -1
        for k in range(len_images):
            # compute gradient and do SGD step
            optimizer.zero_grad()
            output = model(images[k])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), images[k].size(0))
            if k == 0:
                first_output = output

        images = images[0]
        output = first_output

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mAP = AverageMeter("mAP", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, mAP],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    # if args.dataset == "VOC2007":
    #    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    #    transformations_valid = transforms.Compose([
    #        transforms.FiveCrop(224),
    #    ])
    # implement our own random crop
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # print(images.size())
            if args.gpu is not None and args.dataset != "VOC2007":
                # if args.add_crop:
                #     for k in range(len(images)):
                #         images[k] = images[k].cuda(args.gpu, non_blocking=True)
                # else:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # if args.add_crop>1:
            #     output_list=[]
            #     for image in images:
            #         output = model(image)
            #         output_list.append(output)
            #     output_list=torch.stack(output_list,dim=0)
            #     output_list=torch.mean(output_list,dim=0)
            #     output=output_list
            #     images=images[0]
            # else:
            # compute output
            output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = torch.mean(concat_all_gather(acc1), dim=0, keepdim=True)
            acc5 = torch.mean(concat_all_gather(acc5), dim=0, keepdim=True)
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} mAP {mAP.avg:.3f} '
              .format(top1=top1, top5=top5, mAP=mAP))

    return top1.avg


def testing(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mAP = AverageMeter("mAP", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, mAP],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    # if args.dataset == "VOC2007":
    #    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    #    transformations_valid = transforms.Compose([
    #        transforms.FiveCrop(224),
    #    ])
    correct_count = 0
    count_all = 0
    # implement our own random crop
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # if args.dataset == "VOC2007":
            #    images = transformations_valid(images)

            # print(images.size())
            if args.gpu is not None and args.dataset != "VOC2007":
                for k in range(len(images)):
                    images[k] = images[k].cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)
            output_list = []
            for image in images:
                output = model(image)
                output = torch.softmax(output, dim=1)
                output_list.append(output)
            output_list = torch.stack(output_list, dim=0)
            output_list, max_index = torch.max(output_list, dim=0)
            output = output_list
            images = images[0]
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = torch.mean(concat_all_gather(acc1), dim=0, keepdim=True)
            acc5 = torch.mean(concat_all_gather(acc5), dim=0, keepdim=True)
            correct_count += float(acc1[0]) * images.size(0)
            count_all += images.size(0)
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} mAP {mAP.avg:.3f} '
              .format(top1=top1, top5=top5, mAP=mAP))
        final_accu = correct_count / count_all
        print("$$our final calculated accuracy %.7f" % final_accu)
    return top1.avg


def testing2(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mAP = AverageMeter("mAP", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, mAP],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    # if args.dataset == "VOC2007":
    #    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    #    transformations_valid = transforms.Compose([
    #        transforms.FiveCrop(224),
    #    ])
    correct_count = 0
    count_all = 0
    # implement our own random crop
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # if args.dataset == "VOC2007":
            #    images = transformations_valid(images)

            # print(images.size())
            if args.gpu is not None and args.dataset != "VOC2007":
                for k in range(len(images)):
                    images[k] = images[k].cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)
            output_list = []
            for image in images:
                output = model(image)
                output = torch.softmax(output, dim=1)
                output_list.append(output)
            output_list = torch.stack(output_list, dim=0)
            output_list = torch.mean(output_list, dim=0)
            output = output_list
            images = images[0]
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = torch.mean(concat_all_gather(acc1), dim=0, keepdim=True)
            acc5 = torch.mean(concat_all_gather(acc5), dim=0, keepdim=True)
            correct_count += float(acc1[0]) * images.size(0)
            count_all += images.size(0)
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} mAP {mAP.avg:.3f} '
              .format(top1=top1, top5=top5, mAP=mAP))
        final_accu = correct_count / count_all
        print("$$our final average accuracy %.7f" % final_accu)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        root_path = os.path.split(filename)[0]
        best_path = os.path.join(root_path, "model_best.pth.tar")
        shutil.copyfile(filename, best_path)


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


import math


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    end_lr = args.final_lr
    # update on cos scheduler
    # this scheduler is not proper enough
    if args.cos:
        lr = 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (lr - end_lr) + end_lr
    else:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_batch_learning_rate(optimizer, cur_epoch, cur_batch, batch_total, args):
    """Decay the learning rate based on schedule"""
    init_lr = args.lr
    # end_lr=args.final_lr
    # update on cos scheduler
    # this scheduler is not proper enough
    current_schdule = 0
    # use_epoch=cur_epoch
    last_milestone = 0
    for milestone in args.schedule:
        if cur_epoch > milestone:
            current_schdule += 1
            init_lr *= 0.1
            last_milestone = milestone
        else:
            cur_epoch -= last_milestone
            break
    if current_schdule < len(args.schedule):
        all_epochs = args.schedule[current_schdule]
    else:
        all_epochs = args.epochs
    end_lr = init_lr * 0.1
    lr = math.cos(
        0.5 * math.pi * (cur_batch + cur_epoch * batch_total) / ((all_epochs - last_milestone) * batch_total)) * (
                     init_lr - end_lr) + end_lr
    if cur_batch % 50 == 0:
        print("[%d] %d/%d learing rate %.9f" % (cur_epoch, cur_batch, batch_total, lr))
    # if args.cos:
    #    lr = 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))*(lr-end_lr)+end_lr
    # else:
    #    for milestone in args.schedule:
    #        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr





# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    main()
