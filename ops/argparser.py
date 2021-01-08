import parser
import argparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def argparser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default="data", type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--log_path', type=str, default="train_log", help="log path for saving models")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        # choices=model_names,
                        type=str,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_final', default=0.0006, type=float,
                        help='final learning rate')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training,args.nodes_num*args.ngpu,here we specify with the number of nodes')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training,rank of total threads, 0 to args.world_size-1')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', type=int, default=1,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # moco specific configs:
    parser.add_argument('--moco_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.12, type=float,
                        help='softmax temperature (default: 0.12)')

    # options for moco v2
    parser.add_argument('--mlp', type=int, default=1,
                        help='use mlp head')
    parser.add_argument('--cos', type=int, default=1,
                        help='use cosine lr schedule')
    parser.add_argument('--dataset', type=str, default="ImageNet", help="Specify dataset: ImageNet or cifar10")
    parser.add_argument('--choose', type=str, default=None, help="choose gpu for training")
    parser.add_argument("--train_url", default="", type=str, help="Cloud path that specifies the output file path")
    parser.add_argument('--data_url', default="", type=str, help="Cloud path that specifies the datasets")
    parser.add_argument('--save_path', default=".", type=str, help="model and record save path")
    # idea from swav#adds crops for it
    parser.add_argument("--nmb_crops", type=int, default=[1, 1, 1, 1, 1], nargs="+",
                        help="list of number of crops (example: [2, 6])")  # when use 0 denotes the multi crop is not applied
    parser.add_argument("--size_crops", type=int, default=[224, 192, 160, 128, 96], nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.2, 0.172, 0.143, 0.114, 0.086], nargs="+",
                        help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1.0, 0.86, 0.715, 0.571, 0.429], nargs="+",
                        help="argument in RandomResizedCrop (example: [1., 0.14])")
    parser.add_argument('--cluster', type=int, default=65536, help="number of learnable comparison features")
    parser.add_argument('--memory_lr', type=float, default=3, help="learning rate for adversial memory bank")
    parser.add_argument("--ad_init", type=int, default=1, help="use feature encoding to init or not")
    parser.add_argument("--nodes_num", type=int, default=1, help="number of nodes to use")
    parser.add_argument("--ngpu", type=int, default=8, help="number of gpus per node")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="addr for master node")
    parser.add_argument("--master_port", type=str, default="1234", help="port for master node")
    parser.add_argument('--node_rank', type=int, default=0, help='rank of machine, 0 to nodes_num-1')

    parser.add_argument('--mem_t', default=0.07, type=float,
                        help='temperature for memory bank(default: 0.07)')
    parser.add_argument('--mem_wd', default=1e-4, type=float,
                        help='weight decay of memory bank (default: 0)')
    parser.add_argument("--sym", type=int, default=0, help="train with symmetric loss or not")

    return parser