import os
import resource
import torch
import warnings
import random
import torch.backends.cudnn as cudnn

def Config_Environment(args):
    # increase the limit of resources to make sure it can run under any conditions
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    # config gpu settings
    choose = args.choose
    if choose is not None and args.nodes_num == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = choose
        print("Current we choose gpu:%s" % choose)
    use_cuda = torch.cuda.is_available()
    print("Cuda status ", use_cuda)
    ngpus_per_node = torch.cuda.device_count()
    print("in total we have ", ngpus_per_node, " gpu")
    if ngpus_per_node <= 0:
        print("We do not have gpu supporting, exit!!!")
        exit()
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    #init random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    return ngpus_per_node