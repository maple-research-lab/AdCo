#Copyright (C) 2020 Xiao Wang
#License: MIT for academic use.
#Contact: Xiao Wang (wang3702@purdue.edu, xiaowang20140001@gmail.com)

#Some codes adopted from https://github.com/facebookresearch/moco

from ops.argparser import  argparser
from ops.Config_Environment import Config_Environment
import torch.multiprocessing as mp
from training.main_worker import main_worker
def main(args):
    #config environment
    ngpus_per_node=Config_Environment(args)

    # call training main control function
    if args.multiprocessing_distributed==1:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    #use_cuda = torch.cuda.is_available()
    #print("starting check cuda status",use_cuda)
    #if use_cuda:
    parser = argparser()
    args = parser.parse_args()
    main(args)