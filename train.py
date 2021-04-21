import torch
import numpy as np
from exper.experiment import Experiment
from utils.dataset_properties import get_dataset_properties

dataset_list    = [
                   'CIFAR100',
                   'CIFAR10',
                   'FashionMNIST',
                   'STL10',
                   'SVHN',
                   ]

lr_list        = np.logspace(np.log10(0.25), np.log10(0.0001), 25)

p_list = [1,0.5,0.25,0.1,0.05,0.01]

histo_list = []
batchSizeList = [64, 128, 256, 512]
nc_list          = [1, 3, 5, 7, 9, 10, 12, 74, 75, 76, 78, 80, 100, 120, 140, 150, 180, 200]

epoch_list = []

for dataset_idx in range(1):
    for net_idx in range(1):
        for lr_idx in range(25):
            for prune_idx in range(6):
                for batch_idx in range(4):
                    im_size, num_classes, input_ch, size_dataset \
                    = get_dataset_properties(dataset_list[dataset_idx])
                    loader_opts  = {
                                    'dataset'           : dataset_list[dataset_idx],
                                    'loader_type'       : 'Natural',
                                    'pytorch_dataset'   : True,
                                    # 'dataset_path'      : '../../data',
                                    'dataset_path'      : '../../../../data',
                                    #'dataset_path'      : '/scratch/users/papyan/data',
                                    'im_size'           : im_size,
                                    'padded_im_size'    : 32,
                                    'num_classes'       : num_classes,
                                    'input_ch'          : input_ch,
                                    'threads'           : 4,
                                    'epc_seed'          : 0,
                                    }
                    optim_kwargs = {
    #                                'eps'               : 1.2e-03,
                                    'weight_decay'      : 5e-4,
                                    'momentum'          : 0.9,
                                    }

                    #Baseline Experiments
                    if prune_idx == 1:
                        prune_technique = "None"
                        probability_list = [0]
                        net = 'DenseNet16'
                    elif prune_idx == 2:
                        prune_technique = "ClassDropout"
                        probability_list = [0.2]*8
                        net = 'DenseNet16'
                    elif prune_idx == 0:
                        prune_technique = "Dropout"
                        probability_list = [0.2]*8
                        net = 'DenseNet16'
                    elif prune_idx == 3:
                        prune_technique = "None"
                        probability_list = []
                        net = 'Wide_ResNet'
                    elif prune_idx == 4:
                        prune_technique = "ClassDropout"
                        probability_list = [0.3, 0.3, 0.3]
                        net = 'Wide_ResNet'
                    elif prune_idx == 5:
                        prune_technique = "Dropout"
                        probability_list = [0.3, 0.3, 0.3]
                        net = 'Wide_ResNet'
                    else:
                        raise Exception

                    train_opts   = {
                                    'crit'              : 'CrossEntropyLoss',
                                    'net'               : net,
                                    'optim'             : 'SGD',
                                    'optim_kwargs'      : optim_kwargs,
                                    'epochs'            : 200,
                                    'lr'                : lr_list[lr_idx],
                                    'lr_milestones'     : [60, 120, 160],
                                    'gamma'             : 0.2,
                                    'train_batch_size'  : batchSizeList[batch_idx],
                                    'test_batch_size'   : 128,
                                    'device'            : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                    'seed'              : 0,
                                    'stats'             : ['top1', 'top5', 'loss'],
                                    'prune_technique'   : prune_technique,
                                    'droprate'          : 0,
                                    'memory_efficient'  : False,
                                    'resnet_type'       : 'small',
                                    'pretrained'        : False,
                                    'save_checkpoints'  : epoch_list,
                                    'nc_list'           : nc_list,
                                    'p_list'            : p_list,
                                    'weight_decay'      : optim_kwargs['weight_decay'],
                                    'histo_list'        : histo_list,
                                    'probability_list'  : probability_list
                                    }

                    results_opts = {
                                    'training_results_path': './results',
                                    'train_dump_file'   : 'training_results.json',
                                    }

                    opts = dict(loader_opts, **train_opts)
                    opts = dict(opts, **results_opts)

                    exp = Experiment(opts)

                    exp.run()

