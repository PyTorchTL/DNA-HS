import argparse

DATASETS = ['Fatigue', 'SEED']
MODELS = ['vgg11, vgg13, vgg16, vgg19, vgg16_normed']

DATA_PARAMS = {
    'Fatigue.num_sample': 1400,
    'Fatigue.num_feats': 1647,
    'Fatigue.num_nodes': 61,  # channels
    'Fatigue.num_freq': 27,
    'Fatigue.num_classes': 2,  # classes of label
    'SEED.num_sample': 842,
    'SEED.num_feats': 2604,
    'SEED.num_nodes': 62,  # channels
    'SEED.num_freq': 42,
    'SEED.num_classes': 3,  # classes of label
}

parser = argparse.ArgumentParser(description='Classify SEED')
parser.add_argument('-device', default='cuda:0', help='device')
parser.add_argument('-cupy', action='store_true', default=0, help='use CUDA neuron and multi-step forward mode')
parser.add_argument('-b', default=32, type=int, help='batch size')
parser.add_argument('-indep_epochs', default=0, type=int, metavar='N', help='number of indep_Train epochs to run')
parser.add_argument('-epochs', default=50, type=int, metavar='N', help='number of total epochs to run') # 50
parser.add_argument('-j', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-flod', default=5, type=int, metavar='N', help='k-flod')
parser.add_argument('-LOSO', default='', type=str, metavar='N', help='LOSO')

# train para
parser.add_argument('--seed', type=int, default=2022)  ## seed must to be identical with the seed used in the super-network training
parser.add_argument('--search_seed', type=int, default=99)

# evoluation search setting
parser.add_argument('--GA_iter', type=int, default=49)  # The GA algorithm is introduced when the co-train is traversed a certain number of times, 25, 49（mid=GA_iter）
parser.add_argument('--GA_epoch', type=int, default=15)  # Number of GA algorithm iterations,15,10,5
parser.add_argument('--num_mutation', type=int, default=10)  # Number of mutations,10
parser.add_argument('--mutation_prob', type=float, default=0.5)  # [0,0.5]
parser.add_argument('--num_crossover', type=int, default=10)
parser.add_argument('--num_topk', type=int, default=5)  # select top5
parser.add_argument('--return_topk', type=int, default=1)  # Finally return the highest
parser.add_argument('--fit', default=1, type=float)  # fitness=train+fit*test

# SA_architectures
parser.add_argument('-a', '--arch', default='EEGNet', type=str, help='network architecture', choices=['vgg5', 'cnn', 'alexnet', 'resnet18', 'EEGNet', 'capsnet', 'DGCN']) # cnn=ANN
parser.add_argument('--pretrained_ann', default='', type=str, help='pretrained ANN model')
parser.add_argument('--pretrained_snn', default='', type=str, help='pretrained SNN for inference')
parser.add_argument('--activation', default='Linear', type=str, help='SNN activation function', choices=['Linear', 'STDB'])
parser.add_argument('--scalePara', default=0.3, type=float, help='alpha for weighting para_weight, default: 0.2')

# 参数
parser.add_argument('--alpha', default=0.3, type=float, help='parameter alpha for STDB')
parser.add_argument('--beta', default=0.01, type=float, help='parameter beta for STDB')
parser.add_argument('--timesteps', default=6, type=int, help='simulation timesteps')
parser.add_argument('--leak', default=0.25, type=float, help='membrane leak')
parser.add_argument('--scaling_factor', default=1, type=float, help='scaling factor for thresholds at reduced timesteps')
parser.add_argument('--default_threshold', default=0.25, type=float, help='intial threshold to train SNN from scratch')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout percentage for conv layers')
parser.add_argument('--kernel_size', default=3, type=int, help='filter size for the conv layers')
# parser.add_argument('--test_only', action='store_true', help='perform only inference')
parser.add_argument('--log', action='store_true', help='to print the output on terminal or to log file')
parser.add_argument('-init_tau', default=2.0, type=float)

# datasets
parser.add_argument('--dataset', type=str, default='SEED', choices=DATASETS, help='name of dataset;')
parser.add_argument("--num_classes", type=int, default=3, help="")
# 第一层卷积的输出通道数
parser.add_argument('-channels', default=64, type=int, help='default:128')
parser.add_argument('-num_nodes', default=64, type=int, help='default:128')
parser.add_argument('-channel_2Dto1D', default=64, type=int, help='default:128')
# 第一层卷积的输入通道数
parser.add_argument('-input_channels', default=5, type=int)
# 卷积
parser.add_argument('-input_shape1', default=8, type=int, help='default:128')
parser.add_argument('-input_shape2', default=9, type=int, help='default:128')
# FC通道数
parser.add_argument('-channels_FC1', default=5, type=int, help='default:128')
parser.add_argument('-channels_FC2', default=3, type=int, help='default:128')
parser.add_argument('-channels_Alex1', default=3, type=int, help='default:128')
parser.add_argument('-channels_Alex2', default=3, type=int, help='default:128')
parser.add_argument('-avgpool_Alex', default=3, type=int, help='default:128')
parser.add_argument('-channels_ANN1', default=3, type=int, help='default:128')
parser.add_argument('-channels_ANN2', default=3, type=int, help='default:128')
parser.add_argument('-channels_alexnet1', default=3, type=int, help='default:128')
parser.add_argument('-channels_alexnet2', default=3, type=int, help='default:128')
parser.add_argument('-channels_eegnet', default=3, type=int, help='default:128')
parser.add_argument('-num_capsnet', default=3, type=int, help='default:128')
# 层内横向交互
parser.add_argument('-if_lateral', type=bool, default=True)
# 数据集路径
parser.add_argument('-data_dir', default=" ", type=str)
# 输出路径
parser.add_argument('-out_dir', default="./logs/", type=str)

# log目录
# parser.add_argument('-desc', default="SEED_DE", type=str)
parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
parser.add_argument('-opt', default='Adam', type=str, help='use which optimizer. SGD or Adam')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
parser.add_argument('-step_size', default=32, type=float, help='step_size for StepLR')
parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
parser.add_argument('-T_max', default=32, type=int, help='T_max for CosineAnnealingLR')


def get_config():

    config = parser.parse_args()
    # config['data_params'] = DATA_PARAMS
    return config
