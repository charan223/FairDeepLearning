from os import getenv
import distutils
import argparse
from utils.utils import add_argument
from utils.log_utils import save_args, load_args


def add_model_args(parser):

    add_argument(parser, '--arch', type=str,
                 choices=['ffvae', 'conv', 'cfair', 'cfair-eo', 'mlp', 'laftr-dp',
                          'laftr-eqopp0', 'laftr-eqopp1', 'laftr-eqodd', 'ffvae_cfair', 'ffvae_laftr'],
                 default='ffvae'
                 )

    add_argument(parser, '--fair-coeff', type=float, default=1.,
                 help='(default: 1.)')
    add_argument(parser, '--aud-steps', type=int, default=2,
                 help='(default: 2)')

    add_argument(parser, '--adv-coeff', type=float, default=10.,
                 help='(default: 10.)')

    add_argument(parser, '--gamma', type=float, default=0.5,
                 help='gamma in paper (default: 10.)')
    add_argument(parser, '--alpha', type=float, default=1,
                 help='alpha in paper (default: 10.)')


def add_shared_args(parser):
    '''Shared training settings'''

    parser.add_argument_group('Shared parameters')

    add_argument(parser, '--sensattr', default='bck', type=str,
                 help='sensitive attribute')

    add_argument(parser, '--beta_1',
                 help='background blue ratio for evens in training phase, \
                 for multiple colors input a list [0.5, 0.4, 0.1], indicating 0.5 of even are first color, 0.4 of even are second, so on')
    add_argument(parser, '--beta_2',
                 help='background blue ratio for odds in training phase, \
                 for multiple colors input a list [0.5, 0.4, 0.1], indicating 0.5 of even are first color, 0.4 of even are second, so on')
    add_argument(parser, '--useshade', action='store_false', default=True,
                 help='use shade bool val')

    add_argument(parser, '--green_yellow', action='store_false', default=True,
                 help='green_yellow bool val')
    add_argument(parser, '--egr', help='position ratio for even, \
                 for multiple positions input a list [0.5, 0.4, 0.1], indicating 0.5 of even have box in first position, 0.4 of even have it in second, so on')
    add_argument(parser, '--ogr', help='position ratio for odd, \
                 for multiple positions input a list [0.5, 0.4, 0.1], indicating 0.5 of even have box in first position, 0.4 of even have it in second, so on')
    add_argument(parser, '--adult_threshold', type=int,
                 help='threshold of sensitive attribute')

    add_argument(parser, '--seed', type=int, default=0, metavar='S',
                 help='random seed (default: 0)')
    add_argument(parser, '--data', type=str, choices=['clr-mnist', 'adult'], default='clr-mnist',
                 help='dataset name (default: clr-mnist)')

    add_argument(parser, '--num-classes', type=int, default=2,
                 help='number of classes fixed 2')
    add_argument(parser, '--num-groups', type=int, default=2,
                 help='number of groups, should be less <= 13 as the 4x4 box doesnt have anymore positions in the image,\
                      to support more groups use smaller square boxes')
    add_argument(parser, '--input-dim', type=int, default=3072,
                 help='number of inputs 3 * 32 * 32')
    add_argument(parser, '--edepth', type=check_positive, default=2,
                 help='Encoder MLP depth as in depth*[width]')
    add_argument(parser, '--ewidths', type=check_positive, default=32,
                 help='Encoder MLP width')
    add_argument(parser, '--cdepth', type=int, default=2,
                 help='Classifier MLP depth as in depth*[width]')
    add_argument(parser, '--cwidths', type=check_positive, default=32,
                 help='Classifier MLP width')
    add_argument(parser, '--adepth', type=check_positive, default=2,
                 help='Auditor MLP depth as in depth*[width]')
    add_argument(parser, '--awidths', type=check_positive, default=32,
                 help='Auditor MLP width')
    add_argument(parser, '--zdim', type=check_positive, default=16,
                 help='All MLPs has this as input or output (default: 16)')
    add_argument(parser, '--bck', default='blue-red', type=str, choices=['blue-red', 'multi-color', 'black'],
                 help='bck value')

    add_argument(parser, '--replicate', type=int, default=1,
                 help='(default: 1)')
    add_argument(parser, '--batch-size', type=int, default=64, metavar='N',
                 help='input batch size for training (default: 64)')
    add_argument(parser, '--test-batch-size', type=int, default=1000, metavar='N',
                 help='input batch size for testing (default: 1000) - just used in debugging')
    add_argument(parser, '--epochs', type=int, default=150, metavar='N',
                 help='Max number of epochs to train (default: 150)')
    add_argument(parser, '--vae-epochs', type=int, default=30, metavar='N',
                 help='Max number of epochs to train (default: 30)')
    add_argument(parser, '--patience', default=5, type=check_positive,
                 help='patience for early stopping shared between all metrics (default: 5)')
    add_argument(parser, '--estopping', action='store_false', default=True,
                 help='early stopping or not')

    add_argument(parser, '--no-cuda', action='store_true', default=False,
                 help='disables CUDA training (default: False)')
    add_argument(parser, '--log-interval', type=int, default=100, metavar='N',
                 help='batches to wait before logging training status (default: 100)')
    add_argument(parser, '--no-save-model', action='store_true', default=False,
                 help='For Saving the current model at each epoch (default: False)')

    add_argument(parser, '--data-dir', default='./data',
                 help='Data directory (default: ./data)')
    add_argument(parser, '--output-dir', default=getenv('PT_OUTPUT_DIR', './output'),
                 help='Output directory (default: ./output or $PT_OUTPUT_DIR)')

    add_argument(parser, '--name', default=None,
                 help='experiment folder name. Date and time if None.')
    add_argument(parser, '--load-model', type=str,
                 help='Set checkpoint address to load when using load.py.')

    add_argument(parser, '--label_type', default='even-odd', type=str,
                 help='label type')

    add_argument(parser, '--wandb-init', type=str, default="./scripts/wandb_init.json", metavar='N',
                 help='file specifiying wandb init arguments e.g the API KEY (default: ./wandb_init.json)')
    add_argument(parser, '--wandb_name', type=str, default='deepfairness-0', metavar='N',
                 help='name to identify this wandb run')
    add_argument(parser, '--ifwandb', dest='ifwandb',
                 type=lambda x: bool(distutils.util.strtobool(x)),
                 help='Flag to activate/deactivate wandb logging')


def parse_args(input_args=None, parse_known=False):
    parser = argparse.ArgumentParser(
        description='Benchmarking Fair Deep Learning Algorithms')
    add_shared_args(parser)
    add_model_args(parser)
    args, _ = parser.parse_known_args(input_args)
    args = parser.parse_args(input_args)
    return args


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue
