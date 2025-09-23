import argparse
import os
import codeLib
import torch
from .training import Trainer
from . import dataset
from .sgfn import SGFN
from .sgpn import SGPN
from .jointSG import JointSG
from .imp import IMP
from .esgnn import ESGNN
from termcolor import cprint
__all__ = ['SGFN', 'SGPN', 'dataset',
           'Trainer', 'IMP', 'JointSG', 'ESGNN']


def default_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # General setup
    parser.add_argument('-c', '--config', type=str, default='./configs/config_default.yaml',
                        help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'validation', 'trace', 'eval',
                        'sample', 'trace', 'interference'], default='train', help='mode. can be [train,trace,eval, interference]', required=False)
    parser.add_argument('--loadbest', type=int, default=0, choices=[
                        0, 1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--log', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL'], help='')
    parser.add_argument('-o', '--out_dir', type=str, default='',
                        help='overwrite output directory given in the config file.')
    parser.add_argument('--dry_run', action='store_true',
                help='disable logging in wandb (if that is the logger).')
    
    # Save dataset to cache before training
    parser.add_argument('--cache', action='store_true',
                        help='load data to RAM.')
    
    # Training setting: From scratch or from latest epoch:
    parser.add_argument("--from-scratch", action='store_true')
    
    # Set up model
    # parser.add_argument("--gpu", nargs="?", default=0, help="GPU ID: 0,1")
    # parser.add_argument("--model", type=str, default="sgfn", help="sgfn or esgnn")
    # parser.add_argument("--num-layers", type=int, default=2)
    # parser.add_argument("--with-x", action='store_true')
    # parser.add_argument("--without-fan", action='store_true')
    # parser.add_argument("--fan2gcl1", action='store_true')
    # parser.add_argument("--dropout-rate", type=int, default=30, help="dropout rate 10/20/30/... over 100")
    # parser.add_argument("--is-toy", action='store_true', help="playing around with different settings and architectures")
    
    # Set up logger
    parser.add_argument("--run-id", type=int, default=-1, help="Create new run id with -1. Otherwise run with parsed id.")
    parser.add_argument("--project-name", type=str, default="")
    
    return parser


def load_config(args):
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise RuntimeError(
            'Targer config file does not exist. {}'.format(config_path))

    # load config file
    config = codeLib.Config(config_path)
    # configure config based on the input arguments
    config.config_path = config_path
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    if len(args.out_dir) > 0:
        config.training.out_dir = args.out_dir
    if args.dry_run:
        config.wandb.dry_run = True
    if args.cache:
        config.data.load_cache = True
    cprint(f"load cache: {config.data.load_cache}", "yellow")
    config.GPU = config.GPU #[args.gpu]

    # Check if name exist
    if 'name' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['name'] = name

    # Init device
    if torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device(f"cuda:{config.GPU[0]}")
    else:
        config.DEVICE = torch.device("cpu")

    config.log_level = args.log
    
    # Training setting
    if args.from_scratch:
        config.TRAIN_FROM_SCRATCH = True
    
    # Change model architecture
    # if args.model == 'sgfn':
    #     config.model.gnn.method = 'fan'
    # elif args.model == 'esgnn':
    #     config.model.gnn.method = 'esgnn'
    # elif args.model == '3dssg':
    #     config.model.gnn.method = 'triplet'
    # elif args.model == 'eqgnn':
    #     config.model.gnn.method = 'eqgnn'
        
    # config.model.gnn.num_layers = args.num_layers
    # model_name = f"{args.model}_{config.model.gnn.num_layers}"
    model_name = f"{config.model.method}_{config.model.gnn.num_layers}"
    # config.model.gnn.with_x = args.with_x
    if config.model.gnn.with_x:
        model_name = f"{model_name}X"
        
    if not config.model.gnn.with_fan:
        # config.model.gnn.with_fan = False
        model_name = f"{model_name}_nofan"
        
    if config.model.gnn.fan2_gcl1:
        # config.model.gnn.fan2_gcl1 = True
        model_name = f"{model_name}_fan2gcl1"
        
    # if config.is_toy:
    #     model_name = f"{model_name}_toy"
        
    if config.model.gnn.drop_out != 30:
        # config.model.gnn.drop_out = config.model.gnn.drop_out / 100
        model_name = f"{model_name}_dropout{round(config.model.gnn.drop_out, 2)}"
        
    config.SAVED_MODEL_NAME = model_name
    cprint("Model used: {}".format(config.SAVED_MODEL_NAME), "cyan")
    # Change Wandb log
    config.wandb.name = model_name
    if args.run_id== -1:
        config.wandb.id = args.run_id
    else:
        config.wandb.id = str(args.run_id).zfill(4)
    
    if args.project_name != "":
        config.wandb.project = args.project_name
    return config


def Parse():
    r"""loads model config
    """
    args = default_parser().parse_args()
    return load_config(args)
