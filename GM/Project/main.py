import argparse
import logging
import logging.config
import random
import os
import sys
import wandb
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from trainer import Trainer

# should setup root logger before importing any relevant libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream = sys.stdout
)
logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device_num', type=str, default='0'
    )
    parser.add_argument(
        '--patience', type=int, default=4,
        help= (
            'early stop training if valid performance does not '
            + 'improve for N consecutive validation runs'
        )
    )
    parser.add_argument(
        '--disable_validation', action='store_true',
        help='isable validation'
    )
    parser.add_argument(
        '--disable_save', action='store_true',
        help='disable save'
    )

    # dataset parameter
    parser.add_argument('--valid_subsets', type=str, default='valid, query') 
    parser.add_argument("--layers", nargs="+", default=[512, 256], 
                        help="The number of units of each layer of the GNN. Default is [512, 128]")
    parser.add_argument("--embedding_dim", type=int, default=16,
                        help="Initial dimension of node embeddings.")
    parser.add_argument("--negative_sample", type=int, default=1)

    # augmentation parameter
    parser.add_argument("--aug_params", nargs="+", default=[0.2, 0.3, 0.1, 0.1], help="Hyperparameters for Augmentation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate. Default is 0.2")

    # training parameter
    parser.add_argument("--epochs", type=int, default=500, help="The number of epochs")
    parser.add_argument("--seed", type=int, default=777, help='Seed to fix')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=8)

    # model parameter
    ## graph
    parser.add_argument('--graph_model', type=str, default='gat')
    parser.add_argument('--self_loop', action='store_false')
    parser.add_argument('--reverse', action='store_false')
    parser.add_argument('--g_num_layers', type=int, default=3)
    parser.add_argument('--g_num_heads', type=int, default=8)
    parser.add_argument('--g_activation', type=str, default='gelu', choices=['elu', 'relu', 'gelu'])
    parser.add_argument('--g_residual', action='store_false')
    parser.add_argument('--g_feat_drop', type=float, default=0.2)
    parser.add_argument('--g_attn_drop', type=float, default=0.2)
    parser.add_argument('--g_negative_slope', type=float, default=0.2)
    ## model
    parser.add_argument('--model', type=str, default='pred_model',help='name of the model to be trained')
    parser.add_argument('--ma_decay', type=float, default=0.99)
    parser.add_argument('--model_dropout', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=4) 

    ## checkpoint (Just in case)
    parser.add_argument('--load_checkpoint', action='store_false') 
    

    return parser

def main():
    run = wandb.init(project='GraphMining_Project', entity='swryu')

    current_dir = os.getcwd()

    args = get_parser().parse_args()
    args.current_dir = current_dir
    args.input_path = os.path.join(current_dir, 'dataset')
    args.answer_path = os.path.join(current_dir, 'answer')
    args.save_dir = os.path.join(current_dir, 'gm_checkpoint')
    args.g_hidden_dim = args.embedding_dim
    args.proj_dim = args.g_hidden_dim // 2

    activation = {'elu': F.elu, 'relu': F.relu, 'gelu': F.gelu}
    args.g_activation = activation[args.g_activation]

    args.valid_subsets = (
        args.valid_subsets.replace(' ','').split(',')
        if (
            not args.disable_validation
            and args.valid_subsets
        )
        else []
    )
 
    wandb.config.update(args)
    wandb.run.name = 'GraphMining_Project'
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_num)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device_ids = list(range(len(args.device_num.split(','))))
    print('device_number : ', args.device_ids)
    set_struct(vars(args))

    mp.set_sharing_strategy('file_system')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True

    trainer = Trainer(args)
    trainer.train(wandb, run)
    
    wandb.finish()
    logger.info("done training")


def set_struct(cfg: dict):
    root = os.path.abspath(
        os.path.dirname(__file__)
    )
    from datetime import datetime
    now = datetime.now()
    from pytz import timezone
    # apply timezone manually
    now = now.astimezone(timezone('Asia/Seoul'))

    output_dir = os.path.join(
        root,
        "outputs",
        now.strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S")
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(output_dir)

    job_logging_cfg = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'train.log'
            }
        },
        'root': {
            'level': 'INFO', 'handlers': ['console', 'file']
            },
        'disable_existing_loggers': False
    }
    logging.config.dictConfig(job_logging_cfg)

    cfg_dir = ".config"
    os.mkdir(cfg_dir)
    os.makedirs(cfg['save_dir'], exist_ok=True)

    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        for k, v in cfg.items():
            print("{}: {}".format(k, v), file=f)


if __name__ == '__main__':
    main()