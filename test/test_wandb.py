import wandb
import torch
import argparse
import os

# 设置命令行参数
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--track', type=bool, default=True, help='the track of wandb')
    parse.add_argument('--seed', type=int, default=0, help='the random seed')
    parse.add_argument('--wandb-project', type=str,
                        default=os.path.basename(__file__).split('.')[0],
                        help='the name of wandb name')
    parse.add_argument('--wandb-entity', type=str,
                        default=None,help='the name of wandb name')
    return parse.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # 设置随机种子
    torch.manual_seed(args.seed)
    # 设置wandb参数
    if args.track:
        wandb.init(project=args.wandb_project,
                     entity=args.wandb_entity,
                     config=args)

    for ii in range(10):
        print(ii)
        if args.track:
            wandb.log({'test':ii})