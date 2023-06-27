import wandb
import torch
import argparse
import os
import gym

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
                    monitor_gym=True,
                     config=args)
    
    def make_env(gym_id):
        def thunk():
            env = gym.make(gym_id)
            env.seed(args.seed)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.RecordVideo(env, "videos")
            return env

        return thunk
    envs = gym.vector.SyncVectorEnv([make_env('CartPole-v1') for _ in range(4)])

    observations = envs.reset()
    for i in range(3000):
        action = envs.action_space.sample()
        observation, reward, done, info = envs.step(action)
        episode_reward = 0
        episode_reward += reward
        for item in info:
            if "episode" in item:
                episode_reward += item["episode"]["r"]
        wandb.log({'reward': episode_reward})
        if done.any():
            break
    