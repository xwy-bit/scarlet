from typing import Any
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import torch.nn.functional as F
import gym
import wandb



# config
class C:
    # basic config
    state_dim = 2
    action_dim = 4
    max_action = 1
    max_train_steps = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ppo config
    batch_size = 64
    mini_batch_size = 10
    entropy_coef = 0.01
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 10
    eps_clip = 0.2
    n_workers = 8
    optimizer = nn.Adam
    activate_fun = 'relu'
    use_grad_clip = True
    use_adv_normalization = True
    
    # trick switch
    use_state_norm = True
    use_reward_norm = True
    use_reward_scaling = True


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.activate_fun = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }[C.activate_fun]

        self.max_action = max_action

    def forward(self, x):
        x = self.activate_fun(self.l1(x))
        x = self.activate_fun(self.l2(x))
        x = self.activate_fun(self.l3(x))
        x = nn.Softmax(dim=-1)(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)
    
class replay_buffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []
        self.counter = 0
        
    def store(self, state, action,action_prob, reward, next_state, done):
        experience = (state, action,action_prob, reward, next_state, done)
        self.s[self.counter] = state
        self.a[self.counter] = action
        self.ap[self.counter] = action_prob
        self.r[self.counter] = reward
        self.ns[self.counter] = next_state
        self.d[self.counter] = done
        self.counter += 1
        
    def sample(self, batch_size):
    
    @property
    def counter():
        return len(self.buffer)
        
    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        action_prob_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = np.random.choice(self.buffer, batch_size, replace=False)

        for experience in batch:
            state, action , action_prob , reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            action_prob_batch.append(action_prob)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done) 
        return  state_batch , action_batch , action_prob_batch , reward_batch,next_state_batch , done_batch

class Agent:
    def __init__(self) -> None:
        self.actor = Actor(C.state_dim, C.action_dim, C.max_action).to(C.device)
        self.critic = Critic(C.state_dim).to(C.device)
        self.actor_optimizer = C.optimizer(self.actor.parameters(), lr=C.lr, betas=C.betas)
        self.critic_optimizer = C.optimizer(self.critic.parameters(), lr=C.lr, betas=C.betas)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(C.device).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()
    def evaluate(self,s):
        s = torch.unsqueeze(torch.FloatTensor(s),0)
        a_prob = self.actor(s).detach().numpy().flatten()
        return a_prob
    def update(self, replay_buffer, step):
        s ,a ,a_prob , r , s_ , done = replay_buffer.sample(C.batch_size) # sampled from replay buffer
        gae = 0
        adv = []
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + C.gamma * vs_ * (1 - done) - vs # sampled multi states
            for delta , d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + C.gamma * C.gae_lambda * (1 - d) * gae
                adv.insert(0, gae) #  needed for advantages normalization
            adv = torch.tensor(adv,dtype=torch.float32).to(C.device)
            v_target = adv + vs
            # advantage normalization
            if C.use_adv_normalization:
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            
            # optimize policy for K epochs
            for _ in range(C.K_epochs):
                # use sampled batch data for optimization
                idx_sampler = BatchSampler(SubsetRandomSampler(range(C.batch_size)), C.mini_batch, drop_last=False)
                for idx in idx_sampler:
                    dist = Categorical(logits=self.actor(s[idx]))
                    dist_entropy = dist.entropy().view(-1,1) # shape: (mini_batch_size, 1)
                    action_log_prob = dist.log_prob(a[idx]).view(-1,1) # shape: (mini_batch_size, 1)

                    # importance sampling weight
                    ratio = torch.exp(action_log_prob - a_prob)
                    surr1 = ratio * adv[idx]
                    surr2 = torch.clamp(ratio, 1-C.eps_clip, 1+C.eps_clip) * adv[idx]
                    actor_loss = -torch.min(surr1, surr2) - C.entropy_coef * dist_entropy
                    # update actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.mean().backward()
                    if C.use_grad_clip:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                    
                    v_s = self.critic(s[idx])
                    critic_loss = F.mse_loss(v_target[idx], v_s)
                    # update critic
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    if C.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                    self.critic_optimizer.step()
            
            if C.use_lr_decay:
                self.lr_decay(step)
    
    def lr_decay(self,step):
        lr_a = self.lr_a * (1 - step / self.step_max)
        lr_c = self.lr_c * (1 - step / self.step_max)
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr_a
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr_c

# normalization
class RunningMeanStd:
    # calculate mean & std dynamicly
    def __init__(self,shape) -> None:
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
        
    def update(self,x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            last_mean = self.mean.copy()
            self.mean =last_mean + (x - self.mean) / self.n
            self.S = self.S + (x - last_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)
            

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x
class Rewardscaling:
    def __init__(self,gamma ,shape) -> None:
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=shape)
        self.R  = np.zeros(shape)
    
    def __call__(self, x) -> Any:
        self.R = self.gamma * self.R + x
        x = x / (self.running_ms.std + 1e-8)
        return x
    def reset(self): # reset when an eposide is done,
        self.R = np.zeros(self.shape)
        

def main():
    # wandb 

    # set environment and agent
    env = gym.make('CartPole-v1')
    env_evaluate = gym.make('CartPole-v1')
    agent = Agent()
    replay_buffer = ReplayBuffer()

    # set seed
    env.seed(C.seed)
    env.action_space.seed(C.seed)
    env_evaluate.seed(C.seed)
    env_evaluate.action_space.seed(C.seed)
    np.random.seed(C.seed)
    torch.manual_seed(C.seed)
    torch.cuda.manual_seed_all(C.seed)
    
    # set step
    total_steps = 0
    
    # evaluation initialization
    evaluate_num = 0
    evaluate_reward = []
    
    # normalization initialization
    if C.use_state_norm:
        norm_state = Normalization(shape = C.state_dim)
    if C.use_reward_norm:
        norm_reward = Normalization(shape = 1)
    if C.use_reward_scaling:
        reward_scaling = Rewardscaling(gamma = C.gamma, shape = 1)
    
    
    while total_steps < C.max_train_steps:
        s = env.reset()
        if C.state_norm:
            s = norm_state(s)
        if C.use_reward_norm:
            r = reward_scaling.reset()
        eposide_step = 0
        done = False
        while not done:
            eposide_step += 1
            a , logprob = agent.choose(s)
            
            s_, r , done , info = env.step(a)
            if C.use_state_norm:
                s_ = norm_state(s_)
            if C.use_reward_norm:
                r = norm_reward(r)
            if C.use_reward_scaling:
                r = reward_scaling(r)
            
            if done and eposide_step != C.max_episode_steps:
                dw = True
            else:
                dw = False
            replay_buffer.store(s,a,r,logprob,s_,dw)
            
            s = s_
            total_steps += 1
            
            if replay_buffer.size == C.batch_size:
                agent.update(replay_buffer)
                
            
            if total_steps % C.evalution_frequence == 0:
                # evaluate
                agent.eval()
                evaluate_num += 1
                evaluate_reward = evaluate_policy
                
                
            
            
            
        
        
        
    
