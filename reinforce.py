import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from svpg import SVGD
from utils import *


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--num-agents', type=int, default=4,
                    help='number of agents to run in parallel')

args = parser.parse_args()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.reset()

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def reset(self):
        self.saved_log_probs = []
        self.rewards = []


policies = []
optimizers = []
envs = []
svpg = SVPG()
eps = np.finfo(np.float32).eps.item()
torch.manual_seed(args.seed)

for i in range(args.num_agents):
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    policies.append(policy)
    optimizers.append(optimizer)
    env = gym.make('CartPole-v0')
    env.seed(args.seed + i)
    envs.append(env)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    policy_grads = []
    parameters = [] 
    for i in range(args.num_agents):
        R = 0
        rewards = []
        
        for r in policies[i].rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward in zip(policies[i].saved_log_probs, rewards):
            policy_grad.append(-log_prob * reward)

        optimizer.zero_grad()
        policy_grad = torch.cat(policy_grad).sum()
        policy_grad.backward()

        vec_param, vec_policy_grad = parameters_to_vector(
                    policies[i].parameters(), both=True)

        policy_grads.append(vec_policy_grad.unsqueeze(0))
        parameters.append(vec_param.unsqueeze(0))

    theta = torch.cat(parameters)
    Kxx, dxKxx = SVPG._Kxx_dxKxx(theta)
    grad_log_joint = torch.cat(policy_grads)
    # this line needs S x P memory
    grad_logp = torch.mm(Kxx, grad_log_joint)
    # negate grads here!!!
    grad_theta = - (grad_logp + dxKxx) / args.num_agents
    # explicitly deleting variables does not release memory :(

    # update param gradients
    for i in range(args.num_agents):
        vector_to_parameters(grad_theta[i],
            self.bayes_nn[i].parameters(), grad=True)
        optimizers[i].step()


def main():
    for i_episode in count(100):
        for i in range(args.num_agents):
            state = envs[i].reset()
            for t in range(10000):  # Don't infinite loop while learning
                action = select_action(policies[i], state)
                state, reward, done, _ = envs[i].step(action)
                policies[i].rewards.append(reward)
                if done:
                    break

        finish_episode()
        for i in range(args.num_agents): 
            policies[i].reset()

if __name__ == '__main__':
    main()
