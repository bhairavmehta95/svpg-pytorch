import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import math

from utils import *


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--num-agents', type=int, default=16,
                    help='number of agents to run in parallel')
parser.add_argument('--temp', type=float, default=10.0)

# Global Variables
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps.item()

print(device)
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(8, 128)
        self.affine2 = nn.Linear(128, 128)
        self.affine3 = nn.Linear(128, 4)

        self.reset()

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)

    def reset(self):
        self.saved_log_probs = []
        self.rewards = []

def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def _squared_dist(X):
    """Computes squared distance between each row of `X`, ||X_i - X_j||^2

    Args:
        X (Tensor): (S, P) where S is number of samples, P is the dim of 
            one sample

    Returns:
        (Tensor) (S, S)
    """
    XXT = torch.mm(X, X.t())
    XTX = XXT.diag()
    return -2.0 * XXT + XTX + XTX.unsqueeze(1)


def _Kxx_dxKxx(X):
    """
    Computes covariance matrix K(X,X) and its gradient w.r.t. X
    for RBF kernel with design matrix X, as in the second term in eqn (8)
    of reference SVGD paper.

    Args:
        X (Tensor): (S, P), design matrix of samples, where S is num of
            samples, P is the dim of each sample which stacks all params
            into a (1, P) row. Thus P could be 1 millions.
    """
    squared_dist = _squared_dist(X)
    l_square = 0.5 * squared_dist.median() / math.log(args.num_agents)
    Kxx = torch.exp(-0.5 / l_square * squared_dist)
    # matrix form for the second term of optimal functional gradient
    # in eqn (8) of SVGD paper
    dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square

    return Kxx, dxKxx


def finish_episode(policies, optimizers):
    policy_grads = []
    parameters = [] 
    for i in range(args.num_agents):
        policy_grad_agent = []
        R = 0
        rewards = []
        
        for r in policies[i].rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(0).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        for log_prob, reward in zip(policies[i].saved_log_probs, rewards):
            policy_grad_agent.append(log_prob * reward)

        optimizers[i].zero_grad()

        policy_grad = torch.cat(policy_grad_agent).sum()
        policy_grad.backward()

        vec_param, vec_policy_grad = parameters_to_vector(
                    policies[i].parameters(), both=True)

        policy_grads.append(vec_policy_grad.unsqueeze(0))
        parameters.append(vec_param.unsqueeze(0))

    # calculating the kernel matrix and its gradients
    parameters = torch.cat(parameters)
    Kxx, dxKxx = _Kxx_dxKxx(parameters)
    policy_grads = torch.cat(policy_grads)
    # this line needs S x P memory
    grad_logp = torch.mm(Kxx, policy_grads)
    # negate grads here!!!
    grad_theta = - (grad_logp + dxKxx) / args.num_agents
    # explicitly deleting variables does not release memory :(

    # update param gradients
    for i in range(args.num_agents):
        vector_to_parameters(grad_theta[i],
             policies[i].parameters(), grad=True)
        optimizers[i].step()

def main():
    policies = []
    optimizers = []
    envs = []

    torch.manual_seed(args.seed)

    for i in range(args.num_agents):
        policy = Policy().to(device)
        optimizer = optim.Adam(policy.parameters(), lr=1e-2)
        policies.append(policy)
        optimizers.append(optimizer)
        env = gym.make('LunarLander-v2')
        env.seed(args.seed + i)
        envs.append(env)

    max_reward = 0
    for i_episode in range(100000):
        if i_episode % 500 == 0:
            print("----Episode {}----".format(i_episode))
            for i in range(args.num_agents): 
                print('Agent: {}, Reward: {}'.format(i, np.sum(policies[i].rewards)))
            print("------------------")

        for i in range(args.num_agents):     
            policies[i].reset()

        for i in range(args.num_agents):
            state = envs[i].reset()
            for t in range(1000):  # Don't infinite loop while learning
                action = select_action(policies[i], state)
                state, reward, done, _ = envs[i].step(action)
                policies[i].rewards.append(reward)
                if done:
                    break

            if np.sum(policies[i].rewards) > max_reward:
                print('Agent: {}, Reward: {}, Episode: {}'.format(i, np.sum(policies[i].rewards), i_episode))
                max_reward = np.sum(policies[i].rewards)

        finish_episode(policies, optimizers)
        

if __name__ == '__main__':
    main()
