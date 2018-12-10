import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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
parser.add_argument('--temp', type=float, default=10.0)

# Global Variables
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps.item()

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

def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


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
            policy_grad_agent.append(-log_prob * reward)

        optimizers[i].zero_grad()

        policy_grad = torch.cat(policy_grad_agent).sum()
        policy_grad.backward()

        vec_param, vec_policy_grad = parameters_to_vector(
                    policies[i].parameters(), both=True)

        policy_grads.append(vec_policy_grad.unsqueeze(0))
        parameters.append(vec_param.unsqueeze(0))

    params = torch.cat(parameters).cpu().numpy()

    # No negation, negated in L73
    gradient = torch.cat(policy_grads).cpu().numpy()

    ## Get distance matrix
    distance_matrix = np.sum(np.square(params[None, :, :] - params[:, None, :]), axis=-1)
    # get median
    distance_vector = distance_matrix.flatten()
    distance_vector.sort()
    median = 0.5 * (
    distance_vector[int(len(distance_vector) / 2)] + distance_vector[int(len(distance_vector) / 2) - 1])
    h = median / (2 * np.log(args.num_agents + 1))

    # if args.adaptive_kernel:
    #     L_min = None
    #     alpha_best = None
    #     for alpha in args.search_space:
    #         kernel_alpha = np.exp(distance_matrix * (-alpha / h))
    #         mean_kernel = np.sum(kernel_alpha, axis = 1)
    #         L = np.mean(np.square(mean_kernel - 2.0 * np.ones_like(mean_kernel)))
    #         logger.log("Current Loss {:} and Alpha : {:}".format(L, alpha))
    #         if L_min is None:
    #             L_min = L
    #             alpha_best = alpha
    #         elif L_min > L:
    #             L_min = L
    #             alpha_best = alpha
    #     logger.record_tabular('Best Alpha', alpha_best)
    #     h =  h / alpha_best
    
    kernel = np.exp(distance_matrix[:, :] * (-1.0 / h))
    kernel_gradient = kernel[:, :, None] * (2.0 / h) * (params[None, :, :] - params[:, None, :])
    
    weights = (1.0 / args.temp) * kernel[:, :, None] * gradient[:, None, :] + kernel_gradient[:, :, :]

    weights = -np.mean(weights[:, :, :], axis=0)

    weights = torch.tensor(weights).float().to(device)
    # update param gradients
    for i in range(args.num_agents):
        vector_to_parameters(weights[i],
            policies[i].parameters(), grad=True)
        
        # I don't think we need to step?
        # optimizers[i].step()


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
        env = gym.make('CartPole-v0')
        env.seed(args.seed + i)
        envs.append(env)

    for i_episode in count(100):
        for i in range(args.num_agents):
            state = envs[i].reset()
            for t in range(10000):  # Don't infinite loop while learning
                action = select_action(policies[i], state)
                state, reward, done, _ = envs[i].step(action)
                policies[i].rewards.append(reward)
                if done:
                    break

            print('Agent: {}, Reward: {}'.format(i, np.sum(policies[i].rewards)))

        finish_episode(policies, optimizers)
        for i in range(args.num_agents): 
            policies[i].reset()

if __name__ == '__main__':
    main()
