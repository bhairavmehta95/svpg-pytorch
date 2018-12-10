"""
Stein Variational Gradient Descent for Deep ConvNet on GPU.
Current implementation is mainly using for-loops over model instances.

Oct 29, 2017
Apr 30, 2018
"""  

import torch
import numpy as np
import torch.nn as nn


class SVGD(object):
    """Base class for Stein Variational Gradient Descent, with for-loops...
    The Bayesian neural network is defined in `bayes_nn.BayesNN` class.    

    References:
        Liu, Qiang, and Dilin Wang. "Stein variational gradient descent:
        A general purpose bayesian inference algorithm."
        Advances In Neural Information Processing Systems. 2016.

    Args:
        model (nn.Module): The model to be instantiated `n_samples` times
        data_loader (utils.data.DataLoader): For training and testing
        n_samples (int): Number of samples for uncertain parameters
    """

    def __init__(self, bayes_nn, train_loader):
        """
        For-loop implementation of SVGD.

        Args:
            bayes_nn (nn.Module): Bayesian NN
            train_loader (utils.data.DataLoader): Training data loader
            logger (dict)

        """
        pass


    def _squared_dist(self, X):
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


    def _Kxx_dxKxx(self, X):
        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in eqn (8)
        of reference SVGD paper.

        Args:
            X (Tensor): (S, P), design matrix of samples, where S is num of
                samples, P is the dim of each sample which stacks all params
                into a (1, P) row. Thus P could be 1 millions.
        """
        squared_dist = self._squared_dist(X)
        l_square = 0.5 * squared_dist.median() / math.log(self.n_samples)
        Kxx = torch.exp(-0.5 / l_square * squared_dist)
        # matrix form for the second term of optimal functional gradient
        # in eqn (8) of SVGD paper
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square

        return Kxx, dxKxx