import pywt
import numpy as np
from scipy import linalg as scilinalg
import math
import matplotlib.pyplot as plt
import datetime
import os
import torch
from scipy.stats import entropy
from sklearn import linear_model
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.linear_model import OrthogonalMatchingPursuit


def generate_WaveletDict2D(N,d='db1'):
    """generates the 2D Wavelet Basis"""
    x = np.zeros((N, N))
    # generate the correct data structure for 2D wavelet decomposition of a 2D signal x
    coefs = pywt.wavedec2(x, d)
    # compute the number of wavelet levels
    n_levels = len(coefs)
    # initialize dictionary
    D_2d = []
    for i in range(n_levels):
        coefs[i] = list(coefs[i])
        # compute the number of filters for each level
        n_filters = len(coefs[i])
        for j in range(n_filters):
            for m in range(coefs[i][j].shape[0]):
                try:
                    # iterates through the detail coefficients and computes the dictionary elements
                    for n in range(coefs[i][j].shape[1]):
                        coefs[i][j][m][n] = 1
                        basis_vector = pywt.waverec2(coefs, d)
                        D_2d.append(basis_vector)
                        coefs[i][j][m][n] = 0
                except IndexError:
                    # iterates through the approximation coefficients and computes the dictionary elements
                    coefs[i][j][m] = 1
                    basis_vector = pywt.waverec2(coefs, d)
                    D_2d.append(basis_vector)
                    coefs[i][j][m] = 0

    D_2d = np.array(D_2d)
    return D_2d

def generate_WaveletDic1D(N,d = 'db1'):
    """generates the 1D Wavelet Basis"""
    x = np.zeros((N))
    # generate the correct data structure for 2D wavelet decomposition of a 2D signal x
    coefs = pywt.wavedec(x, d)
    # compute the number of wavelet levels
    n_levels = len(coefs)
    # initialize dictionary
    D_1d = []
    for i in range(n_levels):
        # compute the number of filter types for each level
        n_filters = len(coefs[i])
        for j in range(n_filters):
            try:
                for m in range(coefs[i][j].shape[0]):
                    coefs[i][j][m] = 1
                    basis_vector = pywt.waverec(coefs, d)
                    D_1d.append(basis_vector)
                    coefs[i][j][m] = 0
            except IndexError:
                coefs[i][j] = 1
                basis_vector = pywt.waverec(coefs, d)
                D_1d.append(basis_vector)
                coefs[i][j] = 0

    D_1d = np.array(D_1d)
    return D_1d

def compute_inv_cholesky(A): # A : (n_components, n_dim, n_dim)
    """computes the cholesky matrix of the inverse of A (component-wise in its first argument)"""
    [n_components,n_dim,_] = np.shape(A)
    inv_chol = np.empty((n_components, n_dim, n_dim))

    for k, A_matrix in enumerate(A):
        try:
            A_chol = scilinalg.cholesky(A_matrix, lower=True)
        # this part is there just to be sure, typically the cholesky decomposition should be numerically stable
        except scilinalg.LinAlgError:
            try:
                A_chol = scilinalg.cholesky(A_matrix + 0.01 * np.eye(A_matrix.shape[0],A_matrix.shape[0]), lower=True)
            except TypeError:
                print('problem')
        inv_chol[k] = scilinalg.solve_triangular(A_chol, np.eye(n_dim), lower=True).T

    return inv_chol

def compute_inv_cholesky_torch(A,device): # n_components, n_dim, n_dim
    """computes the cholesky matrix of the inverse of A (component-wise in its first argument)"""
    [n_components,n_dim,_] = A.size()
    inv_chol = torch.zeros((n_components, n_dim, n_dim)).to(device)
    for k, A_matrix in enumerate(A):
        try:
            A_chol = torch.linalg.cholesky(A_matrix)
        # this part is there just to be sure, typically the cholesky decomposition should be numerically stable
        except torch._C._LinAlgError:
            try:
                A_chol = torch.linalg.cholesky(A_matrix + torch.tensor(0.01).to(device) * torch.eye(A_matrix.shape[0], A_matrix.shape[0]).to(device))
                print(f'The inversion had to be slightly regularized!')
            except torch._C._LinAlgError:
                print(f'The inversion had to be strongly regularized!')
                L, V = torch.linalg.eig(A_matrix)
                L[torch.real(L) <= 0.1] = 0.1
                A_matrix = V @ torch.diag_embed(L) @ V.T
                A_chol = torch.linalg.cholesky(A_matrix + torch.tensor(0.001).to(device) * torch.eye(A_matrix.shape[0], A_matrix.shape[0]).to(device))

        inv_chol[k] = torch.linalg.solve_triangular(A_chol, torch.eye(n_dim).to(device),upper=False).T

    return inv_chol