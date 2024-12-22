import torch
from utils import utils as tu
from sklearn import linear_model
import numpy as np


def apply_sbl_torch(y,AD,max_iter,device,zeta_in):
    """
    applies the classical SBL algorithm from "Sparse Bayesian Learning for Basis Selection" (Wipf et al.)
    with the reformulations from Appendix I in our work to speed up the reconstructions
    """
    AD = AD.float().to(device)
    y = y.float().to(device)
    [odim,sdim] = AD.shape

    cov_diag = torch.rand(sdim).to(device)
    zeta = zeta_in
    posterior_mean_old = 3000 * torch.ones(AD.shape[1]).to(device)
    for iter in range(max_iter):

        # e-step
        Covsy = cov_diag[:, None] * AD.T
        Eys = torch.eye(odim).to(device)
        CovY = (AD * cov_diag[None, :]) @ AD.T + zeta * Eys
        L_PreY = torch.squeeze(tu.compute_inv_cholesky_torch(CovY[None, :, :], device))
        PreY = L_PreY @ torch.transpose(L_PreY,dim0=0,dim1=1)
        CovsyPy = Covsy @ PreY
        diagCs_yk = cov_diag - torch.sum(CovsyPy * Covsy, dim=1)
        postMeans = CovsyPy @ y

        # check stopping criterion
        error = torch.linalg.norm(postMeans - posterior_mean_old)**2
        if (error < 1 * 1e-4) & (iter > 5):
            break

        # m-step
        posterior_mean_old = postMeans
        cov_diag = torch.real(torch.abs(postMeans)**2 + diagCs_yk)

    # compute the final parameters
    Covsy = cov_diag[:, None] * AD.T
    Eys = torch.eye(odim).to(device)
    CovY = (AD * cov_diag[None, :]) @ AD.T + zeta * Eys
    L_PreY = torch.squeeze(tu.compute_inv_cholesky_torch(CovY[None, :, :], device))
    PreY = L_PreY @ torch.transpose(L_PreY, dim0=0, dim1=1)
    CovsyPy = Covsy @ PreY
    diagCs_yk = cov_diag - torch.sum(CovsyPy * Covsy, dim=1)
    postMeans = CovsyPy @ y
    return postMeans.to('cpu').numpy()

def apply_lasso_sklearn(y,A,D,lam):
    """
    applies Lasso regression (i.e., it solves ||y - ADs||^2_2 + lam * ||s||_1) using the sklearn package
    """
    # scale the shrinkage parameter to neutralize the normalization of the sklearn implemented MSE in Lasso
    alpha = lam * 1 / (2 * y.shape[0])
    clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
    clf.fit(A @ D, y)
    sLASSO = clf.coef_
    xLASSO = D @ sLASSO
    xLASSO[xLASSO < 0] = 0
    xLASSO[xLASSO > 1] = 1
    return xLASSO