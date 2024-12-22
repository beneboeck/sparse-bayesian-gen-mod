import numpy as np
from scipy.special import logsumexp
from utils import utils


###
# the core module of the CSGMM
###
class CSGMM_core():
    def __init__(self,n_components,sdim,init='random'):
        self.n_components = n_components
        self.sdim = sdim
        self.init = init
        self.gamma = np.empty((self.n_components, self.sdim))
        self.weights = np.empty((self.n_components))

    def init_paras(self):
        """initializes the parameters of the core modules randomly, but normed"""
        if self.init == 'random':
            self.weights = np.ones(self.n_components) / self.n_components
            self.gamma = np.random.rand(self.n_components, self.sdim)
            self.gamma = self.gamma / np.sum(self.gamma, axis=1)[:, None] * self.sdim

###
# standard CSGMM with fixed measurement matrix
###

class CSGMM():
    def __init__(self, n_components, odim, sdim, hdim, A, D,init_method='random', fix_zeta=0):
        self.n_components = n_components
        self.odim = odim
        self.sdim = sdim
        self.hdim = hdim
        self.A = A
        self.D = D
        self.AD = self.A @ self.D
        self.init_method = init_method
        self.zeta = fix_zeta

        self.CSGMM_core = CSGMM_core(self.n_components, self.sdim, init=init_method)
        # additional CSGMM parameters to the core module parameters, which are updated during training
        self.CovY = np.empty((self.n_components, self.odim, self.odim))
        self.PreY= np.empty((self.n_components, self.odim, self.odim))
        self.L_PreY = np.empty((self.n_components, self.odim, self.odim))

        self.cmeFilters = np.empty((self.n_components,self.sdim,self.odim))

    def init_paras(self):
        """initializes all parameters"""
        self.CSGMM_core.init_paras()
        Eys = np.eye(self.odim,self.odim)
        self.CovY = (self.AD[None,:,:] * self.CSGMM_core.gamma[:,None,:]) @ self.AD.T[None,:,:] + self.zeta * Eys[None,:,:]
        self.L_PreY = utils.compute_inv_cholesky(self.CovY)
        self.PreY = self.L_PreY @ np.transpose(self.L_PreY, (0, 2, 1))

    def fit(self, Y, max_iter):
        """
        runs the EM algorithm (see Section 3.3)
        """
        print('Start Training')
        # initialization
        self.init_paras()
        ref_logl = -10 ** 5
        # initalize the log likelihood tracking for the stopping criterion
        logl_track = []
        for iter in range(max_iter):
            # e-step
            log_respos, log_likeli,posterior_means,diag_post_covs = self.e_step(Y)
            # compute the mean log likelihood
            logl = np.mean(log_likeli)
            logl_track.append(logl)
            # compute the gap from the last two steps
            gap = np.real(logl - ref_logl)
            print(f'iteration: {iter}, gap: {gap:.4f}')
            if gap < 1e-3:
                break
            ref_logl = logl
            # m-step
            self.m_step(log_respos,posterior_means,diag_post_covs)

        # prepare the filters for the CME (see Eq. (44))
        self.prepare_cme()
        return logl_track

    def e_step(self, Y):
        """
        computes the log-responsibilites, the posterior means and diagonal of the posterior covariances (see Eq. (8) & Appendix I)
        """
        log_respos, log_likeli = self.compute_log_respos(Y)
        posterior_means,diag_posterior_covs = self.compute_sparse_posterior(Y)
        return log_respos, log_likeli,posterior_means,diag_posterior_covs

    def compute_log_gaussian_prob(self, Y):
        """computes log-gaussian probablities of each compressed training sample for each component"""
        log_det = np.real(2 * np.sum(np.log(np.diagonal(self.L_PreY, axis1=1, axis2=2)), axis=1))  # n_components
        Ly_shift = np.einsum('kji,lj->lki', self.L_PreY, Y,optimize='greedy')  # n_samples,n_components,n_dim
        log_prob = -np.sum(np.abs(Ly_shift) ** 2, axis=2)  # n_samples, n_components
        return 0.5 * (log_det[None, :] + log_prob)  # n_samples, n_components
    def compute_log_weights(self):
        """computes the log prior weights"""
        return np.log(self.CSGMM_core.weights)  # n_components

    def compute_log_respos(self, Y):
        """computes the log-responsibilites and log-likelihoods for each sample"""
        log_gauss = self.compute_log_gaussian_prob(Y)  # n_samples, n_components
        log_weights = self.compute_log_weights()  # n_components
        log_likeli = logsumexp(log_gauss + log_weights[None, :], axis=1)  # n_samples
        log_respos = (log_gauss + log_weights[None, :] - log_likeli[:, None])
        log_gauss = None
        del log_gauss
        return log_respos, log_likeli  # n_samples, n_compnents || n_samples

    def compute_sparse_posterior(self, Y):
        """computes the posterior means and diagonal of the posterior covariances (see Eq. (8) & Appendix I)"""
        Covsy = self.CSGMM_core.gamma[:,:,None] * self.AD.T[None,:,:]
        CovsyPy = Covsy @ self.PreY
        # efficient computation of Eq. (53)
        diagCs_yk = self.CSGMM_core.gamma - np.sum(CovsyPy * Covsy, axis=2)
        postMeans = np.einsum('kil,nl->nki', CovsyPy, Y,optimize='optimal')
        return postMeans,diagCs_yk

    def m_step(self, log_respos,posterior_means,diag_post_covs):
        """updates the parameters of the core module (i.e., the voriances and weights) as well as the parameters of the CSGMM module"""
        respos = np.exp(log_respos)
        log_respos = None
        del log_respos
        nk = np.real(respos).sum(axis=0) + 10 * np.finfo(respos.dtype).eps
        # apply Eq. (18)
        self.CSGMM_core.gamma = np.sum(respos[:, :, None] * np.abs(posterior_means) ** 2, axis=0) / nk[:,None] + np.real(diag_post_covs)  # [K,dimA]
        posterior_means = None
        diag_post_covs = None
        del diag_post_covs
        del posterior_means
        # for being numerically stable apply some lower bound on the variances
        self.CSGMM_core.gamma[self.CSGMM_core.gamma < 1e-7] = 1e-7
        # apply Eq. (18)
        self.CSGMM_core.weights = nk / respos.shape[0]
        Eys = np.eye(self.odim)
        # apply Eq. (17)
        self.CovY = (self.AD[None,:,:] * self.CSGMM_core.gamma[:,None,:]) @ self.AD.T + self.zeta * Eys
        self.L_PreY = utils.compute_inv_cholesky(self.CovY)
        # compute inverse of (17)
        self.PreY = self.L_PreY @ np.transpose(self.L_PreY,(0,2,1))

    def prepare_cme(self):
        """precomputes the filters needed for the CME and MAP-based estimation with the CSGMM (see Eq. (50))"""
        Eys = np.eye(self.odim)
        self.CovY = (self.AD[None, :, :] * self.CSGMM_core.gamma[:, None, :]) @ self.AD.T + self.zeta * Eys
        self.L_PreY = utils.compute_inv_cholesky(self.CovY)
        self.PreY = self.L_PreY @ np.transpose(self.L_PreY, (0, 2, 1))
        ACovxT = self.CSGMM_core.gamma[:,:,None] * self.AD.T
        self.cmeFilters = ACovxT @ self.PreY

    def cme(self, y,mode='cme'):
        """computes either the CME (Eq. (44)) or the MAP (Eq. (45))-based estimator"""
        Y = y[None, :]
        if mode == 'cme':
            respo = np.squeeze(np.exp(self.compute_log_respos(Y)[0]))
            estimators = np.empty((self.n_components,self.sdim))
            for k in range(self.n_components):
                estimators[k] = self.cmeFilters[k] @ y
            if self.n_components > 1:
                return np.sum(respo[:, None] * estimators, axis=0)
            else:
                return estimators
        elif mode == 'map':
            respo = np.squeeze(np.exp(self.compute_log_respos(Y)[0]))
            k_max = np.argmax(respo)
            estimators = self.cmeFilters[k_max] @ y
            return estimators