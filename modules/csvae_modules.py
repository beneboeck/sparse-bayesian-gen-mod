from utils import utils as tu
import torch
import torch.nn as nn

###
# the core module of the CSVAE containing the encoder and decoder
###

class CSVAE_core(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, n_enc, n_dec, end_width, device):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_enc = n_enc
        self.n_dec = n_dec
        self.end_width = end_width

        # Encoder construction
        if (self.n_enc - 1) != 0:
            # Calculate the width difference from layer to layer
            if self.end_width > self.input_dim:
                steps_enc = (self.end_width - self.input_dim) // (self.n_enc - 1)
            else:
                steps_enc = - ((self.input_dim - self.end_width) // (self.n_enc - 1))
            encoder = []
            layer_dim = self.input_dim
            for n in range(n_enc - 1):
                encoder.append(nn.Linear(layer_dim, layer_dim + steps_enc))
                encoder.append(nn.ReLU())
                layer_dim = layer_dim + steps_enc
            encoder.append(nn.Linear(layer_dim, self.end_width))
            encoder.append(nn.ReLU())
        else:
            # Special case: Single encoder layer
            encoder = []
            layer_dim = self.input_dim
            encoder.append(nn.Linear(layer_dim, self.end_width))
            encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

        # Latent space mappings
        self.fc_mu = nn.Linear(self.end_width, self.latent_dim)
        self.fc_var = nn.Linear(self.end_width, self.latent_dim)

        # Decoder construction
        if (self.n_dec - 1) != 0:
            # Calculate the width difference from layer to layer
            steps_dec = (self.end_width - self.latent_dim) // (self.n_dec - 1)
            decoder = []
            layer_dim = self.latent_dim
            for n in range(n_enc - 1):
                decoder.append(nn.Linear(layer_dim, layer_dim + steps_dec))
                decoder.append(nn.ReLU())
                layer_dim = layer_dim + steps_dec
            decoder.append(nn.Linear(layer_dim, self.end_width))
            decoder.append(nn.ReLU())
        else:
            # Special case: Single decoder layer
            decoder = []
            layer_dim = self.latent_dim
            decoder.append(nn.Linear(layer_dim, self.end_width))
            decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)

        # Final layer
        n_out = self.output_dim
        self.final_layer = nn.Linear(self.end_width, n_out)

    def encode(self, x):
        """
        computes the latent mean and log variance using the encoder
        """
        out = self.encoder(x)
        mu, log_var = self.fc_mu(out), self.fc_var(out)
        return mu, log_var

    def reparameterize(self, log_var, mu):
        """
        samples from the variational encoder distribution
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mu + eps * std

    def decode(self, z):
        """
        computes the log variance of s|z in Eq. (2)
        """
        out = self.decoder(z)
        log_gamma = self.final_layer(out)
        # lower bound for gamma (exp(-12)) to not allow for too small gammas resulting in numerical problems
        log_gamma[log_gamma < -12] = -12
        return log_gamma

    def forward(self, x):
        """
        computes log variance of s|z, the latent mean and the latent log variance
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        log_gamma = self.decode(z)
        return log_gamma, mu, log_var

###
# standard CSVAE with fixed measurement matrix
###

class CSVAE(nn.Module):
    def __init__(self, odim, sdim, ldim, A, D, n_enc, n_dec, end_width, device, fix_zeta=0):
        super().__init__()
        self.sdim = sdim
        self.odim = odim
        self.A = A.float().to(device)
        self.D = torch.tensor(D).float().to(device)
        self.AD = self.A @ self.D
        self.device = device
        self.zeta = fix_zeta.float().to(device)
        self.CSVAE_core = CSVAE_core(odim, ldim, sdim, n_enc, n_dec, end_width, device).to(device)

    def fit(self, lr, loader_train, loader_val, epochs, device, n_t, n_val):
        """
        takes training specific parameters (learning rate, dataloader for training and evaluation, maximal number epochs, device, number training samples, number validation samples) as input
        outputs training and validation loss-specific values over training epochs
        """

        # initalize several training-controlling parameters
        rec_train, kl1_train, kl2_train, rec_val, kl1_val, kl2_val, risk_val = torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs)
        slope, adapt_lr = -1., False

        # initialize Adam optimizer
        optimizer = torch.optim.Adam(lr=lr, params=self.CSVAE_core.parameters())

        print('Start Training ')

        for i in range(epochs):
            for ind, samples in enumerate(loader_train):
                sample_in = samples.to(device)
                # compute the CSVAE parameters
                log_gamma, mu, log_var = self.CSVAE_core(sample_in)
                # compute the posterior means E[s|y,z] and Cov(y|z)
                posterior_means, CovY = self.compute_sparse_posterior(sample_in, log_gamma)
                # compute the standard VAE KL divergence
                kl1 = self.compute_kl1_divergence(log_var, mu)
                # compute the new CSVAE KL divergence (IMPORTANT: We apply the reformulations from Appendix I)
                kl2 = self.compute_kl2_divergence(log_gamma, posterior_means, CovY)
                # compute the new reconstruction loss (IMPORTANT: We apply the reformulations from Appendix I)
                rec = self.reconstruction_loss(sample_in, posterior_means, log_gamma, self.zeta)
                # save some memory
                del posterior_means, CovY, log_gamma, mu, log_var
                torch.cuda.empty_cache()
                # calculate risk to be minimized (negative ELBO in Eq. (15))
                risk = - (rec - kl2 - kl1)

                # track training loss
                kl1_train[i] += kl1.detach().to('cpu') * len(sample_in)
                kl2_train[i] += kl2.detach().to('cpu') * len(sample_in)
                rec_train[i] += rec.detach().to('cpu') * len(sample_in)

                # apply a gradient step
                optimizer.zero_grad()
                risk.backward()
                optimizer.step()

            # average training loss over all training samples (only for tracking)
            kl1_train[i] = kl1_train[i] / n_t
            kl2_train[i] = kl2_train[i] / n_t
            rec_train[i] = rec_train[i] / n_t

            print(f'epoch: {i}, kl1: {kl1_train[i].item():.4f}, kl2: {kl2_train[i].item():.4f}, rec: {rec_train[i].item():.4f}, total: {- (rec_train[i] - kl2_train[i] - kl1_train[i]).item():4f}')

            # evaluate validation loss and adapt the learning rate once during training (not particularly important to do this)
            with torch.no_grad():
                if i % 5 == 0:
                    i5 = int(i/5)
                    self.CSVAE_core.eval()
                    for ind, samples in enumerate(loader_val):
                        sample_in = samples.to(device)
                        log_gamma, mu, log_var = self.CSVAE_core(sample_in)
                        posterior_means, CovY = self.compute_sparse_posterior(sample_in, log_gamma)
                        kl1 = self.compute_kl1_divergence(log_var, mu)
                        kl2 = self.compute_kl2_divergence(log_gamma, posterior_means, CovY)
                        rec = self.reconstruction_loss(sample_in, posterior_means, log_gamma, self.zeta)
                        del posterior_means, CovY, log_gamma, mu, log_var
                        torch.cuda.empty_cache()
                        risk = - (rec - kl2 - kl1)

                        kl1_val[i5] += kl1.detach().to('cpu') * len(sample_in)
                        kl2_val[i5] += kl2.detach().to('cpu') * len(sample_in)
                        rec_val[i5] += rec.detach().to('cpu') * len(sample_in)
                        risk_val[i5] += risk.detach().to('cpu') * len(sample_in)

                    kl1_val[i5] = kl1_val[i5] / n_val
                    kl2_val[i5] = kl2_val[i5] / n_val
                    rec_val[i5] = rec_val[i5] / n_val
                    risk_val[i5] = risk_val[i5] / n_val
                    self.CSVAE_core.train()
                    print(f'Validation: kl1: {kl1_val[i5]:.4f}, kl2: {kl2_val[i5]:.4f}, rec: {rec_val[i5]:.4f}, total: {- (rec_val[i5] - kl2_val[i5] - kl1_val[i5]).item():4f}')

                    if ((i > 40) & (adapt_lr == False)) | ((i > 60) & (adapt_lr == True)):
                        steps = 5 if adapt_lr == False else 10
                        x_range_lr = torch.arange(steps)
                        x_lr = torch.ones(steps, 2)
                        x_lr[:, 0] = x_range_lr
                        beta_lr = torch.linalg.inv(x_lr.T @ x_lr) @ x_lr.T @ risk_val[i5-steps:i5].clone()[:, None]
                        slope_lr = beta_lr[0].detach().to('cpu').numpy()[0]
                        print(f'slope risk val: {slope_lr}')
                        if (slope_lr > 0) & (adapt_lr == False):
                            print('adapting learning rate')
                            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 5
                            adapt_lr = True
                        elif (slope_lr > 0) & (adapt_lr == True):
                            break
        kl1_train, kl2_train, rec_train, risk_val, kl1_val, kl2_val, rec_val = kl1_train[:i].numpy(), kl2_train[:i].numpy(), rec_train[:i].numpy(), risk_val[:i5].numpy(), kl1_val[:i5].numpy(), kl2_val[:i5].numpy(), rec_val[:i5].numpy()
        return - (rec_train - kl2_train - kl1_train), kl1_train, kl2_train, rec_train, risk_val, kl1_val, kl2_val, rec_val

    def compute_sparse_posterior(self, Y, log_gamma):
        """
        computes E[s|y,z] and Cov(y|z) using (50) and (52)
        """
        bs = log_gamma.shape[0]
        Covsy = log_gamma.exp()[:, :, None] * self.AD.T[None, :, :]
        Eys = torch.eye(self.odim)[None, :, :].repeat(bs, 1, 1).to(self.device)
        CovY = (self.AD[None, :, :] * log_gamma.exp()[:, None, :]) @ self.AD.T[None, :, :] + self.zeta * Eys
        L_PreY = tu.compute_inv_cholesky_torch(CovY, self.device)
        PreY = L_PreY @ torch.transpose(L_PreY, 1, 2)
        CovsyPy = Covsy @ PreY
        postMeans = torch.einsum('zij,zj->zi', CovsyPy, Y)
        del CovsyPy, PreY, L_PreY, Eys, Covsy
        torch.cuda.empty_cache()
        return postMeans, CovY

    def compute_kl1_divergence(self, log_var, mu):
        """
        standard VAE KL divergence (see Appendix E, Eq. (31))
        """
        return torch.mean(torch.sum(-0.5 * (1 + log_var - mu ** 2 - (log_var).exp()), dim=1))
    def compute_kl2_divergence(self, log_gamma, posterior_means, CovY):
        """
        computes the second KL divergence (see Appendix E, Eq. (30)) IMPORTANT: We apply Appendix I, i.e., the trace term with S cancels out
        """
        logdet_post_cov = - self.odim * torch.log(1 / self.zeta) - torch.logdet(CovY) + torch.sum(log_gamma, dim=1)
        kl = 0.5 * (torch.sum(log_gamma, dim=1) - logdet_post_cov + torch.sum(posterior_means ** 2 / log_gamma.exp(), dim=1))
        return torch.mean(kl)
    def reconstruction_loss(self, Y, posterior_mean, log_gamma,zeta=0):
        """
        computes the reconstruction loss (see Appendix D, Eq. (29)) IMPORTANT: We apply Appendix I, i.e., the trace term cancels out
        """
        s_error = torch.linalg.norm(Y - torch.einsum('ij,hj->hi', self.AD, posterior_mean), axis=1) ** 2
        return torch.mean(- 0.5 * (Y.shape[1] * torch.log(self.zeta) + (s_error) / zeta))

    def cme(self, y, mode='approx', n_samples=64):
        """
        reconstructs x from y using either the map estimator ("approx", (43)) or the CME approximation ("montecarlo", (42))
        """
        if mode == 'approx':
            z_hat, _ = self.CSVAE_core.encode(y)
            log_gamma = self.CSVAE_core.decode(z_hat)
            posterior_means, _ = self.compute_sparse_posterior(y, log_gamma)
            return posterior_means
        elif mode == 'montecarlo':
            mu, log_var = self.CSVAE_core.encode(y)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn(n_samples, self.CSVAE_core.latent_dim).to(self.device)
            z = torch.squeeze(std)[None, :] * eps + torch.squeeze(mu)[None, :]
            log_gamma = self.CSVAE_core.decode(z)
            posterior_means, _ = self.compute_sparse_posterior(y, log_gamma)
            return torch.mean(posterior_means, dim=0)


###
# CSVAE, which allows for varying measurement matrices for each training, validation and test sample
###

class CSVAE_vA(nn.Module):
    def __init__(self, odim, ddim, sdim, ldim, D, n_enc, n_dec, end_width, device, fix_zeta=0):
        super().__init__()
        self.odim = odim
        self.ldim = ldim
        self.sdim = sdim
        self.ddim = ddim
        self.n_enc = n_enc
        self.n_dec = n_dec
        self.end_width = end_width
        self.device = device
        self.D = torch.tensor(D).float().to(device)
        self.zeta = fix_zeta
        self.CSVAE_core = CSVAE_core(ddim, ldim, sdim, n_enc, n_dec, end_width, device).to(device)

    def fit(self, lr, loader_train, loader_val, epochs, device, n_t, n_val):
        """
        takes training specific parameters (learning rate, dataloader for training and evaluation, maximal number epochs, device, number training samples, number validation samples) as input
        outputs training and validation loss-specific values over training epochs
        """

        # initalize several training-controlling parameters
        rec_train, kl1_train, kl2_train, rec_val, kl1_val, kl2_val, risk_val = torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs)
        slope, adapt_lr = -1., False

        # initialize Adam optimizer
        optimizer = torch.optim.Adam(lr=lr, params=self.CSVAE_core.parameters())
        print('Start Training ')

        for i in range(epochs):
            for ind, samples in enumerate(loader_train):
                sample_in = samples[1].to(device)
                A_sample = samples[0].to(device)
                # compute sample-wise A times D
                AD_sample = torch.einsum('kij,kjl->kil',A_sample,self.D[None,:,:])
                # compute the inverse needed for the least squares estimate (see Section 3.2)
                LS_inv = torch.linalg.inv(torch.einsum('kij,kjl->kil',A_sample,torch.permute(A_sample,(0,2,1))))
                # compute the least squares matrix/filter
                filter_LS = torch.einsum('kij,kjl->kil',torch.permute(A_sample,(0,2,1)),LS_inv)
                # compute least squares estimate
                sample_in_LS = torch.einsum('kil,kl->ki',filter_LS,sample_in)
                # compute CSVAE parameters
                log_gamma, mu, log_var = self.CSVAE_core(sample_in)
                # compute the posterior means E[s|y,z] and Cov(y|z)
                posterior_means, CovY = self.compute_sparse_posterior(sample_in, log_gamma)
                # compute the standard VAE KL divergence
                kl1 = self.compute_kl1_divergence(log_var, mu)
                # compute the new CSVAE KL divergence (IMPORTANT: We apply the reformulations from Appendix I)
                kl2 = self.compute_kl2_divergence(log_gamma, posterior_means, CovY)
                # compute the new reconstruction loss (IMPORTANT: We apply the reformulations from Appendix I)
                rec = self.reconstruction_loss(sample_in, posterior_means, log_gamma, self.zeta)
                # save some memory
                del posterior_means, CovY, log_gamma, mu, log_var
                torch.cuda.empty_cache()
                # calculate risk to be minimized (negative ELBO in Eq. (15))
                risk = - (rec - kl2 - kl1)

                # track training loss
                kl1_train[i] += kl1.detach().to('cpu') * len(sample_in)
                kl2_train[i] += kl2.detach().to('cpu') * len(sample_in)
                rec_train[i] += rec.detach().to('cpu') * len(sample_in)

                # apply a gradient step
                optimizer.zero_grad()
                risk.backward()
                optimizer.step()

            # average training loss over all training samples (only for tracking)
            kl1_train[i] = kl1_train[i] / n_t
            kl2_train[i] = kl2_train[i] / n_t
            rec_train[i] = rec_train[i] / n_t

            print(f'epoch: {i}, kl1: {kl1_train[i].item():.4f}, kl2: {kl2_train[i].item():.4f}, rec: {rec_train[i].item():.4f}, total: {- (rec_train[i] - kl2_train[i] - kl1_train[i]).item():4f}')

            # evaluate validation loss and adapt the learning rate once during training (not particularly important to do this)
            with torch.no_grad():
                if i % 5 == 0:
                    i5 = int(i / 5)
                    self.CSVAE_core.eval()
                    for ind, samples in enumerate(loader_val):
                        sample_in = samples.to(device)
                        log_gamma, mu, log_var = self.CSVAE_core(sample_in)
                        posterior_means, CovY = self.compute_sparse_posterior(sample_in, log_gamma)
                        kl1 = self.compute_kl1_divergence(log_var, mu)
                        kl2 = self.compute_kl2_divergence(log_gamma, posterior_means, CovY)
                        rec = self.reconstruction_loss(sample_in, posterior_means, log_gamma, self.zeta)
                        del posterior_means, CovY, log_gamma, mu, log_var
                        torch.cuda.empty_cache()
                        risk = - (rec - kl2 - kl1)

                        kl1_val[i5] += kl1.detach().to('cpu') * len(sample_in)
                        kl2_val[i5] += kl2.detach().to('cpu') * len(sample_in)
                        rec_val[i5] += rec.detach().to('cpu') * len(sample_in)
                        risk_val[i5] += risk.detach().to('cpu') * len(sample_in)

                    kl1_val[i5] = kl1_val[i5] / n_val
                    kl2_val[i5] = kl2_val[i5] / n_val
                    rec_val[i5] = rec_val[i5] / n_val
                    risk_val[i5] = risk_val[i5] / n_val
                    self.CSVAE_core.train()
                    print(f'Validation: kl1: {kl1_val[i5]:.4f}, kl2: {kl2_val[i5]:.4f}, rec: {rec_val[i5]:.4f}, total: {- (rec_val[i5] - kl2_val[i5] - kl1_val[i5]).item():4f}')

                    if ((i > 40) & (adapt_lr == False)) | ((i > 60) & (adapt_lr == True)):
                        steps = 5 if adapt_lr == False else 10
                        x_range_lr = torch.arange(steps)
                        x_lr = torch.ones(steps, 2)
                        x_lr[:, 0] = x_range_lr
                        beta_lr = torch.linalg.inv(x_lr.T @ x_lr) @ x_lr.T @ risk_val[i5 - steps:i5].clone()[:, None]
                        slope_lr = beta_lr[0].detach().to('cpu').numpy()[0]
                        print(f'slope risk val: {slope_lr}')
                        if (slope_lr > 0) & (adapt_lr == False):
                            print('adapting learning rate')
                            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 5
                            adapt_lr = True
                        elif (slope_lr > 0) & (adapt_lr == True):
                            break
        kl1_train, kl2_train, rec_train, risk_val, kl1_val, kl2_val, rec_val = kl1_train[:i].numpy(), kl2_train[:i].numpy(), rec_train[:i].numpy(), risk_val[:i5].numpy(), kl1_val[:i5].numpy(), kl2_val[:i5].numpy(), rec_val[:i5].numpy()
        return - (rec_train - kl2_train - kl1_train), kl1_train, kl2_train, rec_train, risk_val, kl1_val, kl2_val, rec_val

    def compute_sparse_posterior(self, Y, log_gamma, log_file=0):
        """
        computes E[s|y,z] and Cov(y|z) using (50) and (52)
        """
        bs = log_gamma.shape[0]
        Covsy = log_gamma.exp()[:, :, None] * self.AD.T[None, :, :]
        Eys = torch.eye(self.odim)[None, :, :].repeat(bs, 1, 1).to(self.device)
        CovY = (self.AD[None, :, :] * log_gamma.exp()[:, None, :]) @ self.AD.T[None, :, :] + self.zeta * Eys
        L_PreY = tu.compute_inv_cholesky_torch(CovY, self.device)
        PreY = L_PreY @ torch.transpose(L_PreY, 1, 2)
        CovsyPy = Covsy @ PreY
        postMeans = torch.einsum('zij,zj->zi', CovsyPy, Y)
        del CovsyPy, PreY, L_PreY, Eys, Covsy
        torch.cuda.empty_cache()
        return postMeans, CovY

    def compute_kl1_divergence(self, log_var, mu):
        """
        standard VAE KL divergence (see Appendix E, Eq. (31))
        """
        return torch.mean(torch.sum(-0.5 * (1 + log_var - mu ** 2 - (log_var).exp()), dim=1))

    def compute_kl2_divergence(self, log_gamma, posterior_means, CovY):
        """
        computes the second KL divergence (see Appendix E, Eq. (30)) IMPORTANT: We apply Appendix I, i.e., the trace term with S cancels out
        """
        logdet_post_cov = - self.odim * torch.log(1 / self.zeta) - torch.logdet(CovY) + torch.sum(log_gamma, dim=1)
        kl = 0.5 * (torch.sum(log_gamma, dim=1) - logdet_post_cov + torch.sum(posterior_means ** 2 / log_gamma.exp(),
                                                                              dim=1))
        return torch.mean(kl)

    def reconstruction_loss(self, Y, posterior_mean, log_gamma, zeta=0):
        """
        computes the reconstruction loss (see Appendix D, Eq. (29)) IMPORTANT: We apply Appendix I, i.e., the trace term cancels out
        """
        s_error = torch.linalg.norm(Y - torch.einsum('ij,hj->hi', self.AD, posterior_mean), axis=1) ** 2
        return torch.mean(- 0.5 * (Y.shape[1] * torch.log(self.zeta) + (s_error) / zeta))

    def cme(self, y,A, mode='approx', n_samples=64):
        """
        reconstructs x from y using either the map estimator ("approx", (43)) or the CME approximation ("montecarlo", (42))
        """
        if mode == 'approx':
            LS_inv = torch.linalg.inv(torch.einsum('kij,kjl->kil', A[None, :, :], torch.permute(A[None, :, :], (0, 2, 1))))
            filter_LS = torch.einsum('kij,kjl->kil', torch.permute(A[None, :, :], (0, 2, 1)), LS_inv)
            y_LS = torch.einsum('kil,kl->ki', filter_LS, y)
            z_hat, _ = self.CSVAE_core.encode(y)
            log_gamma = self.CSVAE_core.decode(z_hat)
            posterior_means, _ = self.compute_sparse_posterior(y, log_gamma)
            return posterior_means
        elif mode == 'montecarlo':
            LS_inv = torch.linalg.inv(torch.einsum('kij,kjl->kil', A[None, :, :], torch.permute(A[None, :, :], (0, 2, 1))))
            filter_LS = torch.einsum('kij,kjl->kil', torch.permute(A[None, :, :], (0, 2, 1)), LS_inv)
            y_LS = torch.einsum('kil,kl->ki', filter_LS, y)
            mu, log_var = self.CSVAE_core.encode(y_LS)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn(n_samples, self.CSVAE_core.latent_dim).to(self.device)
            z = torch.squeeze(std)[None, :] * eps + torch.squeeze(mu)[None, :]
            log_gamma = self.CSVAE_core.decode(z)
            posterior_means, _ = self.compute_sparse_posterior(y, log_gamma)
            return torch.mean(posterior_means, dim=0)
