import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F


class VAEAnomaly(nn.Module):
    def __init__(self, input_size=8, latent_size=2, L=10):
        super(VAEAnomaly, self).__init__()
        self.L = L
        self.prior = Normal(0, 1)

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, latent_size * 2)
            # times 2 because this is the concatenated vector of latent mean and variance
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_size * 2)
            # times 2 because this is the concatenated vector of reconstructed mean and variance
        )

    def forward(self, x, w=100):
        latent_mu, latent_logvar = self.encoder(x).chunk(2, dim=-1)
        latent_std = torch.exp(0.5 * latent_logvar)

        dist = Normal(latent_mu, latent_std)
        z = dist.rsample([self.L]).transpose(0, 1)

        recon_mu, recon_logvar = self.decoder(z).chunk(2, dim=-1)
        recon_std = torch.exp(0.5 * recon_logvar)

        log_p = Normal(recon_mu, recon_std).log_prob(x.unsqueeze(1)).mean(dim=-1).mean(dim=-1)
        kl = kl_divergence(dist, self.prior).mean(dim=-1)
        loss = (kl - log_p).sum()
        return loss

    def reconstructed_probability(self, x):
        latent_mu, latent_logvar = self.encoder(x).chunk(2, dim=-1)
        latent_std = torch.exp(0.5 * latent_logvar)

        dist = Normal(latent_mu, latent_std)
        z = dist.rsample([self.L]).transpose(0, 1)

        recon_mu, recon_logvar = self.decoder(z).chunk(2, dim=-1)
        recon_std = torch.exp(0.5 * recon_logvar)

        p = F.sigmoid(Normal(recon_mu, recon_std).log_prob(x.unsqueeze(1)).mean(dim=1).mean(dim=-1))
        return p

    def generate(self, batch_size):
        z = self.prior.sample((batch_size, self.latent_size))
        recon_mu, recon_logvar = self.decoder(z).chunk(2, dim=-1)
        recon_std = torch.exp(0.5 * recon_logvar)
        return recon_mu + recon_std * torch.rand_like(recon_std)
