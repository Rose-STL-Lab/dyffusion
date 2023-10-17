import torch
from torch import Tensor
from torch.distributions import Gamma

from src.diffusion._base_diffusion import BaseDiffusion
from src.models.mcvd import get_sigmas


def _pow_l1(x):
    return x.abs()


def _pow_l2(x):
    return 1 / 2.0 * x.square()


class UNetMore_DDPM(BaseDiffusion):
    def __init__(
        self,
        version: str = "DDPM",
        beta_schedule: str = "linear",
        sigma_begin: float = 0.02,
        sigma_end: float = 0.0001,
        gamma: bool = False,
        noise_in_cond: bool = False,
        step_lr: float = 0.0,
        n_steps_each: int = 0,
        sampling_denoise: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)  # timesteps: int = 1000,
        self.save_hyperparameters(ignore=["model"])
        self.version = version.upper()
        assert (
            self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM"
        ), f"models/unet : version is not DDPM or DDIM! Given: {self.version}"

        self.schedule = beta_schedule  # getattr(config.model, 'sigma_dist', 'linear')
        timesteps = self.hparams.timesteps
        if self.schedule == "linear":
            self.register_buffer("betas", get_sigmas(beta_schedule, sigma_begin, sigma_end, timesteps))
            self.register_buffer("alphas", torch.cumprod(1 - self.betas.flip(0), 0).flip(0))
            self.register_buffer("alphas_prev", torch.cat([self.alphas[1:], torch.tensor([1.0]).to(self.alphas)]))
        elif self.schedule == "cosine":
            self.register_buffer("alphas", get_sigmas(beta_schedule, sigma_begin, sigma_end, timesteps))
            self.register_buffer("alphas_prev", torch.cat([self.alphas[1:], torch.tensor([1.0]).to(self.alphas)]))
            self.register_buffer("betas", 1 - self.alphas / self.alphas_prev)
        self.gamma = gamma
        self.num_timesteps = len(self.betas)
        if self.gamma:
            self.theta_0 = 0.001
            self.register_buffer(
                "k", self.betas / (self.alphas * (self.theta_0**2))
            )  # large to small, doesn't match paper, match code instead
            self.register_buffer(
                "k_cum", torch.cumsum(self.k.flip(0), 0).flip(0)
            )  # flip for small-to-large, then flip back
            self.register_buffer("theta_t", torch.sqrt(self.alphas) * self.theta_0)

        self.noise_in_cond = noise_in_cond
        if self.hparams.loss_function in ["l2", "mse", "mean_squared_error"]:
            self.pow_ = _pow_l2
        elif self.hparams.loss_function in ["l1", "mae", "mean_absolute_error"]:
            self.pow_ = _pow_l1

    def p_losses(self, x, t: Tensor = None, condition: Tensor = None, cond_mask=None):
        # z, perturbed_x
        b = x.shape[0]
        if self.version == "SMLD":
            sigmas = self.sigmas
            if t is None:
                t = torch.randint(0, len(sigmas), (b,), device=x.device)
            used_sigmas = sigmas[t].reshape(b, *([1] * len(x.shape[1:])))
            z = torch.randn_like(x)
            perturbed_x = x + used_sigmas * z
        elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
            used_alphas = self.alphas[t].reshape(b, *([1] * len(x.shape[1:])))
            if self.gamma:
                used_k = self.k_cum[t].reshape(b, *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
                used_theta = self.theta_t[t].reshape(b, *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
                z = Gamma(used_k, 1 / used_theta).sample()
                z = (z - used_k * used_theta) / (1 - used_alphas).sqrt()
            else:
                z = torch.randn_like(x)
            perturbed_x = used_alphas.sqrt() * x + (1 - used_alphas).sqrt() * z
        else:
            raise NotImplementedError(f"models/unet : version is not DDPM or DDIM! Given: {self.version}")

        if self.noise_in_cond and condition is not None:  # We add noise to cond
            alphas = self.alphas
            # if labels is None:
            #     labels = torch.randint(0, len(alphas), (cond.shape[0],), device=cond.device)
            labels = t
            used_alphas = alphas[labels].reshape(condition.shape[0], *([1] * len(condition.shape[1:])))
            if self.gamma:
                used_k = (
                    self.k_cum[labels]
                    .reshape(condition.shape[0], *([1] * len(condition.shape[1:])))
                    .repeat(1, *condition.shape[1:])
                )
                used_theta = (
                    self.theta_t[labels]
                    .reshape(condition.shape[0], *([1] * len(condition.shape[1:])))
                    .repeat(1, *condition.shape[1:])
                )
                z = torch.distributions.gamma.Gamma(used_k, 1 / used_theta).sample()
                z = (z - used_k * used_theta) / (1 - used_alphas).sqrt()
            else:
                z = torch.randn_like(condition)
            condition = used_alphas.sqrt() * condition + (1 - used_alphas).sqrt() * z

        z_pred = self.model(perturbed_x, time=t, condition=condition, cond_mask=cond_mask)
        loss = self.criterion(z_pred, z)
        # original code: loss = self.pow_((z - z_pred).reshape(b, -1)).sum(dim=-1).mean(dim=0)

        return loss

    @torch.no_grad()
    def sample(self, condition, **kwargs):
        batch_size = condition.shape[0]

        if self.version == "SMLD":
            from src.models.mcvd import anneal_Langevin_dynamics, anneal_Langevin_dynamics_consistent

            consistent = False  # getattr(self.config.sampling, 'consistent', False)
            sampler = anneal_Langevin_dynamics_consistent if consistent else anneal_Langevin_dynamics
        elif self.version == "DDPM":
            from src.models.mcvd import ddpm_sampler

            sampler = ddpm_sampler
        elif self.version == "DDIM":
            from src.models.mcvd import ddim_sampler

            sampler = ddim_sampler
        elif self.version == "FPNDM":
            from src.models.mcvd import FPNDM_sampler

            sampler = FPNDM_sampler
        else:
            raise NotImplementedError(f"models/unet : version is not DDPM or DDIM! Given: {self.version}")

        # Initial samples
        # n_init_samples = min(36, batch_size)
        init_samples_shape = (batch_size, self.num_input_channels, self.spatial_shape[0], self.spatial_shape[1])
        if self.version == "SMLD":
            init_samples = torch.rand(init_samples_shape, device=condition.device)
        elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
            if self.gamma:
                used_k, used_theta = self.k_cum[0], self.theta_t[0]
                z = (
                    Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta))
                    .sample()
                    .to(condition.device)
                )
                init_samples = z - used_k * used_theta  # we don't scale here
            else:
                init_samples = torch.randn(init_samples_shape, device=condition.device)
        else:
            raise NotImplementedError(f"models/unet : version is not DDPM or DDIM! Given: {self.version}")

        all_samples = sampler(
            init_samples,
            self,
            cond=condition,
            cond_mask=None,
            n_steps_each=self.hparams.n_steps_each,
            step_lr=self.hparams.step_lr,
            just_beta=False,
            final_only=True,
            denoise=self.hparams.sampling_denoise,
            subsample_steps=self.hparams.sampling_timesteps,
            clip_before=False,  # getattr(self.config.sampling, 'clip_before', True),
            log=False,
            gamma=self.gamma,
            **kwargs,
        )
        return {"preds": all_samples[-1]}
