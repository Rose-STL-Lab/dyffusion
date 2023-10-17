from collections import namedtuple
from functools import partial
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from src.diffusion._base_diffusion import BaseDiffusion
from src.diffusion.schedules import cosine_beta_schedule, linear_beta_schedule
from src.utilities.utils import default, extract_into_tensor, identity


# constants
ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


# helpers functions


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules


# gaussian diffusion trainer class


class GaussianDiffusion(BaseDiffusion):
    def __init__(
        self,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_noise",
        beta_schedule="cosine",
        p2_loss_weight_gamma=0.0,
        hardcode_betaN: bool = False,
        # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        ddim_sampling_eta=1.0,
        log_every_t: Union[str, int] = None,
        **kwargs,
    ):
        super().__init__(timesteps=timesteps, **kwargs)
        # assert not (type(self) == GaussianDiffusion), 'GaussianDiffusion is an abstract class, please use a subclass'
        assert (
            not hasattr(self.model.hparams, "learned_sinusoidal_cond")
            or not self.model.hparams.learned_sinusoidal_cond
        )

        self.channels = self.model.num_output_channels
        # self.conditioned = self.model.hparams.conditioned

        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x0",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start)"

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        # if hardcode_betaN:
        #   betas[-1] = 1.

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.diffusion_steps = list(reversed(range(0, self.num_timesteps)))

        self.results_keys = {
            "diffusion_vars": ["x_true", "x_dmodel", "intermediate_x0hat"],
            "output_vars": ["targets", "preds"],
        }

        # sampling related parameters

        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        if self.is_ddim_sampling:
            self.log_text.info(f"using ddim sampling with eta {ddim_sampling_eta}")
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        if hardcode_betaN:
            self.sqrt_alphas_cumprod[-1] = 0.0  # hardcode mult1 to 0
            self.sqrt_one_minus_alphas_cumprod[-1] = 1.0  # hardcode mult2 to 1.

        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # calculate p2 reweighting
        register_buffer(
            "p2_loss_weight", (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, condition=None, clip_x_start=False):
        model_output = self.model(x, time=t, condition=condition)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=False):
        preds = self.model_predictions(x, t, x_self_cond, clip_x_start=clip_denoised)
        x_start, _ = preds.pred_x_start, preds.pred_noise

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, batched_times: Tensor, t: int, x_self_cond=None, clip_denoised=False):
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, condition, shape, log_every_t=None, verbose: bool = False):
        batch_size, device = shape[0], self.betas.device
        log_every_t = log_every_t or self.hparams.log_every_t
        log_every_t = log_every_t if log_every_t != "auto" else self.num_timesteps // 10

        img = torch.randn(shape, device=device)

        # When verbose, use tqdm to show progress bar
        intermediates = dict()
        tsteps_r = (
            tqdm(self.diffusion_steps, desc="Sampling loop time step", total=self.num_timesteps, leave=False)
            if verbose
            else self.diffusion_steps
        )
        for t in tsteps_r:
            # self_cond = x_start if self.conditioned else None
            batched_times = torch.full((batch_size,), t, device=device, dtype=torch.long)
            img, x_start = self.p_sample(img, batched_times, x_self_cond=condition, clip_denoised=False, t=t)

            if (
                False
                and log_every_t
                and (
                    t % log_every_t == 0
                    or t in [1, 2, self.num_timesteps - 10] + list(range(self.num_timesteps - 5, self.num_timesteps))
                )
            ):
                s1 = t  # - 1
                intermediates[f"intermediate_{s1}_x0hat"] = x_start
                intermediates[f"x_{s1}_dmodel"] = img

        # img = unnormalize_to_zero_to_one(img)
        return {"preds": img, **intermediates}

    @torch.no_grad()
    def ddim_sample(self, condition, shape, clip_denoised=False, verbose: bool = True):
        batch, device, total_timesteps, sampling_timesteps, eta, _ = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None
        # When verbose, use tqdm to show progress bar
        time_pairs = tqdm(time_pairs, desc="sampling loop time step", total=len(time_pairs)) if verbose else time_pairs
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, condition, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, condition=None, **kwargs):
        batch_size = condition.shape[0]
        channels = self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(
            condition, shape=(batch_size, channels, self.spatial_shape[0], self.spatial_shape[1]), **kwargs
        )

    def q_sample(self, x_start, t, noise=None):
        """Draw the intermediate degraded data (given the start/target data and the diffused data)"""
        noise = default(noise, lambda: torch.randn_like(x_start))  # create random noise if not provided
        # multiply x_start (and noise) for the alpha cumprod values corresponding to the t's in the batch
        # extract simply returns the alphas for timestep t (in shape of x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, condition, t, noise=None):
        """

        Args:
            x_start: the start/target data
            condition: the condition data
            t: the time step of the diffusion process
            noise: the noise to use to sample the diffused data
        """
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noised data sample, x_t, where t is the corresponding time step (varies for each batch element)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        # x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #    with torch.no_grad():
        #        x_self_cond = self.model_predictions(x, t).pred_x_start
        #        x_self_cond.detach_()

        # predict and take gradient step
        model_preds = self.model(x_t, time=t, condition=condition)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = self.criterion(model_preds, target)  # , reduction='none')
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = (loss * extract(self.p2_loss_weight, t, loss.shape)).mean()
        return loss
