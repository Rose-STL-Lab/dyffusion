import math
from abc import abstractmethod
from contextlib import ExitStack
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.diffusion._base_diffusion import BaseDiffusion
from src.experiment_types.interpolation import InterpolationExperiment
from src.interface import get_checkpoint_from_path_or_wandb
from src.utilities.utils import freeze_model, raise_error_if_invalid_value


class BaseDYffusion(BaseDiffusion):
    def __init__(
        self,
        forward_conditioning: str = "data",
        schedule: str = "before_t1_only",
        additional_interpolation_steps: int = 0,
        additional_interpolation_steps_factor: int = 0,
        interpolate_before_t1: bool = False,
        sampling_type: str = "cold",  # 'cold' or 'naive'
        sampling_schedule: Union[List[float], str] = None,
        time_encoding: str = "dynamics",
        refine_intermediate_predictions: bool = False,
        prediction_timesteps: Optional[Sequence[float]] = None,
        enable_interpolator_dropout: Union[bool, str] = True,
        log_every_t: Union[str, int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, sampling_schedule=sampling_schedule)
        sampling_schedule = None if sampling_schedule == "None" else sampling_schedule
        self.save_hyperparameters(ignore=["model"])
        self.num_timesteps = self.hparams.timesteps

        fcond_options = ["data", "none", "data+noise"]
        raise_error_if_invalid_value(forward_conditioning, fcond_options, "forward_conditioning")

        # Add additional interpolation steps to the diffusion steps
        # we substract 2 because we don't want to use the interpolator in timesteps outside [1, num_timesteps-1]
        horizon = self.num_timesteps  # = self.interpolator_horizon
        assert horizon > 1, f"horizon must be > 1, but got {horizon}. Please use datamodule.horizon with > 1"
        if schedule == "linear":
            assert (
                additional_interpolation_steps == 0
            ), "additional_interpolation_steps must be 0 when using linear schedule"
            self.additional_interpolation_steps_fac = additional_interpolation_steps_factor
            if interpolate_before_t1:
                interpolated_steps = horizon - 1
                self.di_to_ti_add = 0
            else:
                interpolated_steps = horizon - 2
                self.di_to_ti_add = additional_interpolation_steps_factor

            self.additional_diffusion_steps = additional_interpolation_steps_factor * interpolated_steps
        elif schedule == "before_t1_only":
            assert (
                additional_interpolation_steps_factor == 0
            ), "additional_interpolation_steps_factor must be 0 when using before_t1_only schedule"
            assert interpolate_before_t1, "interpolate_before_t1 must be True when using before_t1_only schedule"
            self.additional_diffusion_steps = additional_interpolation_steps
        else:
            raise ValueError(f"Invalid schedule: {schedule}")

        self.num_timesteps += self.additional_diffusion_steps
        d_to_i_step = {d: self.diffusion_step_to_interpolation_step(d) for d in range(1, self.num_timesteps)}
        self.dynamical_steps = {d: i_n for d, i_n in d_to_i_step.items() if float(i_n).is_integer()}
        self.i_to_diffusion_step = {i_n: d for d, i_n in d_to_i_step.items()}
        self.artificial_interpolation_steps = {d: i_n for d, i_n in d_to_i_step.items() if not float(i_n).is_integer()}
        # check that float tensors and floats return the same value
        for d, i_n in d_to_i_step.items():
            i_n2 = float(self.diffusion_step_to_interpolation_step(torch.tensor(d, dtype=torch.float)))
            assert math.isclose(
                i_n, i_n2, abs_tol=4e-6
            ), f"float and tensor return different values for diffusion_step_to_interpolation_step({d}): {i_n} != {i_n2}"
        # note that self.dynamical_steps does not include t=0, which is always dynamical (but not an output!)
        if additional_interpolation_steps_factor > 0 or additional_interpolation_steps > 0:
            self.log_text.info(
                f"Added {self.additional_diffusion_steps} steps.. total diffusion num_timesteps={self.num_timesteps}. \n"
                # f'Mapping diffusion -> interpolation steps: {d_to_i_step}. \n'
                f"Diffusion -> Dynamical timesteps: {self.dynamical_steps}."
            )
        self.enable_interpolator_dropout = enable_interpolator_dropout
        raise_error_if_invalid_value(enable_interpolator_dropout, [True, False], "enable_interpolator_dropout")
        if refine_intermediate_predictions:
            self.log_text.info("Enabling refinement of intermediate predictions.")

        # which diffusion steps to take during sampling
        self.full_sampling_schedule = list(range(0, self.num_timesteps))
        self.sampling_schedule = sampling_schedule or self.full_sampling_schedule

    @property
    def diffusion_steps(self) -> List[int]:
        return list(range(0, self.num_timesteps))

    def diffusion_step_to_interpolation_step(self, diffusion_step: Union[int, Tensor]) -> Union[float, Tensor]:
        """
        Convert a diffusion step to an interpolation step
        Args:
            diffusion_step: the diffusion step  (in [1, num_timesteps-1])
        Returns:
            the interpolation step
        """
        # assert correct range
        if torch.is_tensor(diffusion_step):
            assert (0 <= diffusion_step).all() and (
                diffusion_step <= self.num_timesteps - 1
            ).all(), f"diffusion_step must be in [1, num_timesteps-1]=[{1}, {self.num_timesteps - 1}], but got {diffusion_step}"
        else:
            assert (
                0 <= diffusion_step <= self.num_timesteps - 1
            ), f"diffusion_step must be in [1, num_timesteps-1]=[1, {self.num_timesteps - 1}], but got {diffusion_step}"
        if self.hparams.schedule == "linear":
            i_n = (diffusion_step + self.di_to_ti_add) / (self.additional_interpolation_steps_fac + 1)
        elif self.hparams.schedule == "before_t1_only":
            # map d_N to h-1, d_N-1 to h-2, ..., d_n to 1, and d_n-1..d_1 uniformly to [0, 1)
            # e.g. if h=5, then d_5 -> 4, d_4 -> 3, d_3 -> 2, d_2 -> 1, d_1 -> 0.5
            # or                d_6 -> 4, d_5 -> 3, d_4 -> 2, d_3 -> 1, d_2 -> 0.66, d_1 -> 0.33
            # or                d_7 -> 4, d_6 -> 3, d_5 -> 2, d_4 -> 1, d_3 -> 0.75, d_2 -> 0.5, d_1 -> 0.25
            if torch.is_tensor(diffusion_step):
                i_n = torch.where(
                    diffusion_step >= self.additional_diffusion_steps + 1,
                    (diffusion_step - self.additional_diffusion_steps).float(),
                    diffusion_step / (self.additional_diffusion_steps + 1),
                )
            elif diffusion_step >= self.additional_diffusion_steps + 1:
                i_n = diffusion_step - self.additional_diffusion_steps
            else:
                i_n = diffusion_step / (self.additional_diffusion_steps + 1)
        else:
            raise ValueError(f"schedule=``{self.hparams.schedule}`` not supported.")

        return i_n

    def q_sample(
        self,
        x0,
        x_end,
        t: Optional[Tensor],
        interpolation_time: Optional[Tensor] = None,
        is_artificial_step: bool = True,
        **kwargs,
    ) -> Tensor:
        # q_sample = using model in interpolation mode
        # just remember that x_end here refers to t=0 (the initial conditions)
        # and x_0 (terminology of diffusion models) refers to t=T, i.e. the last timestep
        assert t is None or interpolation_time is None, "Either t or interpolation_time must be None."
        t = interpolation_time if t is None else self.diffusion_step_to_interpolation_step(t)  # .float()
        do_enable = self.training or self.enable_interpolator_dropout

        ipol_handles = [self.interpolator] if hasattr(self, "interpolator") else [self]
        with ExitStack() as stack:
            # inference_dropout_scope of all handles (enable and disable) is managed by the ExitStack
            for ipol in ipol_handles:
                stack.enter_context(ipol.inference_dropout_scope(condition=do_enable))

            x_ti = self._interpolate(initial_condition=x_end, x_last=x0, t=t, **kwargs)
        return x_ti

    @abstractmethod
    def _interpolate(
        self,
        initial_condition: Tensor,
        x_last: Tensor,
        t: Tensor,
        static_condition: Optional[Tensor] = None,
        num_predictions: int = 1,
    ):
        """This is an internal method. Please use q_sample to access it."""
        raise NotImplementedError(f"``_interpolate`` must be implemented in {self.__class__.__name__}")

    def get_condition(
        self,
        condition,
        x_last: Optional[Tensor],
        prediction_type: str,
        static_condition: Optional[Tensor] = None,
        shape: Sequence[int] = None,
    ) -> Tensor:
        if static_condition is None:
            return condition
        elif condition is None:
            return static_condition
        else:
            return torch.cat([condition, static_condition], dim=1)

    def _predict_last_dynamics(self, forward_condition: Tensor, x_t: Tensor, t: Tensor):
        if self.hparams.time_encoding == "discrete":
            time = t
        elif self.hparams.time_encoding == "normalized":
            time = t / self.num_timesteps
        elif self.hparams.time_encoding == "dynamics":
            time = self.diffusion_step_to_interpolation_step(t)
        else:
            raise ValueError(f"Invalid time_encoding: {self.hparams.time_encoding}")

        x_last_pred = self.model.predict_forward(x_t, time=time, condition=forward_condition)
        return x_last_pred

    def predict_x_last(
        self,
        condition: Tensor,
        x_t: Tensor,
        t: Tensor,
        is_sampling: bool = False,
        static_condition: Optional[Tensor] = None,
    ):
        # predict_x_last = using model in forward mode
        assert (0 <= t).all() and (t <= self.num_timesteps - 1).all(), f"Invalid timestep: {t}"
        cond_type = self.hparams.forward_conditioning
        if cond_type == "data":
            forward_cond = condition
        elif cond_type == "none":
            forward_cond = None
        elif "data+noise" in cond_type:
            # simply use factor t/T to scale the condition and factor (1-t/T) to scale the noise
            # this is the same as using a linear combination of the condition and noise
            tfactor = t / (self.num_timesteps - 1)  # shape: (b,)
            tfactor = tfactor.view(condition.shape[0], *[1] * (condition.ndim - 1))  # shape: (b, 1, 1, 1)
            # add noise to the data in a linear combination, s.t. the noise is more important at the beginning (t=0)
            # and less important at the end (t=T)
            forward_cond = tfactor * condition + (1 - tfactor) * torch.randn_like(condition)
        else:
            raise ValueError(f"Invalid forward conditioning type: {cond_type}")

        forward_cond = self.get_condition(
            condition=forward_cond,
            x_last=None,
            prediction_type="forward",
            static_condition=static_condition,
            shape=condition.shape,
        )
        x_last_pred = self._predict_last_dynamics(x_t=x_t, forward_condition=forward_cond, t=t)
        return x_last_pred

    @property
    def sampling_schedule(self) -> List[Union[int, float]]:
        return self._sampling_schedule

    @sampling_schedule.setter
    def sampling_schedule(self, schedule: Union[str, List[Union[int, float]]]):
        """Set the sampling schedule. At the very minimum, the sampling schedule will go through all dynamical steps.
        Notation:
        - N: number of diffusion steps
        - h: number of dynamical steps
        - h_0: first dynamical step

        Options for diffusion sampling schedule trajectories ('<name>': <description>):
        - 'only_dynamics': the diffusion steps corresponding to dynamical steps (this is the minimum)
        - 'only_dynamics_plus_discreteINT': add INT discrete non-dynamical steps, uniformly drawn between 0 and h_0
        - 'only_dynamics_plusINT': add INT non-dynamical steps (possibly continuous), uniformly drawn between 0 and h_0
        - 'everyINT': only use every INT-th diffusion step (e.g. 'every2' for every second diffusion step)
        - 'firstINT': only use the first INT diffusion steps
        - 'firstFLOAT': only use the first FLOAT*N diffusion steps

        """
        schedule_name = schedule
        if isinstance(schedule_name, str):
            base_schedule = [0] + list(self.dynamical_steps.keys())  # already included: + [self.num_timesteps - 1]
            artificial_interpolation_steps = list(self.artificial_interpolation_steps.keys())
            if "only_dynamics" in schedule_name:
                schedule = []  # only sample from base_schedule (added below)

                if "only_dynamics_plus" in schedule_name:
                    # parse schedule 'only_dynamics_plusN' to get N
                    plus_n = int(schedule_name.replace("only_dynamics_plus", "").replace("_discrete", ""))
                    # Add N additional steps to the front of the schedule
                    schedule = list(np.linspace(0, base_schedule[1], plus_n + 1, endpoint=False))
                    if "_discrete" in schedule_name:  # floor the values
                        schedule = [int(np.floor(s)) for s in schedule]
                else:
                    assert "only_dynamics" == schedule_name, f"Invalid sampling schedule: {schedule}"

            elif schedule_name.startswith("every"):
                # parse schedule 'everyNth' to get N
                every_nth = schedule.replace("every", "").replace("th", "").replace("nd", "").replace("rd", "")
                every_nth = int(every_nth)
                assert 1 <= every_nth <= self.num_timesteps, f"Invalid sampling schedule: {schedule}"
                schedule = artificial_interpolation_steps[::every_nth]

            elif schedule.startswith("first"):
                # parse schedule 'firstN' to get N
                first_n = float(schedule.replace("first", "").replace("v2", ""))
                if first_n < 1:
                    assert 0 < first_n < 1, f"Invalid sampling schedule: {schedule}, must end with number/float > 0"
                    first_n = int(np.ceil(first_n * len(artificial_interpolation_steps)))
                    schedule = artificial_interpolation_steps[:first_n]
                    self.log_text.info(f"Using sampling schedule: {schedule_name} -> (first {first_n} steps)")
                else:
                    assert first_n.is_integer(), f"If first_n >= 1, it must be an integer, but got {first_n}"
                    assert 1 <= first_n <= self.num_timesteps, f"Invalid sampling schedule: {schedule}"
                    first_n = int(first_n)
                    # Simple schedule: sample using first N steps
                    schedule = artificial_interpolation_steps[:first_n]
            else:
                raise ValueError(f"Invalid sampling schedule: ``{schedule}``. ")

            # Add dynamic steps to the schedule
            schedule += base_schedule
            # need to sort in ascending order and remove duplicates
            schedule = list(sorted(set(schedule)))

        assert (
            1 <= schedule[-1] <= self.num_timesteps
        ), f"Invalid sampling schedule: {schedule}, must end with number/float <= {self.num_timesteps}"
        if schedule[0] != 0:
            self.log_text.warning(
                f"Sampling schedule {schedule_name} must start at 0. Adding 0 to the beginning of it."
            )
            schedule = [0] + schedule

        last = schedule[-1]
        if last != self.num_timesteps - 1:
            self.log_text.warning("------" * 20)
            self.log_text.warning(
                f"Are you sure you don't want to sample at the last timestep? (current last timestep: {last})"
            )
            self.log_text.warning("------" * 20)

        # check that schedule is monotonically increasing
        for i in range(1, len(schedule)):
            assert schedule[i] > schedule[i - 1], f"Invalid sampling schedule not monotonically increasing: {schedule}"

        if all(float(s).is_integer() for s in schedule):
            schedule = [int(s) for s in schedule]
        else:
            self.log_text.info(f"Sampling schedule {schedule_name} uses diffusion steps it has not been trained on!")
        self._sampling_schedule = schedule

    def sample_loop(
        self,
        initial_condition,
        static_condition: Optional[Tensor] = None,
        log_every_t: Optional[Union[str, int]] = None,
        num_predictions: int = None,
    ):
        batch_size = initial_condition.shape[0]
        log_every_t = log_every_t or self.hparams.log_every_t
        log_every_t = log_every_t if log_every_t != "auto" else 1

        sc_kw = dict(static_condition=static_condition)
        assert len(initial_condition.shape) == 4, f"condition.shape: {initial_condition.shape} (should be 4D)"
        x_s = initial_condition[:, -self.num_input_channels :]
        intermediates, x0_hat, dynamics_pred_step = dict(), None, 0
        last_i_n_plus_one = self.sampling_schedule[-1] + 1
        s_and_snext = zip(
            self.sampling_schedule,
            self.sampling_schedule[1:] + [last_i_n_plus_one],
            self.sampling_schedule[2:] + [last_i_n_plus_one, last_i_n_plus_one],
        )
        for s, s_next, s_nnext in tqdm(
            s_and_snext, desc="Sampling time step", total=len(self.sampling_schedule), leave=False
        ):
            is_last_step = s == self.num_timesteps - 1
            # F(x_s, s) = predict target data
            step_s = torch.full((batch_size,), s, dtype=torch.float32, device=self.device)
            x0_hat = self.predict_x_last(condition=initial_condition, x_t=x_s, t=step_s, is_sampling=True, **sc_kw)

            # Are we predicting dynamical time step or an artificial interpolation step?
            time_i_n = self.diffusion_step_to_interpolation_step(s_next) if not is_last_step else np.inf
            is_dynamics_pred = float(time_i_n).is_integer() or is_last_step
            q_sample_kwargs = dict(
                x0=x0_hat,
                x_end=initial_condition,
                is_artificial_step=not is_dynamics_pred,
                reshape_ensemble_dim=not is_last_step,
                num_predictions=1 if is_last_step else num_predictions,
            )
            if s_next <= self.num_timesteps - 1:
                # D(x_s, s-1)
                step_s_next = torch.full((batch_size,), s_next, dtype=torch.float32, device=self.device)
                x_interpolated_s_next = self.q_sample(**q_sample_kwargs, t=step_s_next, **sc_kw)
            else:
                x_interpolated_s_next = x0_hat  # for the last step, we use the final x0_hat prediction

            if self.hparams.sampling_type in ["cold"]:
                # D(x_s, s)
                x_interpolated_s = self.q_sample(**q_sample_kwargs, t=step_s, **sc_kw) if s > 0 else x_s
                # for s = 0, we have x_s_degraded = x_s, so we just directly return x_s_degraded_next
                x_s = x_s - x_interpolated_s + x_interpolated_s_next

            elif self.hparams.sampling_type == "naive":
                x_s = x_interpolated_s_next
            else:
                raise ValueError(f"unknown sampling type {self.hparams.sampling_type}")

            dynamics_pred_step = int(time_i_n) if s < self.num_timesteps - 1 else dynamics_pred_step + 1
            if is_dynamics_pred:
                intermediates[f"t{dynamics_pred_step}_preds"] = x_s  # preds
                if log_every_t is not None:
                    intermediates[f"t{dynamics_pred_step}_preds2"] = x_interpolated_s_next

            s1, s2 = s, s  # s + 1, next_step  # s, next_step
            if log_every_t is not None:
                intermediates[f"intermediate_{s1}_x0hat"] = x0_hat
                intermediates[f"xipol_{s2}_dmodel"] = x_interpolated_s_next
                if self.hparams.sampling_type == "cold":
                    intermediates[f"xipol_{s1}_dmodel2"] = x_interpolated_s

        if self.hparams.refine_intermediate_predictions:
            # Use last prediction of x0 for final prediction of intermediate steps (not the last timestep!)
            q_sample_kwargs["x0"] = x0_hat
            q_sample_kwargs["is_artificial_step"] = False
            dynamical_steps = self.hparams.prediction_timesteps or list(self.dynamical_steps.values())
            dynamical_steps = [i for i in dynamical_steps if i < self.num_timesteps]
            for i_n in dynamical_steps:
                i_n_time_tensor = torch.full((batch_size,), i_n, dtype=torch.float32, device=self.device)
                i_n_for_str = int(i_n) if float(i_n).is_integer() else i_n
                assert (
                    not float(i_n).is_integer() or f"t{i_n_for_str}_preds" in intermediates
                ), f"t{i_n_for_str}_preds not in intermediates"
                intermediates[f"t{i_n_for_str}_preds"] = self.q_sample(
                    **q_sample_kwargs, t=None, interpolation_time=i_n_time_tensor, **sc_kw
                )

        if last_i_n_plus_one < self.num_timesteps:
            return x_s, intermediates, x_interpolated_s_next
        return x0_hat, intermediates, x_s

    @torch.no_grad()
    def sample(self, initial_condition, num_samples=1, **kwargs):
        x_0, intermediates, x_s = self.sample_loop(initial_condition, **kwargs)
        return intermediates


# --------------------------------------------------------------------------------
# DYffusion with a pretrained interpolator
# --------------------------------------------------------------------------------


class DYffusion(BaseDYffusion):
    """
    DYffusion model with a pretrained interpolator
    Args:
        interpolator: the interpolator model
        lambda_reconstruction: the weight of the reconstruction loss
        lambda_reconstruction2: the weight of the reconstruction loss (using the predicted xt_last as feedback)
    """

    def __init__(
        self,
        interpolator: Optional[nn.Module] = None,
        interpolator_run_id: Optional[str] = None,
        interpolator_local_checkpoint_path: Optional[str] = None,
        interpolator_wandb_ckpt_filename: Optional[str] = None,
        lambda_reconstruction: float = 1.0,
        lambda_reconstruction2: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["interpolator", "model"])
        self.interpolator: InterpolationExperiment = get_checkpoint_from_path_or_wandb(
            interpolator,
            interpolator_local_checkpoint_path,
            interpolator_run_id,
            wandb_kwargs=dict(epoch="best", ckpt_filename=interpolator_wandb_ckpt_filename),
        )
        # freeze the interpolator (and set to eval mode)
        freeze_model(self.interpolator)

        self.interpolator_window = self.interpolator.window
        self.interpolator_horizon = self.interpolator.true_horizon
        last_d_to_i_tstep = self.diffusion_step_to_interpolation_step(self.num_timesteps - 1)
        if self.interpolator_horizon != last_d_to_i_tstep + 1:
            # maybe: automatically set the num_timesteps to the interpolator_horizon
            raise ValueError(
                f"interpolator horizon {self.interpolator_horizon} must be equal to the "
                f"last interpolation step+1=i_N=i_{self.num_timesteps - 1}={last_d_to_i_tstep + 1}"
            )

    def _interpolate(
        self, initial_condition: Tensor, x_last: Tensor, t: Tensor, static_condition: Optional[Tensor] = None, **kwargs
    ):
        # interpolator networks uses time in [1, horizon-1]
        assert (0 < t).all() and (
            t < self.interpolator_horizon
        ).all(), f"interpolate time must be in (0, {self.interpolator_horizon}), got {t}"
        # select condition data to be consistent with the interpolator training data
        interpolator_inputs = torch.cat([initial_condition, x_last], dim=1)
        kwargs["reshape_ensemble_dim"] = False
        interpolator_outputs = self.interpolator.predict(
            interpolator_inputs, condition=static_condition, time=t, **kwargs
        )
        interpolator_outputs = interpolator_outputs["preds"]
        return interpolator_outputs

    def p_losses(self, xt_last: Tensor, condition: Tensor, t: Tensor, static_condition: Tensor = None):
        r"""

        Args:
            xt_last: the start/target data  (time = horizon)
            condition: the initial condition data  (time = 0)
            t: the time step of the diffusion process
            static_condition: the static condition data (if any)
        """
        # x_t is what multi-horizon exp passes as targets, and xt_last is the last timestep of the data dynamics
        # check that the time step is valid (between 0 and horizon-1)
        # assert torch.all(t >= 0) and torch.all(t <= self.num_timesteps-1), f'invalid time step {t}'
        lam1 = self.hparams.lambda_reconstruction
        lam2 = self.hparams.lambda_reconstruction2

        # Get the interpolated
        # since we do not need to interpolate xt_0, we can skip all batches where t=0
        t_nonzero = t > 0
        x_interpolated = self.q_sample(
            x_end=condition[t_nonzero],
            x0=xt_last[t_nonzero],
            t=t[t_nonzero],
            static_condition=None if static_condition is None else static_condition[t_nonzero],
            num_predictions=1,  # sample one interpolation prediction
        )
        # Now, simply concatenate the inital_conditions for t=0 with the interpolated data for t>0
        x_t = condition.clone()
        x_t[t_nonzero] = x_interpolated.to(x_t.dtype)
        # assert torch.all(x_t[t == 0] == condition[t == 0]), f'x_t[t == 0] != condition[t == 0]'
        # Train the forward predictions (i.e. predict xt_last from xt_t)
        xt_last_target = xt_last
        xt_last_pred = self.predict_x_last(condition=condition, x_t=x_t, t=t, static_condition=static_condition)
        loss_forward = self.criterion(xt_last_pred, xt_last_target)

        # Train the forward predictions II by emulating one more step of the diffusion process
        tnot_last = t <= self.num_timesteps - 2
        t2 = t[tnot_last] + 1  # t2 is the next time step, between 1 and T-1
        calc_t2 = tnot_last.any()
        if lam2 > 0 and calc_t2:
            # train the predictions using x0 = xlast = forward_pred(condition, t=0)
            cond_notlast = condition[tnot_last]
            x0not_last = xt_last_pred[tnot_last]
            sc_notlast = None if static_condition is None else static_condition[tnot_last]

            # Use the predictions of xt_last = x0_not_last to interpolate the next step
            x_interpolated2 = self.q_sample(
                x_end=cond_notlast,
                x0=x0not_last,
                t=t2,
                static_condition=sc_notlast,
                num_predictions=1,
            )
            x_last_pred2 = self.predict_x_last(
                condition=cond_notlast, x_t=x_interpolated2, t=t2, static_condition=sc_notlast
            )
            loss_forward2 = self.criterion(x_last_pred2, xt_last_target[tnot_last])
        else:
            loss_forward2 = 0.0

        loss = lam1 * loss_forward + lam2 * loss_forward2

        log_prefix = "train" if self.training else "val"
        loss_dict = {
            "loss": loss,
            f"{log_prefix}/loss_forward": loss_forward,
            f"{log_prefix}/loss_forward2": loss_forward2,
        }
        return loss_dict
