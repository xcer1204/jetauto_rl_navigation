from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class RGBCommand(CommandTerm):
    """Samples a one-hot target color command [red, green, blue]."""

    cfg: "RGBCommandCfg"

    def __init__(self, cfg: "RGBCommandCfg", env):
        super().__init__(cfg, env)
        self.rgb_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.rgb_prob = cfg.RGB_prob

    @property
    def command(self) -> torch.Tensor:
        return self.rgb_command

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids: Sequence[int]):
        sampled = torch.multinomial(
            torch.tensor(self.rgb_prob, dtype=torch.float32, device=self.device),
            num_samples=len(env_ids),
            replacement=True,
        )
        self.rgb_command[env_ids] = torch.nn.functional.one_hot(sampled, num_classes=3).float()

    def _update_command(self):
        return

    def _set_debug_vis_impl(self, debug_vis: bool):
        return

    def _debug_vis_callback(self, event):
        return


@configclass
class RGBCommandCfg(CommandTermCfg):
    class_type: type = RGBCommand
    resampling_time_range: tuple[float, float] = (1e5, 1e5)
    RGB_prob: list[float] = [0.34, 0.33, 0.33]
