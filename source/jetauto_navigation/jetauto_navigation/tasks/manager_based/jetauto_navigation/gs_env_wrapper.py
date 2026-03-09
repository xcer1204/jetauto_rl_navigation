from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch


class GSEnvWrapper(gym.Wrapper):
    """Augment critic observations with the GS feature slice from policy observations."""

    def __init__(
        self,
        env: gym.Env,
        policy_key: str = "policy",
        critic_key: str = "critic",
        policy_term_name: str = "gs_image",
        fallback_to_full_policy: bool = False,
    ):
        super().__init__(env)
        self.policy_key = policy_key
        self.critic_key = critic_key
        self.policy_term_name = policy_term_name
        self.fallback_to_full_policy = fallback_to_full_policy

        self._policy_slice = self._resolve_policy_slice()
        self.observation_space = self._build_observation_space()

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._augment_observations(obs), info

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_observations(obs), reward, terminated, truncated, info

    def _augment_observations(self, obs: Any) -> Any:
        if not isinstance(obs, dict):
            return obs
        if self.policy_key not in obs or self.critic_key not in obs:
            return obs

        policy_obs = obs[self.policy_key]
        critic_obs = obs[self.critic_key]
        if not isinstance(policy_obs, torch.Tensor) or not isinstance(critic_obs, torch.Tensor):
            return obs

        policy_features = self._extract_policy_features(policy_obs)
        if policy_features is None:
            return obs

        out = dict(obs)
        out[self.critic_key] = torch.cat((policy_features, critic_obs), dim=-1)
        return out

    def _extract_policy_features(self, policy_obs: torch.Tensor) -> torch.Tensor | None:
        if self._policy_slice is None:
            if self.fallback_to_full_policy:
                return policy_obs
            return None
        return policy_obs[..., self._policy_slice]

    def _resolve_policy_slice(self) -> slice | None:
        obs_manager = getattr(self.unwrapped, "observation_manager", None)
        if obs_manager is None:
            return None

        active_terms = getattr(obs_manager, "active_terms", {})
        term_dims = getattr(obs_manager, "group_obs_term_dim", {})
        policy_terms = active_terms.get(self.policy_key, [])
        policy_dims = term_dims.get(self.policy_key, [])

        if not policy_terms or not policy_dims:
            return None

        if self.policy_term_name not in policy_terms:
            if self.fallback_to_full_policy:
                total_dim = sum(self._flat_dim(dim) for dim in policy_dims)
                return slice(0, total_dim)
            return None

        term_index = policy_terms.index(self.policy_term_name)
        start = sum(self._flat_dim(dim) for dim in policy_dims[:term_index])
        width = self._flat_dim(policy_dims[term_index])
        return slice(start, start + width)

    def _build_observation_space(self) -> gym.Space:
        obs_space = self.env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict):
            return obs_space
        if self.policy_key not in obs_space.spaces or self.critic_key not in obs_space.spaces:
            return obs_space

        critic_space = obs_space.spaces[self.critic_key]
        if not isinstance(critic_space, gym.spaces.Box):
            return obs_space

        extra_dim = self._policy_feature_dim()
        if extra_dim <= 0:
            return obs_space

        critic_dim = int(np.prod(critic_space.shape)) + extra_dim
        critic_dtype = critic_space.dtype if critic_space.dtype is not None else np.float32

        spaces = dict(obs_space.spaces)
        spaces[self.critic_key] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(critic_dim,),
            dtype=critic_dtype,
        )
        return gym.spaces.Dict(spaces)

    def _policy_feature_dim(self) -> int:
        if self._policy_slice is None:
            if self.fallback_to_full_policy:
                policy_space = getattr(self.env.observation_space, "spaces", {}).get(self.policy_key)
                if isinstance(policy_space, gym.spaces.Box):
                    return int(np.prod(policy_space.shape))
            return 0
        return int(self._policy_slice.stop - self._policy_slice.start)

    @staticmethod
    def _flat_dim(dim: tuple[int, ...]) -> int:
        return int(np.prod(dim))
