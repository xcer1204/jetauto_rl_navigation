from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def _make_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "selu":
        return nn.SELU()
    if key == "leaky_relu":
        return nn.LeakyReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def _sanitize_layers(values: Sequence[int] | None) -> list[int]:
    if not values:
        return []
    return [int(v) for v in values if int(v) > 0]


def _build_mlp(input_dim: int, layers: Sequence[int], activation: str) -> tuple[nn.Sequential, int]:
    layers = _sanitize_layers(layers)
    if not layers:
        return nn.Identity(), input_dim

    modules: list[nn.Module] = []
    last_dim = input_dim
    for width in layers:
        modules.append(nn.Linear(last_dim, width))
        modules.append(_make_activation(activation))
        last_dim = width
    return nn.Sequential(*modules), last_dim


def _resize_state_batch(state: torch.Tensor, batch_size: int) -> torch.Tensor:
    current_batch = int(state.shape[1])
    if current_batch == batch_size:
        return state
    if current_batch > batch_size:
        return state[:, :batch_size, :].contiguous()

    pad = torch.zeros(
        (state.shape[0], batch_size - current_batch, state.shape[2]),
        dtype=state.dtype,
        device=state.device,
    )
    return torch.cat((state, pad), dim=1)


class _LSTMCore(Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_envs: int = 1,
        encoder_layers: Sequence[int] | None = None,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        sequence_length: int = 1,
        head_layers: Sequence[int] | None = None,
        activation: str = "elu",
    ) -> None:
        super().__init__(observation_space, action_space, device)

        self.num_envs = int(max(1, num_envs))
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.rnn_num_layers = int(rnn_num_layers)
        self.sequence_length = int(max(1, sequence_length))

        self.encoder, encoder_dim = _build_mlp(self.num_observations, encoder_layers or [], activation)
        self.rnn = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
        )
        self.head, self.head_dim = _build_mlp(self.rnn_hidden_size, head_layers or [], activation)

        self.to(self.device)

    def get_specification(self):
        size = (self.rnn_num_layers, self.num_envs, self.rnn_hidden_size)
        return {"rnn": {"sequence_length": self.sequence_length, "sizes": [size, size]}}

    def _zero_state(self, batch_size: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (self.rnn_num_layers, batch_size, self.rnn_hidden_size)
        h = torch.zeros(shape, dtype=dtype, device=self.device)
        c = torch.zeros(shape, dtype=dtype, device=self.device)
        return h, c

    def _coerce_state(
        self,
        rnn_state: Any,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(rnn_state, (list, tuple)) and len(rnn_state) >= 2:
            h = rnn_state[0].to(device=self.device, dtype=dtype)
            c = rnn_state[1].to(device=self.device, dtype=dtype)
        else:
            return self._zero_state(batch_size, dtype)

        expected_flat_batch = batch_size * sequence_length
        if sequence_length > 1 and int(h.shape[1]) == expected_flat_batch:
            h = h[:, ::sequence_length, :].contiguous()
            c = c[:, ::sequence_length, :].contiguous()

        h = _resize_state_batch(h, batch_size)
        c = _resize_state_batch(c, batch_size)
        return h, c

    def _encode_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(0)
        states = states.reshape(states.shape[0], -1)
        return self.encoder(states)

    def _forward_rnn(self, encoded: torch.Tensor, inputs: dict[str, Any]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        terminated = inputs.get("terminated")
        use_sequences = (
            isinstance(terminated, torch.Tensor)
            and self.sequence_length > 1
            and encoded.shape[0] % self.sequence_length == 0
        )

        if use_sequences:
            seq_len = self.sequence_length
            batch_size = encoded.shape[0] // seq_len
            hidden = self._coerce_state(inputs.get("rnn"), batch_size, seq_len, encoded.dtype)

            seq = encoded.view(batch_size, seq_len, -1).transpose(0, 1).contiguous()
            done = terminated.view(batch_size, seq_len, -1).transpose(0, 1).squeeze(-1).bool()

            outputs = []
            h, c = hidden
            for step in range(seq_len):
                if step > 0:
                    keep = (~done[step - 1]).to(dtype=seq.dtype, device=seq.device).view(1, batch_size, 1)
                    h = h * keep
                    c = c * keep
                out_step, (h, c) = self.rnn(seq[step : step + 1], (h, c))
                outputs.append(out_step)

            flat = torch.cat(outputs, dim=0).transpose(0, 1).reshape(-1, self.rnn_hidden_size)
            return flat, [h, c]

        batch_size = encoded.shape[0]
        hidden = self._coerce_state(inputs.get("rnn"), batch_size, 1, encoded.dtype)
        out, (h, c) = self.rnn(encoded.unsqueeze(0), hidden)
        return out.squeeze(0), [h, c]


class LSTMGaussianPolicy(GaussianMixin, _LSTMCore):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_envs: int = 1,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        encoder_layers: Sequence[int] | None = None,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        sequence_length: int = 1,
        head_layers: Sequence[int] | None = None,
        activation: str = "elu",
    ) -> None:
        _LSTMCore.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            num_envs=num_envs,
            encoder_layers=encoder_layers,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            sequence_length=sequence_length,
            head_layers=head_layers,
            activation=activation,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self.mean_layer = nn.Linear(self.head_dim, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))
        self.to(self.device)

    def compute(self, inputs, role):
        encoded = self._encode_states(inputs["states"])
        rnn_features, hidden = self._forward_rnn(encoded, inputs)
        features = self.head(rnn_features)
        mean_actions = self.mean_layer(features)
        return mean_actions, self.log_std_parameter, {"rnn": hidden}


class LSTMDeterministicValue(DeterministicMixin, _LSTMCore):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_envs: int = 1,
        clip_actions: bool = False,
        encoder_layers: Sequence[int] | None = None,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        sequence_length: int = 1,
        head_layers: Sequence[int] | None = None,
        activation: str = "elu",
    ) -> None:
        _LSTMCore.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            num_envs=num_envs,
            encoder_layers=encoder_layers,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            sequence_length=sequence_length,
            head_layers=head_layers,
            activation=activation,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.value_layer = nn.Linear(self.head_dim, 1)
        self.to(self.device)

    def compute(self, inputs, role):
        encoded = self._encode_states(inputs["states"])
        rnn_features, hidden = self._forward_rnn(encoded, inputs)
        features = self.head(rnn_features)
        value = self.value_layer(features)
        return value, {"rnn": hidden}
