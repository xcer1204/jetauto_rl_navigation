from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


DEFAULT_MULTITASK_MODEL_PATH = (
    "/home/ubuntu/PersonalFiles/Liyou/deepLabSegment_code/"
    "logs_multitask/multitask_2026_03_16_16_31_06/best_epoch_weights.pth"
)
DEFAULT_OCCLUSION_CLASS_NAMES = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
_WSL_DISTRO_NAME = "Ubuntu"


def _candidate_paths(path_str: str) -> list[Path]:
    raw = str(path_str).strip()
    if not raw:
        return []

    candidates = [Path(raw)]
    if raw.startswith("/") and not raw.startswith("//"):
        candidates.append(Path(rf"\\wsl$\{_WSL_DISTRO_NAME}" + raw.replace("/", "\\")))
    return candidates


def resolve_runtime_path(path_str: str) -> Path:
    candidates = _candidate_paths(path_str)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if not candidates:
        raise FileNotFoundError("Empty path provided for multitask model loading.")

    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve path '{path_str}'. Checked: {joined}")


def _load_external_module(project_root: Path, module_name: str):
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def _iter_tensors(obj) -> Iterable[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        yield obj
        return

    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
        return

    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors(value)


class MultiTaskOcclusionPredictor:
    """Loads an external DeepLabMultiTask checkpoint and predicts occlusion classes from RGB."""

    def __init__(
        self,
        model_path: str = DEFAULT_MULTITASK_MODEL_PATH,
        device: torch.device | str = "cuda",
        project_root: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        print(f"[MultiTaskOcclusionPredictor] Resolving checkpoint path: {model_path}")
        self.model_path = resolve_runtime_path(model_path)
        print(f"[MultiTaskOcclusionPredictor] Checkpoint path: {self.model_path}")
        self.project_root = self._resolve_project_root(project_root)
        print(f"[MultiTaskOcclusionPredictor] Project root: {self.project_root}")

        print("[MultiTaskOcclusionPredictor] Loading checkpoint...")
        checkpoint = torch.load(self.model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            self.checkpoint_args = dict(checkpoint.get("args", {}))
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            self.checkpoint_args = {}
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported checkpoint payload type: {type(state_dict)}")

        self.occlusion_class_names = tuple(
            str(name) for name in self.checkpoint_args.get("occlusion_class_names", DEFAULT_OCCLUSION_CLASS_NAMES)
        )
        input_shape = self.checkpoint_args.get("input_shape", (512, 512))
        self.input_shape = (int(input_shape[0]), int(input_shape[1]))
        self.downsample_factor = int(self.checkpoint_args.get("downsample_factor", 16))

        print("[MultiTaskOcclusionPredictor] Importing DeepLabMultiTask...")
        deeplab_module = _load_external_module(self.project_root, "nets.deeplabv3_multitask")
        model_cls = getattr(deeplab_module, "DeepLabMultiTask")
        print("[MultiTaskOcclusionPredictor] Building model...")
        self.model = model_cls(
            backbone_pretrained=False,
            downsample_factor=self.downsample_factor,
            num_occlusion_classes=len(self.occlusion_class_names),
        )
        print("[MultiTaskOcclusionPredictor] Loading model weights...")
        self._load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._backbone_output = None
        self._backbone_hook = self._resolve_backbone_module().register_forward_hook(self._capture_backbone_output)
        self.feature_dim = self._infer_feature_dim()
        print(f"[MultiTaskOcclusionPredictor] Backbone feature_dim={self.feature_dim}")
        print("[MultiTaskOcclusionPredictor] Ready.")

    def class_index(self, class_name: str) -> int:
        try:
            return self.occlusion_class_names.index(str(class_name))
        except ValueError as exc:
            raise ValueError(
                f"Unknown occlusion class '{class_name}'. Available classes: {self.occlusion_class_names}"
            ) from exc

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        occ_indices, occ_probs, _ = self.predict_with_features(images)
        return occ_indices, occ_probs

    def predict_with_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB images with shape (N, 3, H, W), got {tuple(images.shape)}")

        x = images.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        if tuple(x.shape[-2:]) != self.input_shape:
            x = F.interpolate(x, size=self.input_shape, mode="bilinear", align_corners=False)

        with torch.inference_mode():
            self._backbone_output = None
            outputs = self.model(x)
            occ_logits = self._extract_occlusion_logits(outputs, batch_size=x.shape[0])
            if occ_logits is None:
                raise RuntimeError("Failed to extract occlusion logits from DeepLabMultiTask outputs.")
            features = self._extract_feature_vector(self._backbone_output, batch_size=x.shape[0])
            if features is None:
                raise RuntimeError("Failed to extract backbone features from DeepLabMultiTask.")
            occ_probs = F.softmax(occ_logits.float(), dim=-1)
            occ_indices = occ_probs.argmax(dim=-1)
        return occ_indices, occ_probs, features

    def _resolve_project_root(self, project_root: str | None) -> Path:
        if project_root:
            root = resolve_runtime_path(project_root)
            model_file = root / "nets" / "deeplabv3_multitask.py"
            if model_file.exists():
                return root
            raise FileNotFoundError(f"Invalid DeepLab project root '{root}': missing nets/deeplabv3_multitask.py")

        for parent in [self.model_path.parent, *self.model_path.parents]:
            model_file = parent / "nets" / "deeplabv3_multitask.py"
            if model_file.exists():
                return parent

        raise FileNotFoundError(
            f"Could not infer the DeepLab project root from '{self.model_path}'. "
            "Pass multitask_project_root explicitly."
        )

    def _load_state_dict(self, state_dict: dict) -> None:
        model_dict = self.model.state_dict()
        filtered_state_dict = {}
        for key, value in state_dict.items():
            normalized_key = str(key)
            if normalized_key.startswith("module."):
                normalized_key = normalized_key[len("module.") :]
            if normalized_key in model_dict and tuple(model_dict[normalized_key].shape) == tuple(value.shape):
                filtered_state_dict[normalized_key] = value

        missing = set(model_dict.keys()) - set(filtered_state_dict.keys())
        if not filtered_state_dict:
            raise RuntimeError(f"No compatible model weights found in checkpoint '{self.model_path}'.")

        model_dict.update(filtered_state_dict)
        self.model.load_state_dict(model_dict)
        if missing:
            print(
                "[MultiTaskOcclusionPredictor] Loaded checkpoint with partial parameter match. "
                f"loaded={len(filtered_state_dict)} missing={len(missing)}"
            )

    def _resolve_backbone_module(self):
        for attr_name in ("backbone", "encoder", "feature_extractor"):
            module = getattr(self.model, attr_name, None)
            if module is not None:
                return module
        raise AttributeError("DeepLabMultiTask does not expose a supported backbone module.")

    def _capture_backbone_output(self, _module, _inputs, output) -> None:
        self._backbone_output = output

    def _infer_feature_dim(self) -> int:
        dummy = torch.zeros((1, 3, self.input_shape[0], self.input_shape[1]), device=self.device, dtype=torch.float32)
        _, _, features = self.predict_with_features(dummy)
        if features.ndim != 2 or features.shape[0] != 1:
            raise RuntimeError(f"Unexpected backbone feature shape: {tuple(features.shape)}")
        return int(features.shape[1])

    def _extract_occlusion_logits(self, outputs, batch_size: int) -> torch.Tensor | None:
        if isinstance(outputs, dict):
            for key in ("occ_logits", "occlusion_logits", "occ", "occlusion", "occ_output", "occlusion_output"):
                value = outputs.get(key)
                if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[0] == batch_size:
                    return value

        candidates: list[torch.Tensor] = []
        for tensor in _iter_tensors(outputs):
            if tensor.ndim == 2 and tensor.shape[0] == batch_size:
                candidates.append(tensor)

        for candidate in candidates:
            if candidate.shape[1] == len(self.occlusion_class_names):
                return candidate

        if candidates:
            return candidates[-1]
        return None

    def _extract_feature_vector(self, outputs, batch_size: int) -> torch.Tensor | None:
        tensor = self._select_feature_tensor(outputs, batch_size=batch_size)
        if tensor is None:
            return None
        if tensor.ndim == 4:
            tensor = F.adaptive_avg_pool2d(tensor.float(), output_size=1).flatten(start_dim=1)
        elif tensor.ndim == 3:
            tensor = tensor.float().mean(dim=1)
        elif tensor.ndim == 2:
            tensor = tensor.float()
        else:
            return None
        return tensor

    def _select_feature_tensor(self, outputs, batch_size: int) -> torch.Tensor | None:
        candidates: list[torch.Tensor] = []
        for tensor in _iter_tensors(outputs):
            if tensor.shape[0] != batch_size:
                continue
            if tensor.ndim in (2, 3, 4):
                candidates.append(tensor)

        if not candidates:
            return None

        def _candidate_score(tensor: torch.Tensor) -> tuple[int, int, int]:
            channel_dim = int(tensor.shape[1]) if tensor.ndim >= 2 else 0
            spatial_extent = int(torch.tensor(tensor.shape[2:]).prod().item()) if tensor.ndim > 2 else 1
            return (tensor.ndim, channel_dim, -spatial_extent)

        candidates.sort(key=_candidate_score, reverse=True)
        return candidates[0]
