from __future__ import annotations

import torch

from ..mdp.multitask_inference import (
    DEFAULT_MULTITASK_MODEL_PATH,
    DEFAULT_OCCLUSION_CLASS_NAMES,
    MultiTaskOcclusionPredictor,
    _load_external_module,
    resolve_runtime_path,
)


class MidOccMultiTaskOcclusionPredictor(MultiTaskOcclusionPredictor):
    """Multitask predictor loader that supports checkpoints without the view branch."""

    def __init__(
        self,
        model_path: str = DEFAULT_MULTITASK_MODEL_PATH,
        device: torch.device | str = "cuda",
        project_root: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        print(f"[MidOccMultiTaskOcclusionPredictor] Resolving checkpoint path: {model_path}")
        self.model_path = resolve_runtime_path(model_path)
        print(f"[MidOccMultiTaskOcclusionPredictor] Checkpoint path: {self.model_path}")
        self.project_root = self._resolve_project_root(project_root)
        print(f"[MidOccMultiTaskOcclusionPredictor] Project root: {self.project_root}")

        print("[MidOccMultiTaskOcclusionPredictor] Loading checkpoint...")
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
        self.enable_view_branch = not bool(self.checkpoint_args.get("disable_view_branch", False))

        print("[MidOccMultiTaskOcclusionPredictor] Importing DeepLabMultiTask...")
        deeplab_module = _load_external_module(self.project_root, "nets.deeplabv3_multitask")
        model_cls = getattr(deeplab_module, "DeepLabMultiTask")
        print(
            "[MidOccMultiTaskOcclusionPredictor] Building model... "
            f"enable_view_branch={self.enable_view_branch}"
        )
        try:
            self.model = model_cls(
                backbone_pretrained=False,
                downsample_factor=self.downsample_factor,
                num_occlusion_classes=len(self.occlusion_class_names),
                enable_view_branch=self.enable_view_branch,
            )
        except TypeError:
            self.model = model_cls(
                backbone_pretrained=False,
                downsample_factor=self.downsample_factor,
                num_occlusion_classes=len(self.occlusion_class_names),
            )
        print("[MidOccMultiTaskOcclusionPredictor] Loading model weights...")
        self._load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._backbone_output = None
        self._backbone_hook = self._resolve_backbone_module().register_forward_hook(self._capture_backbone_output)
        self.feature_dim = self._infer_feature_dim()
        print(f"[MidOccMultiTaskOcclusionPredictor] Backbone feature_dim={self.feature_dim}")
        print("[MidOccMultiTaskOcclusionPredictor] Ready.")
