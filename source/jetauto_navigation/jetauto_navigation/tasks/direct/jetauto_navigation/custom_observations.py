import torch
from isaaclab.envs.mdp import * 
# from your_package.observations import image_features  # ← 引用原类
from isaaclab.managers.manager_term_cfg import ObservationTermCfg


class ImageFeaturesNoHead(image_features):
    """继承原始 image_features，只去掉分类头"""

    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            resnet_weights = {
                "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
                "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
                "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
            }
            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
            # ✅ 去掉分类头
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            image_proc = images.to(model_device)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std
            features = model(image_proc)
            return features.view(features.size(0), -1)

        return {"model": _load_model, "inference": _inference}
