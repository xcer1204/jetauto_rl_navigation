import torch
# from isaaclab.envs.mdp import * 
from isaaclab.envs.mdp.observations import image_features
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



import torch
import numpy as np
import threading
import socket
# import pickle
import atexit
import rpyc

from torchvision import models, transforms


class GSServer:
    """从 render_server.py 通过 socket 接收批量 RGB 图像。"""

    def __init__(self, host="localhost", port=12345, time_out=10):
        self.host = host
        self.port = port
        self.time_out = time_out
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

    def init_data(self, env_num: int, H: int = 180, W: int = 320):
        # 和 VR-Robo 一致：每个 env 存一条展平的 (3 * H * W)
        self.H, self.W = H, W
        self.data = np.zeros((env_num, 3 * H * W), dtype=np.uint8)
        self.last_data = np.zeros((env_num, 3 * H * W), dtype=np.uint8)
        # 随机 0/1 延迟，用来模拟图像 delay（和原版一致，你嫌烦可以全设 0）
        self.latency = np.random.randint(0, 2, size=(env_num, 1), dtype=np.int32)
        self.env_num = env_num

    # def _receive_once(self):
    #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     s.bind((self.host, self.port))
    #     s.listen(1)
    #     conn, addr = s.accept()
    #     conn.settimeout(self.time_out)
    #     data = b""
    #     try:
    #         while True:
    #             packet = conn.recv(40960000)
    #             if not packet:
    #                 break
    #             data += packet
    #     except socket.timeout:
    #         print("[GSServer] recv timeout, no new tensor.")
    #     finally:
    #         conn.close()
    #         s.close()
    #     if not data:
    #         return None
    #     arr = pickle.loads(data)  # (N, 3*H*W)
    #     return arr


    def _receive_once(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.host, self.port))
        s.listen(1)
        conn, addr = s.accept()
        conn.settimeout(self.time_out)

        try:
            # 先收 8 字节 header：N, FLAT
            header = conn.recv(8)
            if len(header) < 8:
                print("[GSServer] header too short.")
                conn.close()
                s.close()
                return None

            N, FLAT = np.frombuffer(header, dtype=np.int32)
            expected_bytes = int(N) * int(FLAT)

            # 再收图像 payload
            buf = bytearray()
            while len(buf) < expected_bytes:
                packet = conn.recv(min(40960000, expected_bytes - len(buf)))
                if not packet:
                    break
                buf.extend(packet)

        except socket.timeout:
            print("[GSServer] recv timeout, no new tensor.")
            conn.close()
            s.close()
            return None
        finally:
            conn.close()
            s.close()

        if len(buf) != expected_bytes:
            print(f"[GSServer] size mismatch: got {len(buf)}, expect {expected_bytes}")
            return None

        # 恢复成 numpy 数组 (N, FLAT)，类型 uint8
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(N, FLAT)
        return arr

    def start(self):
        atexit.register(self.close)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def _run(self):
        while self.running:
            arr = self._receive_once()
            if arr is None:
                continue
            with self.lock:
                self.last_data = self.data
                self.data = arr

    # def get_data(self):
    #     """返回 shape=(N, 3*H*W) 的 uint8 numpy 数组。"""
    #     with self.lock:
    #         is_start = (self.last_data == 0).all(axis=1).reshape(-1, 1)
    #         latency = self.latency
    #         out = (latency * self.last_data + (1 - latency) * self.data) * (1 - is_start) + self.data * is_start
    #     return out

    def get_data(self):
        """返回 shape=(N, 3*H*W) 的 uint8 numpy 数组。"""
        with self.lock:
            is_start = (self.last_data == 0).all(axis=1).reshape(-1, 1)
            latency = self.latency.astype(np.int32)

            out = (latency * self.last_data + (1 - latency) * self.data) * (1 - is_start) + self.data * is_start

            out = np.clip(out, 0, 255).astype(np.uint8)

        return out

    def reset(self, env_ids):
        if env_ids is None:
            return
        env_ids = env_ids.cpu().numpy()
        self.data[env_ids] = 0
        self.last_data[env_ids] = 0
        self.latency[env_ids] = np.random.randint(0, 2, size=(len(env_ids), 1), dtype=np.int32)


class GSImageFeatures:
    """
    通过 rpyc 把相机位姿 + 目标位置发给 render_server，
    再从 GSServer 取回图像，经过 ResNet18 提取 (N, 512) 特征。
    """

    def __init__(self, model_name: str = "resnet18", device: str = "cuda",
                 num_envs: int = 1, H: int = 180, W: int = 320):
        self.device = device
        self.H, self.W = H, W

        # 1) rpyc 连接到 render_server.py 里的 RenderService
        self.conn = rpyc.connect("localhost", 18861, config={"allow_pickle": True})

        # 2) 启动 GSServer，接收 render_server 的 socket 数据
        self.image_server = GSServer()
        self.image_server.start()
        self.image_server.init_data(num_envs, H=H, W=W)

        # 3) 加载 ResNet backbone（去掉分类头）
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights).eval()
        self.backbone = torch.nn.Sequential(*(list(net.children())[:-1]))
        self.backbone.to(device)

        # 4) 图像预处理（可以先简单一点，后面再像 VR-Robo 那样加各种抖动/模糊）
        self.preprocess = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        self.image_server.reset(env_ids)

    def __call__(self, env) -> torch.Tensor:
        """
        env: 你的 JetautoNavigationEnv
        返回: (num_envs, 512) 的特征
        """

        # ---------------------------
        # 1) 构造相机位姿 + 目标位置
        # ---------------------------
        # 直接用 camera_a 的世界位姿（已经绑在机器人上了）
        cam_pos = env._camera_a.data.pos_w.clone()            # (num_envs, 3)
        cam_quat = env._camera_a.data.quat_w_world.clone()    # (num_envs, 4)

        # 这里先不做坐标系转换，直接当成 render_server 的 “pos, ori”
        # 如果你 3DGS 坐标约定是 ROS，相机坐标不同，再去加转换
        # 目标物体：你现在只有一个 Target_A，就全部当成 red_cone 传进去
        # target_pos = env._target_a.data.root_pos_w - env.scene.env_origins   # (N,3)

        #TODO 如果你有多个目标/障碍物，可以分别传给 red/green/blue_cone
        # red_cone = target_pos
        # green_cone = target_pos
        # blue_cone = target_pos

        # 发送到 render_server（不关心返回值，图像走 socket）
        # 注意：rpyc 这边可以直接传 torch.Tensor，原 VR-Robo 就是这么干的
        self.conn.root.exposed_render(cam_pos, cam_quat)
        # ---------------------------
        # 2) 从 GSServer 拿图像
        # ---------------------------
        arr = self.image_server.get_data()   # numpy, (N, 3*H*W), uint8


        # ===== DEBUG: 保存接收到的图片 =====
        if getattr(env, "step_count", 0) == 0:
            H, W = self.H, self.W

            img_np = arr[0].reshape(H, W, 3).astype(np.uint8)
            from PIL import Image
            Image.fromarray(img_np).save("received_from_render_server.png")


        images = torch.from_numpy(arr).to(self.device).float()
        images = images.view(env.num_envs, 3, self.H, self.W)   # (N,3,H,W)
        images = images / 255.0

        # ---------------------------
        # 3) 送入 ResNet 提取特征
        # ---------------------------
        x = self.preprocess(images)
        with torch.no_grad():
            feat = self.backbone(x)           # (N, 512, 1, 1)
            feat = feat.view(env.num_envs, -1)

        return feat