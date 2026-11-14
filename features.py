"""Patch features from a frozen ImageNet ResNet-18 (TorchVision), PatchCore style.

layer2 (stride 8) and layer3 (stride 16) feature maps are taken from the frozen backbone,
layer3 is upsampled to the layer2 grid, both are concatenated and locally averaged over a
3x3 neighbourhood so that each patch describes its surroundings. For a 224 px input that
gives a 28x28 grid of 384-dimensional patches. Nothing is trained; the backbone is the
official checkpoint and the only learning is the memory bank of good-part patches.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class ResNetPatchFeatures:
    def __init__(self, input_size=224, threads=4, cache_dir=None, layers=("layer2", "layer3")):
        if cache_dir:
            os.environ["TORCH_HOME"] = str(Path(cache_dir).resolve())
        import torch
        import torchvision
        from torchvision.models import ResNet18_Weights

        torch.set_num_threads(int(threads))
        self.torch = torch
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = torchvision.models.resnet18(weights=self.weights).eval()
        self.layers = tuple(layers)
        self.input_size = int(input_size)
        self.captured = {}
        for name in self.layers:
            getattr(self.model, name).register_forward_hook(self._hook(name))
        self.grid_shape = (self.input_size // 8, self.input_size // 8)
        self.dim = sum({"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}[n] for n in self.layers)

    def _hook(self, name):
        def capture(module, inputs, output):
            self.captured[name] = output
        return capture

    def preprocess(self, image):
        from PIL import Image

        resized = image.convert("RGB").resize((self.input_size, self.input_size), Image.BILINEAR)
        array = (np.asarray(resized, np.float32) / 255.0 - MEAN) / STD
        return self.torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)

    def extract(self, image):
        """Returns an array (patches, dim) in row-major grid order."""
        tensor = self.preprocess(image)
        with self.torch.no_grad():
            self.model(tensor)
            maps = []
            target = self.captured[self.layers[0]].shape[-2:]
            for name in self.layers:
                fmap = self.captured[name]
                if fmap.shape[-2:] != target:
                    fmap = self.torch.nn.functional.interpolate(fmap, size=target, mode="bilinear", align_corners=False)
                maps.append(fmap)
            stacked = self.torch.cat(maps, dim=1)
            pooled = self.torch.nn.functional.avg_pool2d(stacked, kernel_size=3, stride=1, padding=1)
        array = pooled[0].permute(1, 2, 0).reshape(-1, self.dim).numpy().astype(np.float32)
        return array

    def info(self):
        import torchvision

        checkpoint = Path(self.torch.hub.get_dir()) / "checkpoints" / Path(self.weights.url).name
        return {"backbone": "resnet18", "weights": str(self.weights), "url": self.weights.url,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None,
                "layers": list(self.layers), "input_size": self.input_size, "grid_shape": list(self.grid_shape), "dim": self.dim,
                "local_pool": "3x3 average", "torchvision": torchvision.__version__, "torch": self.torch.__version__}
