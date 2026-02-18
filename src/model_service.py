from __future__ import annotations

import importlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.settings import settings


# MiniFASNetV2SE model definition from NN.py
class L2Norm(nn.Module):
    def forward(self, input):
        return torch.nn.functional.normalize(input)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel,
                              groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(nn.Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True,
                                      kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return module_input * x


class ResidualSE(nn.Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=4):
        super(ResidualSE, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            if i == num_block - 1:
                modules.append(
                    Depth_Wise_SE(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride,
                                  groups=groups, se_reduct=se_reduct))
            else:
                modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding,
                                          stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Depth_Wise_SE(nn.Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se_reduct=8):
        super(Depth_Wise_SE, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        self.se_module = SEModule(c3_out, se_reduct)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.se_module(x)
            output = short_cut + x
        else:
            output = x
        return output


keep_dict = {'1.8M_': [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
                      13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
                      52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
                      128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512]
             }


class MiniFASNetV2SE(nn.Module):
    def __init__(self, embedding_size=128, conv6_kernel=(5, 5), drop_p=0.75, num_classes=2, img_channel=3):
        super(MiniFASNetV2SE, self).__init__()
        self.embedding_size = embedding_size
        keep = keep_dict['1.8M_']

        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = ResidualSE(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
              (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
              (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
              (keep[33], keep[34]), (keep[36], keep[37])]
        self.conv_4 = ResidualSE(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = ResidualSE(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=drop_p)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


class DefaultSpoofModel(nn.Module):
    """Fallback model if no custom model is specified"""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def _build_model_from_class_path(class_path: str) -> nn.Module:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class()


class ModelService:
    def __init__(self) -> None:
        self.device = self._select_device()
        self.model = self._make_model().to(self.device)
        input_size = settings.model_input_size
        # Preprocessing sesuai dengan ONNX dari hairymax:
        # - Resize ke 128x128
        # - ToTensor: convert ke CHW dan bagi 255
        # - TANPA normalisasi (beda dengan ImageNet)
        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ]
        )
        if settings.default_model_path:
            self.load_weights(settings.default_model_path)

    def _select_device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")

        try:
            major, minor = torch.cuda.get_device_capability(0)
            current_arch = major * 10 + minor
            supported_arches = []
            for arch in torch.cuda.get_arch_list():
                if arch.startswith("sm_"):
                    supported_arches.append(int(arch.split("_", 1)[1]))

            if supported_arches and current_arch > max(supported_arches):
                # Fallback to CPU if GPU architecture is newer than this PyTorch build.
                return torch.device("cpu")
        except Exception:
            # If capability probing fails, keep CUDA and let runtime handle it.
            pass

        return torch.device("cuda")

    def _make_model(self) -> nn.Module:
        if settings.model_class_path:
            return _build_model_from_class_path(settings.model_class_path)
        # MiniFASNet uses a final depthwise conv matching final feature-map size.
        conv6_size = max(1, settings.model_input_size // 16)
        return MiniFASNetV2SE(
            embedding_size=128,
            conv6_kernel=(conv6_size, conv6_size),
            drop_p=0.75,
            num_classes=2,
        )

    def load_weights(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle the state_dict format from hairymax/Face-AntiSpoofing
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                raw_state_dict = checkpoint["state_dict"]
            else:
                raw_state_dict = checkpoint

            model_state_dict = self.model.state_dict()
            filtered_state_dict = OrderedDict()
            for key, value in raw_state_dict.items():
                normalized_key = key
                for prefix in ("module.model.", "model.", "module."):
                    if normalized_key.startswith(prefix):
                        normalized_key = normalized_key[len(prefix):]
                        break

                # Ignore extra heads/layers (e.g. FTGenerator) and shape-mismatched tensors.
                if normalized_key in model_state_dict and model_state_dict[normalized_key].shape == value.shape:
                    filtered_state_dict[normalized_key] = value

            self.model.load_state_dict(filtered_state_dict, strict=False)
        elif isinstance(checkpoint, nn.Module):
            self.model = checkpoint.to(self.device)
        else:
            raise ValueError("Unsupported checkpoint format. Expected state_dict or nn.Module.")

        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Model output: index 0 = real (live), index 1 = fake (spoof)
        # Sesuai dengan format ONNX dari hairymax/Face-AntiSpoofing
        live_score = probs[0][0].item()
        spoof_score = probs[0][1].item()
        
        # Threshold: jika prob_fake >= threshold, maka认定为 fake
        # Disesuaikan dengan hasil percobaan
        # threshold = 0.3
        threshold = 0.14 #IDEAL SAAT INI
        label = "spoof" if spoof_score >= threshold else "live"
        
        return {
            "label": label,
            "live_score": round(live_score, 6),
            "spoof_score": round(spoof_score, 6),
            "threshold": threshold,
            "device": str(self.device),
        }


model_service = ModelService()
