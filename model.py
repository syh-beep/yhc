import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models import efficientnet_b0 as tv_efficientnet_b0


# ========== 工具函数 ==========
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer())


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class InvertedResidualConfig:
    def __init__(self,
                 kernel: int,
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,
                 stride: int,
                 use_se: bool,
                 drop_rate: float,
                 index: str,
                 width_coefficient: float):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = OrderedDict()
        activation_layer = nn.SiLU

        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x
        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(EfficientNet, self).__init__()

        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    cnf[-3] = 1
                    cnf[1] = cnf[2]
                cnf[-1] = args[-2] * b / num_blocks
                index = str(stage + 1) + chr(i + 97)
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        layers = OrderedDict()
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ========== 注意力机制 ==========
class DifferentialAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, lambda_init: float = 0.8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.half_dim = self.head_dim // 2
        self.scale = self.half_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.k_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.half_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.half_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.half_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.half_dim) * 0.1)
        self.lambda_init = lambda_init

        self.norm = nn.GroupNorm(num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def _lambda(self):
        lam = ((self.lambda_q1 * self.lambda_k1).sum(-1).exp() -
               (self.lambda_q2 * self.lambda_k2).sum(-1).exp() + self.lambda_init)
        return torch.clamp(lam, min=0.0, max=2.0)  # 数值稳定性

    def forward(self, x: Tensor, mask=None):
        B, T, C = x.shape
        H, d_half = self.num_heads, self.half_dim

        Q = self.q_proj(x).view(B, T, H, -1).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, -1).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, -1).transpose(1, 2)

        Q1, Q2 = Q[..., :d_half], Q[..., d_half:]
        K1, K2 = K[..., :d_half], K[..., d_half:]

        A1 = F.softmax(Q1 @ K1.transpose(-2, -1) * self.scale, dim=-1)
        A2 = F.softmax(Q2 @ K2.transpose(-2, -1) * self.scale, dim=-1)

        if mask is not None:
            A1 = A1.masked_fill(mask, 0.0)
            A2 = A2.masked_fill(mask, 0.0)

        lam = self._lambda().view(1, H, 1, 1)
        A = A1 - lam * A2
        A = torch.clamp(A, min=-10.0, max=10.0)  # 防止指数爆炸
        A = self.dropout(F.softmax(A, dim=-1))
        out = (A @ V).transpose(1, 2).reshape(B, T, C)

        out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        return self.o_proj(out)


class AgentAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_agents: int = 64,
                 use_dwc: bool = True, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.H = num_heads
        self.d = d_model // num_heads
        self.a = num_agents
        self.scale = self.d ** -0.5
        self.use_dwc = use_dwc

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.agents = nn.Parameter(torch.randn(1, num_agents, d_model))
        self.agent_bias = nn.Parameter(torch.zeros(1, num_heads, 1, num_agents))

        if use_dwc:
            self.dwc = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor):
        B, T, C = x.shape
        H, d, a = self.H, self.d, self.a

        Q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        ag = self.agents.expand(B, -1, -1)
        A = self.q_proj(ag).view(B, a, H, d).transpose(1, 2)

        attn_ag2kv = A @ K.transpose(-2, -1) * self.scale
        attn_ag2kv = F.softmax(attn_ag2kv, dim=-1)
        attn_ag2kv = self.dropout(attn_ag2kv)
        V_agent = attn_ag2kv @ V

        attn_q2ag = Q @ A.transpose(-2, -1) * self.scale
        attn_q2ag = attn_q2ag + self.agent_bias
        attn_q2ag = F.softmax(attn_q2ag, dim=-1)
        attn_q2ag = self.dropout(attn_q2ag)
        out = attn_q2ag @ V_agent

        out = out.transpose(1, 2).reshape(B, T, C)

        if self.use_dwc:
            local = self.dwc(x.transpose(1, 2)).transpose(1, 2)
            out = out + local

        return self.o_proj(self.norm(out))


# ========== 双注意力模型（支持预训练） ==========
class EfficientNetWithBothAttention(nn.Module):
    def __init__(self, width_coefficient: float, depth_coefficient: float,
                 num_classes: int, dropout_rate: float = 0.2,
                 agent_kwargs: dict = None, diff_kwargs: dict = None,
                 pretrained: bool = True):
        super().__init__()
        if pretrained:
            # 加载预训练的 EfficientNet-B0
            base_model = tv_efficientnet_b0(weights='DEFAULT')
            self.features = base_model.features
            # 获取特征图通道数
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                feat = self.features(dummy)
                c = feat.shape[1]
        else:
            base_model = EfficientNet(width_coefficient, depth_coefficient,
                                      num_classes=num_classes, dropout_rate=dropout_rate)
            self.features = base_model.features
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                feat = self.features(dummy)
                c = feat.shape[1]

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        hw = feat.shape[2] * feat.shape[3]
        # Agent 配置
        default_agent = {
            'num_heads': 4,
            'num_agents': min(16, hw),
            'use_dwc': False,
            'dropout': 0.1
        }
        if agent_kwargs:
            default_agent.update(agent_kwargs)
        self.agent_attn = AgentAttention(d_model=c, **default_agent)

        # Diff 配置
        default_diff = {
            'num_heads': 4,
            'lambda_init': 0.8,
            'dropout': 0.1
        }
        if diff_kwargs:
            default_diff.update(diff_kwargs)
        self.diff_attn = DifferentialAttention(d_model=c, **default_diff)

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)   # (B, 49, C)

        # Agent Attention
        agent_out = self.agent_attn(x_seq)
        x_seq = self.norm1(x_seq + agent_out)

        # Differential Attention
        diff_out = self.diff_attn(x_seq)
        x_seq = self.norm2(x_seq + diff_out)

        x = x_seq.transpose(1, 2).view(B, C, H, W)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ========== 模型创建函数 ==========
def efficientnet_b0(num_classes=1000):
    return EfficientNet(width_coefficient=1.0, depth_coefficient=1.0,
                        dropout_rate=0.2, num_classes=num_classes)


def efficientnet_b0_agent(num_classes=1000, num_agents=16, num_heads=4, use_dwc=False):
    attn_kwargs = {
        'num_heads': num_heads,
        'num_agents': num_agents,
        'use_dwc': use_dwc,
        'dropout': 0.1
    }
    return EfficientNetWithAttention(width_coefficient=1.0, depth_coefficient=1.0,
                                     num_classes=num_classes, attn_type='agent',
                                     attn_kwargs=attn_kwargs)


def efficientnet_b0_diff(num_classes=1000, num_heads=4, lambda_init=0.8):
    attn_kwargs = {
        'num_heads': num_heads,
        'lambda_init': lambda_init,
        'dropout': 0.1
    }
    return EfficientNetWithAttention(width_coefficient=1.0, depth_coefficient=1.0,
                                     num_classes=num_classes, attn_type='diff',
                                     attn_kwargs=attn_kwargs)


def efficientnet_b0_both(num_classes=1000,
                         agent_heads=4, agent_num=12,
                         diff_heads=4, lambda_init=0.8,
                         pretrained=True):
    agent_kwargs = {'num_heads': agent_heads, 'num_agents': agent_num, 'use_dwc': False, 'dropout': 0.1}
    diff_kwargs = {'num_heads': diff_heads, 'lambda_init': lambda_init, 'dropout': 0.1}
    return EfficientNetWithBothAttention(width_coefficient=1.0, depth_coefficient=1.0,
                                         num_classes=num_classes,
                                         agent_kwargs=agent_kwargs,
                                         diff_kwargs=diff_kwargs,
                                         pretrained=pretrained)
# ========== 仅 Agent Attention 模型 ==========
class EfficientNetAgentOnly(nn.Module):
    def __init__(self, width_coefficient: float, depth_coefficient: float,
                 num_classes: int, dropout_rate: float = 0.2,
                 agent_kwargs: dict = None):
        super().__init__()
        # 构建 EfficientNet 特征提取器
        base_model = EfficientNet(width_coefficient, depth_coefficient,
                                 num_classes=1000, dropout_rate=dropout_rate)
        self.features = base_model.features
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.features(dummy)
            c = feat.shape[1]

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # Agent Attention 默认配置
        default_agent = {
            'num_heads': 4,
            'num_agents': 12,          # 与之前保持一致
            'use_dwc': False,
            'dropout': 0.1
        }
        if agent_kwargs:
            default_agent.update(agent_kwargs)
        self.agent_attn = AgentAttention(d_model=c, **default_agent)
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)   # (B, 49, C)
        x_seq = self.norm(x_seq + self.agent_attn(x_seq))
        x = x_seq.transpose(1, 2).view(B, C, H, W)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)