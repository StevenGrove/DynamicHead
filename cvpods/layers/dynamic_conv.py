import math
import torch
import torch.nn.functional as F
from torch import nn
from prodict import Prodict
from .wrappers import Conv2d
from .batch_norm import get_activation, get_norm
from .masked_conv import masked_conv2d


def get_module_running_cost(net):
    outputs = [[], [], []]
    for module in net.modules():
        if isinstance(module, SpatialGate):
            cost = module.running_cost
            if cost is not None:
                for idx in range(len(cost)):
                    outputs[idx].append(cost[idx].reshape(cost[idx].shape[0], -1).sum(1))
            module.clear_running_cost()
    for idx in range(len(cost)):
        outputs[idx] = sum(outputs[idx])
    return outputs


class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        stride=1, 
        norm="BN", 
        activation=None
    ):
        super().__init__()

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.activation = get_activation(activation)

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

    def masked_inference(self, x, gate):
        gate_max = F.max_pool2d(gate, kernel_size=3, stride=1, padding=1)
        out = masked_conv2d(x, gate_max.squeeze(dim=1), self.conv1.weight,
            self.conv1.bias, 1, 1, [self.conv1.norm, self.activation])
        out = masked_conv2d(out, gate.squeeze(dim=1), self.conv2.weight, 
            self.conv2.bias, 1, 1, self.conv2.norm)
        return out

    def forward(self, x, gate):
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        if not self.training:
            out = gate(x, x, self.masked_inference)
        else:
            out = self.conv1(x)
            out = self.activation(out)
            out = self.conv2(out)
            out = gate(out, x)
            
        out = self.activation(out + shortcut)
        return out


class SpatialGate(nn.Module):
    def __init__(
        self,
        in_channels : int,
        num_groups : int = 1,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None,
        get_running_cost : callable = None
    ):
        super(SpatialGate, self).__init__()
        self.num_groups = num_groups
        self.gate_conv = nn.Conv2d(in_channels,
                                   num_groups,
                                   kernel_size,
                                   padding=padding,
                                   stride=stride)
        self.gate_activation = gate_activation
        self.gate_activation_kargs = gate_activation_kargs
        if gate_activation == "ReTanH":
            self.gate_activate = lambda x : torch.tanh(x).clamp(min=0)
        elif gate_activation == "Sigmoid":
            self.gate_activate = lambda x : torch.sigmoid(x)
        elif gate_activation == "GeReTanH":
            assert "tau" in gate_activation_kargs
            tau = gate_activation_kargs["tau"]
            ttau = math.tanh(tau)
            self.gate_activate = lambda x : ((torch.tanh(x - tau) + ttau) / (1 + ttau)).clamp(min=0)
        else:
            raise NotImplementedError()
        self.get_running_cost = get_running_cost
        self.running_cost = None
        self.init_parameters()

    def init_parameters(self, init_gate=0.99):
        if self.gate_activation == "ReTanH":
            bias_value = 0.5 * math.log((1 + init_gate) /  (1 - init_gate))
        elif self.gate_activation == "Sigmoid":
            bias_value = 0.5 * math.log(init_gate /  (1 - init_gate))
        elif self.gate_activation == "GeReTanH":
            tau = self.gate_activation_kargs["tau"]
            bias_value = 0.5 * math.log((1 + init_gate * math.exp(2 * tau)) /  (1 - init_gate))
        nn.init.normal_(self.gate_conv.weight, std=0.01)
        nn.init.constant_(self.gate_conv.bias, bias_value)

    def encode(self, *inputs):
        outputs = [x.view(x.shape[0] * self.num_groups, -1, *x.shape[2:]) for x in inputs]
        return outputs

    def decode(self, *inputs):
        outputs = [x.view(x.shape[0] // self.num_groups, -1, *x.shape[2:]) for x in inputs]
        return outputs
    
    def update_running_cost(self, gate):
        if self.get_running_cost is not None:
            cost = self.get_running_cost(gate)
            if self.running_cost is not None:
                self.running_cost = [x + y for x, y in zip(self.running_cost, cost)]
            else:
                self.running_cost = cost

    def clear_running_cost(self):
        self.running_cost = None

    def forward(self, data_input, gate_input, masked_func=None):
        gate = self.gate_activate(self.gate_conv(gate_input))
        self.update_running_cost(gate)
        if masked_func is not None: 
            data_input = masked_func(data_input, gate)
        data, gate = self.encode(data_input, gate)
        output, = self.decode(data * gate)
        return output


class DynamicBottleneck(nn.Module):
    def __init__(
        self, 
        in_channels : int,
        out_channels : int,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_groups : int = 1,
        norm: str = "GN",
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None 
    ):
        super(DynamicBottleneck, self).__init__()
        self.num_groups = num_groups
        self.norm = norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck = BasicBlock(in_channels,
                                     out_channels,
                                     stride=stride,
                                     norm=norm,
                                     activation=Prodict(NAME="ReLU", INPLACE=True))
        self.gate = SpatialGate(in_channels,
                                num_groups=num_groups,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                gate_activation=gate_activation,
                                gate_activation_kargs=gate_activation_kargs,
                                get_running_cost=self.get_running_cost)
        self.init_parameters()

    def init_parameters(self):
        self.gate.init_parameters()
        for layer in self.bottleneck.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.GroupNorm):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)

    def get_running_cost(self, gate):
        conv_costs = [x * 3 ** 2 for x in [self.in_channels * self.out_channels, self.out_channels ** 2]]
        if self.in_channels != self.out_channels:
            conv_costs[-1] += self.in_channels * out_channels
        norm_cost = self.out_channels if self.norm != "none" else 0
        unit_costs = [conv_cost + norm_cost for conv_cost in conv_costs]
        
        running_cost = None
        for unit_cost in unit_costs[::-1]:
            num_groups = gate.shape[1]
            hard_gate = (gate != 0).float()
            cost = [gate * unit_cost / num_groups,
                    hard_gate * unit_cost / num_groups,
                    torch.ones_like(gate) * unit_cost / num_groups]
            cost = [x.flatten(1).sum(-1) for x in cost]
            gate = F.max_pool2d(gate, kernel_size=3, stride=1, padding=1)
            gate = gate.max(dim=1, keepdim=True).values
            if running_cost is None:
                running_cost = cost
            else:
                running_cost = [x + y for x, y in zip(running_cost, cost)]
        return running_cost

    def forward(self, input):
        output = self.bottleneck(input, self.gate)
        return output


class DynamicConv2D(nn.Module):
    def __init__(
        self, 
        in_channels : int,
        out_channels : int,
        num_convs : int,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_groups : int = 1,
        norm: str = "GN",
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None,
        depthwise: bool = False
    ):
        super(DynamicConv2D, self).__init__()
        if depthwise:
            assert in_channels == out_channels
        self.num_groups = num_groups
        self.norm = norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        convs = []
        for _ in range(num_convs):
            convs += [nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                groups=in_channels if depthwise else 1),
                      get_norm(norm, in_channels)]
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)
        self.gate = SpatialGate(in_channels,
                                num_groups=num_groups,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                gate_activation=gate_activation,
                                gate_activation_kargs=gate_activation_kargs,
                                get_running_cost=self.get_running_cost)
        self.init_parameters()

    def init_parameters(self):
        self.gate.init_parameters()
        for layer in self.convs.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.GroupNorm):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)

    def get_running_cost(self, gate):
        if self.depthwise:
            conv_cost = self.in_channels * len(self.convs) * \
                self.kernel_size ** 2
        else:
            conv_cost = self.in_channels * self.out_channels * len(self.convs) * \
                self.kernel_size ** 2
        norm_cost = self.out_channels if self.norm != "none" else 0
        unit_cost = conv_cost + norm_cost

        hard_gate = (gate != 0).float()
        cost = [gate.detach() * unit_cost / self.num_groups,
                hard_gate * unit_cost / self.num_groups,
                torch.ones_like(gate) * unit_cost / self.num_groups]
        cost = [x.flatten(1).sum(-1) for x in cost]
        return cost

    def forward(self, input):
        data = self.convs(input)
        output = self.gate(data, input)
        return output


class DynamicScale(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        num_convs: int = 1,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_groups : int = 1,
        num_adjacent_scales: int = 2,
        depth_module: nn.Module = None,
        resize_method: str = "bilinear",
        norm: str = "GN",
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None
    ):
        super(DynamicScale, self).__init__()
        self.num_groups = num_groups
        self.num_adjacent_scales = num_adjacent_scales
        self.depth_module = depth_module
        dynamic_convs = [DynamicConv2D(
            in_channels,
            out_channels,
            num_convs=num_convs,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            num_groups=num_groups,
            norm=norm,
            gate_activation=gate_activation,
            gate_activation_kargs=gate_activation_kargs,
            depthwise=True
        ) for _ in range(num_adjacent_scales)]
        self.dynamic_convs = nn.ModuleList(dynamic_convs)
        if resize_method == "bilinear":
            self.resize = lambda x, s : F.interpolate(
                x, size=s, mode="bilinear", align_corners=True)
        else:
            raise NotImplementedError()
        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))
        self.init_parameters()

    def init_parameters(self):
        for module in self.dynamic_convs:
            module.init_parameters()

    def forward(self, inputs):
        dynamic_scales = []
        for l, x in enumerate(inputs):
            dynamic_scales.append([m(x) for m in self.dynamic_convs])
        
        outputs = []
        for l, x in enumerate(inputs):
            scale_feature = []
            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                if l_source >= 0 and l_source < len(inputs):
                    feature = self.resize(dynamic_scales[l_source][s], x.shape[-2:])
                    scale_feature.append(feature)

            scale_feature = sum(scale_feature) * self.scale_weight + x * self.output_weight
            if self.depth_module is not None:
                scale_feature = self.depth_module(scale_feature)
            outputs.append(scale_feature)

        return outputs
