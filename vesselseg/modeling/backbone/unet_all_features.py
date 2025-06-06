import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from detectron2.modeling import (Backbone, BACKBONE_REGISTRY)
from ..layers.conv_blocks import (get_norm_3d, Conv3d, ShapeSpec3d)
from .resnet3d import BasicBlock, BottleneckBlock

import torch.nn.functional as F

def silu(input_):
    return F.silu(input_, inplace=True)


func_acts = {
    "relu": F.relu_,
    "elu": F.elu_,
    "silu": silu,
}


modu_acts = {
    "relu": nn.ReLU(inplace=True),
    "elu": nn.ELU(inplace=True),
    "silu": nn.SiLU(inplace=True),
}

class UpBlock(nn.Module):
    def __init__(self, inchs, outchs, encoder_type='conv', use_bias=False, norm=None, activation="relu"):
        super(UpBlock, self).__init__()
        assert use_bias ^ (norm is not None)
        norm_layer_1 = get_norm_3d(norm, inchs) if norm else None
        norm_layer_2 = get_norm_3d(norm, outchs) if norm else None
        self.UpSample = nn.Upsample(scale_factor=2.0, mode='trilinear', align_corners=True)
        if encoder_type == 'basicblock':
            self.out_conv = BasicBlock(inchs + outchs, outchs, norm=norm)
        elif encoder_type == 'bottleneck':
            self.out_conv = BottleneckBlock(inchs + outchs, outchs, bottleneck_channels=inchs, stride_in_1x1=True,
                                            norm=norm)
        elif encoder_type == 'conv':
            self.out_conv = nn.Sequential(
                Conv3d(inchs+outchs, inchs, kernel_size=3, padding=1, bias=use_bias,
                    norm=norm_layer_1, activation=modu_acts[activation]),
                Conv3d(inchs, outchs, kernel_size=3, padding=1, bias=use_bias,
                    norm=norm_layer_2, activation=modu_acts[activation]),
            )
        else:
            raise AssertionError('encoder type should be in ["conv", "basicblock", "bottleneck"]')
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                weight_init.c2_msra_fill(module)

    def forward(self, x, xskip):
        x_up = self.UpSample(x)
        x_cat = torch.cat((x_up, xskip), dim=1)
        out = self.out_conv(x_cat)
        return out


def trivial_top_block_fn(in_channels, out_channels, kernel_size=3, padding=1, activation="relu", **kwargs):
    block = nn.Sequential(
        Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=1),
        modu_acts[activation],
        Conv3d(in_channels, out_channels, kernel_size=1),
    )
    return block

class UNetAllFeatures(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        num_classes = 1
        in_channels = input_shape.channels
        norm = cfg.MODEL.UNETENCODER.NORM
        num_layers = cfg.MODEL.UNETENCODER.NUM_LAYERS
        base_channels = cfg.MODEL.UNETENCODER.BASE_CHANNELS
        inter_channels = cfg.MODEL.OUT_CHANNELS
        return_seg_logits = 'seg' in cfg.MODEL.TASK
        return_inter_feats = 'cline' in cfg.MODEL.TASK

        channels = [base_channels] * (num_layers + 1)
        for i in range(2, num_layers + 1):
            channels[i] = channels[i - 1] * 2

        self.encoders = []
        self.decoders = []
        self.inter_blocks = []
        self._out_features = []
        self._out_feature_channels = {}
        self._out_feature_strides = {}

        self.stem = nn.Sequential(
            Conv3d(in_channels, base_channels // 2, kernel_size=3, stride=1, padding=1, bias=False,
                   norm=get_norm_3d(norm, base_channels // 2), activation=nn.ReLU()),
            Conv3d(base_channels // 2, base_channels, kernel_size=3, stride=1, padding=1,
                   bias=False, norm=get_norm_3d(norm, base_channels), activation=nn.ReLU()),
        )

        for i in range(num_layers):
            feature_stride = 2 ** (i + 1)
            name = 'downfeat_' + str(feature_stride) + 'x'
            stage = nn.Sequential(
                Conv3d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, bias=False,
                       norm=get_norm_3d(norm, channels[i + 1]), activation=nn.ReLU()),
                Conv3d(channels[i + 1], channels[i + 1], kernel_size=3, stride=1, padding=1, bias=False,
                       norm=get_norm_3d(norm, channels[i + 1]), activation=nn.ReLU()),
            )
            self.add_module(name, stage)
            self.encoders.append((stage, name))

        bottom_name = 'feat_' + str(2 ** len(self.encoders)) + 'x'
        self._out_features.append(bottom_name)
        self._out_feature_channels[bottom_name] = channels[-1]
        self._out_feature_strides[bottom_name] = 2 ** len(self.encoders)

        for i in range(num_layers)[::-1]:
            feature_stride = 2 ** i
            name = 'feat_' + str(feature_stride) + 'x'
            stage = UpBlock(channels[i + 1], channels[i], 'conv', use_bias=False, norm=norm)
            self.add_module(name, stage)
            self.decoders.append((stage, name))
            if return_inter_feats and i == 2:
                inter_block = trivial_top_block_fn(channels[i], inter_channels[0], activation="relu")
                self.add_module('inter_' + name, inter_block)
                self.inter_blocks.append(inter_block)
            self._out_features.append(name)
            self._out_feature_channels[name] = channels[i]
            self._out_feature_strides[name] = feature_stride

        self.return_inter_feats = return_inter_feats
        self.return_seg_logits = return_seg_logits
        if return_seg_logits:
            self.top_block = trivial_top_block_fn(base_channels, out_channels=num_classes, stride=1, norm=norm,
                                    activation="relu")

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                weight_init.c2_msra_fill(module)

    def forward(self, x):
        x = self.stem(x)
        down_feats = [x]
        for stage, name in self.encoders:
            x = stage(x)
            down_feats.append(x)
        down_feats = down_feats[::-1]
        feat1 = down_feats[0]
        bottom_name = 'feat_' + str(2 ** len(self.encoders)) + 'x'
        outputs = {bottom_name: feat1}
        for idx, (stage, name) in enumerate(self.decoders):
            feat2 = down_feats[idx + 1]
            feat1 = stage(feat1, feat2)
            feat3 = self.inter_blocks[0](feat1) if self.return_inter_feats and name == 'feat_4x' \
                else feat1
            outputs[name] = feat3

        if self.return_seg_logits:
            outputs['seg'] = self.top_block(feat1)

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec3d(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, (stage, _) in enumerate(self.encoders + self.decoders, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self


@BACKBONE_REGISTRY.register()
def build_unetaf_backbone(cfg, input_shape):
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    return UNetAllFeatures(cfg, input_shape).freeze(freeze_at)
