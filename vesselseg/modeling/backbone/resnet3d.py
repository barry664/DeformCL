import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import (Backbone, BACKBONE_REGISTRY)

from ..layers.conv_blocks import (get_norm_3d, Conv3d, CNNBlockBase3d, ShapeSpec3d)


__all__ = [
    "ResNetBlockBase",
    "BasicBlock",
    "BottleneckBlock",
]


class BasicBlock(CNNBlockBase3d):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm_3d(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm_3d(norm, out_channels),
        )

        self.conv2 = Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm_3d(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(CNNBlockBase3d):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm_3d(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv3d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm_3d(norm, bottleneck_channels),
        )

        self.conv2 = Conv3d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm_3d(norm, bottleneck_channels),
        )

        self.conv3 = Conv3d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm_3d(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(CNNBlockBase3d):
    """
    The standard ResNet stem (layers before the first residual block).
    """

    def __init__(self, in_channels=3, out_channels=64, stride=(1, 1, 1), norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)
        self.in_channels = in_channels
        self.conv1 = Conv3d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
            norm=get_norm_3d(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        return x


class ResNet(Backbone):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase3d), block

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))

            self._out_feature_strides[name] = current_stride = tuple(
                [int(s * np.prod([k.stride for k in blocks])) for s in current_stride]
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec3d(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, first_stride, *, in_channels, out_channels, **kwargs):
        """
        Create a list of blocks of the same type that forms one ResNet stage.
        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            first_stride (int): the stride of the first block. The other blocks will have stride=1.
                Therefore this is also the stride of the entire stage.
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of `block_class`.

        Returns:
            list[nn.Module]: a list of block module.
        """
        assert "stride" not in kwargs, "Stride of blocks in make_stage cannot be changed."
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                block_class(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=first_stride if i == 0 else 1,
                    **kwargs,
                )
            )
            in_channels = out_channels
        return blocks


ResNetBlockBase = CNNBlockBase3d
"""
Alias for backward compatibiltiy.
"""


def make_stage(*args, **kwargs):
    """
    Deprecated alias for backward compatibiltiy.
    """
    return ResNet.make_stage(*args, **kwargs)