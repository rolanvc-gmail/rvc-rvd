from sinusoidalPosEmb import SinusoidalPosEmb
from Residual import Residual
from Block import Block
from ConvNextBlock import ConvNetBlock
from ResnetBlock import ResnetBlock
from LayerNorm import LayerNorm
from LinearAttention import LinearAttention
from Upsample import Upsample


def get_backbone(name, params):
    if name == "convnext":
        return ConvNextBlock(*params)
    elif name == "resnet":
        return ResnetBlock(*params)
    else:
        raise NotImplementedError

