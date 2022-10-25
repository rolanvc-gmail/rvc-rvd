from torch import nn
from components import (SinusoidalPosEmb,
                        get_backbone,
                        Residual,
                        PreNorm,
                        LinearAttention,
                        Upsample,
                        Downsample,
                        )


class Unet(nn.Module):
    def __int__(
            self,
            dim,
            out_dim=None,
            context_dim_factor=1,
            dim_mults=(1, 1, 2, 2, 4, 4),
            channels = 3,
            with_time_emb=True,
            backbone="resnet"
    ):
        super().__init__()
        self.channels = channels
        self.context_dim_factor = context_dim_factor
        dims = [channels, *map(lambda m: dim*m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU90, nn.Linear(dim*4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out, time_dim)),
                        get_backbone(backbone, (dim_in+int(dim_out*self.context_dim_factor), dim_out, time_dim)),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

