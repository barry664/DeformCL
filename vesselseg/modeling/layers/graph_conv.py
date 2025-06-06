""" Graph conv blocks for Vox2Cortex.

Implementation based on https://github.com/cvlab-epfl/voxel2mesh.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv

class GraphConvNorm(GraphConv):
    """ Wrapper for pytorch3d.ops.GraphConv that normalizes the features
    w.r.t. the degree of the vertices.
    """
    def __init__(self, input_dim: int, output_dim: int, init: str='normal',
                 directed: bool=False, **kwargs):
        super().__init__(input_dim, output_dim, init, directed)

    def forward(self, verts, edges):
        # Normalize with 1 + N(i)
        # Attention: This requires the edges to be unique!
        D_inv = 1.0 / (1 + torch.unique(edges, return_counts=True)[1].unsqueeze(1))
        return D_inv * super().forward(verts, edges)


class Features2FeaturesResidual(nn.Module):
    """ A residual graph conv block consisting of 'hidden_layer_count' many graph convs """

    def __init__(self, input_dim, output_dim, hidden_layer_count,
                 norm='SyncBN', GC=GraphConvNorm):

        super().__init__()

        self.output_dim = output_dim

        self.gconv_first = GC(input_dim, output_dim)
        self.norm_first = nn.BatchNorm1d(output_dim)

        gconv_hidden = []
        for _ in range(hidden_layer_count):
            # No weighted edges and no propagated coordinates in hidden layers
            gc_layer = GC(output_dim, output_dim)
            norm_layer = nn.BatchNorm1d(output_dim)

            gconv_hidden += [nn.Sequential(gc_layer, norm_layer)]

        self.gconv_hidden = nn.Sequential(*gconv_hidden)

    def forward(self, features, edges):
        if features.shape[-1] == self.output_dim:
            res = features
        else:
            res = F.interpolate(features.unsqueeze(1), self.output_dim,
                                mode='nearest').squeeze(1)

        # Conv --> Norm --> ReLU
        features = F.relu(self.norm_first(self.gconv_first(features, edges)))
        for i, (gconv, nl) in enumerate(self.gconv_hidden, 1):
            if i == len(self.gconv_hidden):
                # Conv --> Norm --> Addition --> ReLU
                features = F.relu(nl(gconv(features, edges)) + res)
            else:
                # Conv --> Norm --> ReLU
                features = F.relu(nl(gconv(features, edges)))

        return features
