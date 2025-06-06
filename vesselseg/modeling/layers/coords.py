import torch
from pytorch3d.ops import knn_points


def get_meshgrid(shape, device):
    assert len(shape) <= 3
    ns = [torch.arange(0, s, dtype=torch.float, device=device) + 0.5 for s in shape]
    grids = torch.meshgrid(*ns)
    return torch.stack(grids, dim=-1)

def batched_dist_map(shape, points):
    """
    compute the minimal distance of each pixel to the input points.
    :param shape: (H, W, D)
    :param points: list of length batchsize, n x 3
    :return: map: (B, H, W, D)
    """

    device = points[0].device
    B = len(points)
    coords = get_meshgrid(shape, device=device)
    coords = coords.view(1, -1, 3).repeat(B, 1, 1)

    length_p = [p.shape[0] for p in points]
    length_p = torch.tensor(length_p, dtype=torch.long, device=device)
    length = max(length_p)
    points_tensor = torch.zeros((B, length, 3), device=device)
    for b in range(B):
        points_tensor[b, :length_p[b]] = points[b]

    coords_nn = knn_points(coords, points_tensor, lengths2=length_p, K=1)
    dist_map = coords_nn.dists.sqrt().squeeze(-1)

    dist_map = dist_map.reshape(B, 1, *shape)

    return dist_map
