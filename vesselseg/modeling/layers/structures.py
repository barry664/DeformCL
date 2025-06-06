from typing import Any, List, Sequence, Tuple, Union
import torch


class ImageList3d(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Attributes:
        image_sizes (list[tuple[int, int, int]]): each tuple is (h, w, d)
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W, D) or (N, C_1, ..., C_K, H, W, D) where K >= 1
            image_sizes (list[tuple[int, int, int]]): Each tuple is (h, w, d).
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Returns:
            Tensor: an image of shape (H, W, D) or (C_1, ..., C_K, H, W, D) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1], : size[2]]  # type: ignore

    def to(self, *args: Any, **kwargs: Any) -> "ImageList3d":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList3d(cast_tensor, self.image_sizes)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: Sequence[torch.Tensor], size_divisibilities: tuple = (0, 0, 0), pad_value: float = 0.0
    ) -> "ImageList3d":
        """
        Args:
            tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi, Di) or
                (C_1, ..., C_K, Hi, Wi, Di) where K >= 1. The Tensors will be padded with `pad_value`
                so that they will have the same shape.
            size_divisibilities (list): If `size_divisibility > 0`, also adds padding to ensure
                the common height and width is divisible by `size_divisibility`
            pad_value (float): value to pad

        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[0:-3] == tensors[0].shape[0:-3], t.shape
        # per dimension maximum (H, W, D) or (C_1, ..., C_K, H, W, D) where K >= 1 among all tensors
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        if sum(size_divisibilities) > 0:
            import math

            stride = size_divisibilities
            max_size = list(max_size)  # type: ignore
            max_size[-3] = int(math.ceil(max_size[-3] / stride[-3]) * stride[-3])  # type: ignore
            max_size[-2] = int(math.ceil(max_size[-2] / stride[-2]) * stride[-2])  # type: ignore
            max_size[-1] = int(math.ceil(max_size[-1] / stride[-1]) * stride[-1])  # type: ignore
            max_size = tuple(max_size)

        image_sizes = [tuple(im.shape[-3:]) for im in tensors]

        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new_full(batch_shape, pad_value)
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[..., : img.shape[-3], : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList3d(batched_imgs.contiguous(), image_sizes)
