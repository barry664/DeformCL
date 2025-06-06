import numpy as np
from detectron2.data.transforms import Augmentation

from .transform import CropTransform, FlipTransform, SwapAxesTransform

class RandomCrop(Augmentation):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute"]
        self._init(locals())

    def get_transform(self, img):
        h, w, d = img.shape[:3]
        croph, cropw, cropd = self.get_crop_size((h, w, d))
        assert h >= croph and w >= cropw and d >= cropd, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        d0 = np.random.randint(d - cropd + 1)
        return CropTransform(h0, w0, d0, croph, cropw, cropd)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w, d = image_size
        if self.crop_type == "relative":
            ch, cw, cd = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5), int(d * cd + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw, cd = crop_size + np.random.rand(3) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5), int(d * cd + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w), min(self.crop_size[2], d))
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomFlip(Augmentation):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): probability of flip of each axis.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        flip_y = self._rand_range() < self.prob
        flip_x = self._rand_range() < self.prob
        flip_z = self._rand_range() < self.prob
        return FlipTransform(flip_y, flip_x, flip_z)


class RandomSwapAxes(Augmentation):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        axes = [0, 1, 2]
        np.random.shuffle(axes)
        return SwapAxesTransform(axes)

