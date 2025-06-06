import numpy as np
from fvcore.transforms.transform import Transform

class CropTransform(Transform):
    def __init__(self, y0: int, x0: int, z0: int, h: int, w: int, d: int):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped image(s).
        """
        assert len(img.shape) in {3, 4}
        return img[self.y0: self.y0 + self.h, self.x0: self.x0 + self.w, self.z0: self.z0 + self.d]


    def apply_coords(self, coords: np.ndarray):
        return NotImplementedError


class FlipTransform(Transform):
    """
    Perform flip along each axis.
    """

    def __init__(self, flip_y: bool=False, flip_x: bool=False, flip_z: bool=False):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        assert len(img.shape) in {3, 4}

        if self.flip_y:
            img = np.flip(img, axis=0)
        if self.flip_x:
            img = np.flip(img, axis=1)
        if self.flip_z:
            img = np.flip(img, axis=2)
        return img


    def apply_coords(self, coords: np.ndarray):
        return NotImplementedError


class SwapAxesTransform(Transform):
    """
    Perform axes swap.
    """

    def __init__(self, axes):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        assert len(img.shape) in {3, 4}
        if len(img.shape) == 4:
            self.axes.append(3)
        return np.transpose(img, self.axes)

    def apply_coords(self, coords: np.ndarray):
        return NotImplementedError