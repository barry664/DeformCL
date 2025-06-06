import random
import copy
import torch
import numpy as np
import logging
from detectron2.data.transforms import Augmentation, AugmentationList, AugInput

from .transform_gen import RandomCrop
from .transform import FlipTransform, SwapAxesTransform, CropTransform
import torch.nn.functional as F

logger = logging.getLogger(__name__)
class ClineDeformDatasetMapper:

    def __init__(self, cfg, transform_builder, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        augmentations = transform_builder(cfg, is_train)
        self.augmentations = AugmentationList(augmentations)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        self.class_id = cfg.MODEL.PRED_CLASS

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        npz_file = np.load(dataset_dict["file_name"], allow_pickle=True)

        pad = np.array([50, 50, 50])
        cline = npz_file["cline"]
        src_shape = cline.shape
        indices = np.array(np.where(cline == self.class_id))
        start = np.maximum(indices.min(1) - pad, 0).tolist()
        end = np.minimum(indices.max(1) + 1 + pad, src_shape).tolist()

        #  if evaluation, use the bbox predicted by the ROI model
        if not self.is_train:
            class_id = self.class_id
            self.pred_bbox = np.load(f'bbox_pred{class_id}/bbox_pred.npz', allow_pickle=True)
            bbox = self.pred_bbox['metrics'].item()[dataset_dict["file_id"]]['bbox_pred']['bbox_pred']
            start = (bbox[0], bbox[2] * 2, bbox[4] * 2)
            start = np.maximum(start - pad, 0).tolist()
            end = (bbox[1], bbox[3] * 2, bbox[5] * 2)
            end = np.minimum(end + pad, src_shape).tolist()
        image = npz_file["img"].astype(np.float32)
        image = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        image = image / 1024.

        seg = npz_file["seg"]
        seg = seg[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        seg[seg!=self.class_id] = 0
        aug_input = AugInput(image=image, sem_seg=seg)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image[None, ...]))

        cline = cline[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        cline[cline!=self.class_id] = 0
        cline = transforms.apply_image(cline)
        dataset_dict["cline"] = torch.as_tensor(np.ascontiguousarray(cline[None, ...]))

        seg = transforms.apply_image(seg)
        dataset_dict["seg"] = torch.as_tensor(np.ascontiguousarray(seg[None, ...]))

        return dataset_dict

class VesselSegDatasetMapper:

    def __init__(self, cfg, transform_builder, is_train=True):
        self.is_train = is_train
        augmentations = transform_builder(cfg, is_train)
        self.augmentations = AugmentationList(augmentations)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        self.class_id = cfg.MODEL.PRED_CLASS

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        npz_file = np.load(dataset_dict["file_name"], allow_pickle=True)

        image = npz_file["img"].astype(np.float32)
        image = image / 1024.
        seg = npz_file["seg"]
        seg[seg!=self.class_id] = 0
            
        image = interpolate(image, type='img')
        seg = interpolate(seg, type='seg')
        aug_input = AugInput(image=image, sem_seg=seg)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image = torch.as_tensor(np.ascontiguousarray(image[None, ...]))
        dataset_dict["image"] = image

        seg = transforms.apply_image(seg)
        dataset_dict["seg"] = torch.as_tensor(np.ascontiguousarray(seg[None, ...]))

        return dataset_dict


def interpolate(img, type='img'):
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    if type == 'img':
        H, W, D = img.shape[2:]
        new_shape = (H, W // 2, D // 2)
        img_new = F.interpolate(img, size=new_shape, mode='trilinear', align_corners=False)
        img_new = img_new
    else:
        H, W, D = img.shape[2:]
        new_shape = (H, W // 2, D // 2)
        img_new = F.interpolate(img.float(), size=new_shape, mode='nearest')
        img_new = img_new.int()
    return img_new[0, 0].numpy()


def build_cline_deform_transform_gen(cfg, is_train):
    tfm_gens = []
    crop_size = cfg.INPUT.CROP_SIZE_TRAIN
    if is_train:
        tfm_gens.append(RandomCrop("absolute", crop_size))
        tfm_gens.append(RandomCrop("relative_range", (0.9, 0.9, 0.9)))
        tfm_gens.append(RandomFlip_Z(prob=0.5))
        tfm_gens.append(RandomFlip_X(prob=0.5))
        tfm_gens.append(RandomSwapAxesXZ())
        pass
    else:
        pass
    return tfm_gens

def build_bbox_transform_gen(cfg, is_train):
    tfm_gens = []
    crop_size = cfg.INPUT.CROP_SIZE_TRAIN
    if is_train:
        tfm_gens.append(InferCrop("relative_range", (0.75, 0.75, 0.75)))
        tfm_gens.append(InferCrop("absolute", crop_size))
        pass
    else:
        pass
    return tfm_gens


class InferCrop(Augmentation):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, type, crop_size):
        """
        Args:
            prob (float): probability of flip of each axis.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image, sem_seg):
        h, w, d = image.shape[:3]
        if random.random() < 0.2:
            pos = np.where(sem_seg >= 0)
        else:
            pos = np.where(sem_seg > 0)
        idx = np.random.randint(0, len(pos[0]))
        croph, cropw, cropd = self.crop_size if self.type == "absolute" else \
            (
                np.random.randint(int(h*self.crop_size[0]), h + 1), 
                np.random.randint(int(w*self.crop_size[1]), w + 1),
                np.random.randint(int(d*self.crop_size[2]), d + 1)
            )
        jitter = np.random.randint(-50, 50, 3)
        h0 = pos[0][idx] - croph // 2 + jitter[0]
        w0 = pos[1][idx] - cropw // 2 + jitter[1]
        d0 = pos[2][idx] - cropd // 2 + jitter[2]
        h0 = max(0, min(h - croph, h0))
        w0 = max(0, min(w - cropw, w0))
        d0 = max(0, min(d - cropd, d0))
        return CropTransform(h0, w0, d0, croph, cropw, cropd)

    
class RandomFlip_Z(Augmentation):
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
        flip_y = self._rand_range() < 0.
        flip_x = self._rand_range() < 0.
        flip_z = self._rand_range() < self.prob
        return FlipTransform(flip_y, flip_x, flip_z)
    

class RandomFlip_X(Augmentation):
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
        flip_y = self._rand_range() < 0.
        flip_x = self._rand_range() < self.prob
        flip_z = self._rand_range() < 0.
        return FlipTransform(flip_y, flip_x, flip_z)   
class RandomSwapAxesXZ(Augmentation):

    def __init__(self):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        if np.random.rand() < 0.8:
            axes = [0, 1, 2]
        else:
            axes = [0, 2, 1]
        return SwapAxesTransform(axes)