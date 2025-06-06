import logging
from collections import OrderedDict
from itertools import chain
import numpy as np
import torch
from detectron2.data.catalog import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
import os
from skimage import measure
import SimpleITK as sitk

class SegmentationBaseEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, cfg):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        self._meta = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._predictions = list()
        self.predictions = None

    def reset(self):
        self._predictions = list()

    def process(self, inputs, outputs):
        raise NotImplementedError

    def gather_predictions(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = dict(chain.from_iterable(all_predictions))
        del all_predictions
        self.predictions = predictions

    def evaluate(self):
        raise NotImplementedError


class CommonDiceEvaluator(SegmentationBaseEvaluator):
    def __init__(self, dataset_name, cfg):
        super().__init__(dataset_name, cfg)
        self.model = cfg.MODEL.META_ARCHITECTURE
        self.pred_class = cfg.MODEL.PRED_CLASS
        
        if self.model == "Bbox3d":
            self.save_dir = f'bbox_pred{self.pred_class}'
            os.makedirs(self.save_dir, exist_ok=True)
        
    def process(self, inputs, outputs):
        for inp, output in zip(inputs, outputs):
            file_id = inp['file_id']
            metrics = dict()
            
            seg_pred = output['seg'].squeeze(0)
            seg_gt = inp['seg'].to(seg_pred.device).squeeze(0)

            seg_gt = seg_gt.gt(0).float()
            
            gt_has_label = seg_gt.sum() > 0
            if seg_pred.sum() == 0:
                metrics['seg'] = dict(dice=0)
                self._predictions.append((file_id, metrics))
                continue
            if gt_has_label:
                dice = get_dice_coeff(seg_pred, seg_gt.gt(0)).item()
            else:
                dice = 0
            metrics['seg'] = dict(dice=dice)

            if self.model == "Bbox3d":
                
                labeled_array, _ = measure.label(seg_pred.cpu().numpy(),return_num=True,connectivity=3,)

                volumes = np.bincount(labeled_array.ravel())
                mask = volumes > 10
                mask[0] = 0
                padded = mask[labeled_array]
                
                bbox_pred = torch.nonzero(torch.tensor(padded, device=seg_pred.device))
                
                bbox_pred = bbox_pred[:, 0].min().item(), bbox_pred[:, 0].max().item(), bbox_pred[:, 1].min().item(), bbox_pred[:, 1].max().item(), bbox_pred[:, 2].min().item(), bbox_pred[:, 2].max().item()
                bbox_gt = torch.nonzero(seg_gt)
                bbox_gt = bbox_gt[:, 0].min().item(), bbox_gt[:, 0].max().item(), bbox_gt[:, 1].min().item(), bbox_gt[:, 1].max().item(), bbox_gt[:, 2].min().item(), bbox_gt[:, 2].max().item()
                
                intersection = [max(bbox_pred[0], bbox_gt[0]), min(bbox_pred[1], bbox_gt[1]), max(bbox_pred[2], bbox_gt[2]), min(bbox_pred[3], bbox_gt[3]), max(bbox_pred[4], bbox_gt[4]), min(bbox_pred[5], bbox_gt[5])]
                union = [min(bbox_pred[0], bbox_gt[0]), max(bbox_pred[1], bbox_gt[1]), min(bbox_pred[2], bbox_gt[2]), max(bbox_pred[3], bbox_gt[3]), min(bbox_pred[4], bbox_gt[4]), max(bbox_pred[5], bbox_gt[5])]
                
                intersection = (intersection[1] - intersection[0]) * (intersection[3] - intersection[2]) * (intersection[5] - intersection[4])
                union = (union[1] - union[0]) * (union[3] - union[2]) * (union[5] - union[4])
                
                metrics['bbox_pred'] = dict(bbox_pred=bbox_pred)

                print(f'{file_id} dice: {dice} bbox_pred: {bbox_pred} bbox_gt: {bbox_gt} intersection: {intersection} union: {union}')
        
            self._predictions.append((file_id, metrics))
            

    def evaluate(self):
        if self.predictions is None:
            self.gather_predictions()
        if not comm.is_main_process():
            return
        ret = OrderedDict()
        metrics_list = list(self.predictions.values())
        mean_dice = np.mean([m['seg']['dice'] for m in metrics_list])
        ret["seg"] = dict(dice=mean_dice)
        if self.model == "Bbox3d":
            np.savez_compressed(os.path.join(self.save_dir, f'bbox_pred.npz'
                                ), metrics=self.predictions)
        return ret
    

def get_dice_coeff(pred, target):
    mask = target >= 0
    smooth = 1.
    m1 = pred[mask]
    m2 = target[mask]
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

