import logging
import torch
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from torch import nn
from .layers.structures import ImageList3d
from .layers.coords import batched_dist_map
from .layers.conv_blocks import ShapeSpec3d, get_dice_coeff
from .cline_deform_with_seg import Cline_Deformer
__all__ = ["ClineDeformModel", ]

logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class ClineDeformModel(nn.Module):

    def __init__(self, cfg):
        print(cfg)
        super(ClineDeformModel, self).__init__()
        self.backbone = build_backbone(cfg, ShapeSpec3d(channels=1))
        self.cline_deformer = Cline_Deformer(cfg)
        self.feats = ["feat_4x"]
        self.merger = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(3,3,3), stride=1, padding=1)
        self.scratcher = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3,3,3), stride=1, padding=1)
        self.loss_type = cfg.MODEL.LOSS
        
    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        
        assert len(batched_inputs) == 1

        images = self.preprocess_image(batched_inputs, "image")
        gt_seg = self.preprocess_image(batched_inputs, "seg")
        gt_seg = gt_seg.tensor.gt(0.5).float()
        pad_img_shape = images.tensor.shape

        backbone_outputs = self.backbone(images.tensor)
        features = [backbone_outputs[key] for key in self.feats]
    
        losses = dict()
        pred_seg_ = backbone_outputs["seg"]
        pred_seg_s = self.scratcher(pred_seg_)
        loss_dice_ = 1 - get_dice_coeff(pred_seg_s[:, 0].sigmoid(), gt_seg[:, 0])
        pred_cline, loss_cline = self.cline_deformer(batched_inputs, features, pad_img_shape, 
                                                     gt_seg,  pred_seg_s.sigmoid()) 
        losses.update(loss_cline)
        cline_label = batched_inputs[0]['cline'][0]
        gt_cpoints = pred_cline['pred_cline']['verts'] if (torch.rand(1) > 0.35 and loss_cline['loss_local_chamfer_2'] < 0.05) else \
            torch.nonzero(cline_label)
        dist_map = batched_dist_map(pred_seg_[0][0].shape, [gt_cpoints.to(self.device)])
        dist_map = torch.clip(dist_map, min=0, max=3).int()
        dist_map = 1. - dist_map.float() / 3.
        pred_seg = torch.cat([pred_seg_, dist_map], dim=1)
        pred_seg = self.merger(pred_seg)
        loss_dice = 1 - get_dice_coeff(pred_seg[:, 0].sigmoid(), gt_seg[:, 0])
        losses.update(dict(loss_dice_scratch = loss_dice_ * 0.2,
                           loss_dice=loss_dice * 1.))

        return losses

    def inference(self, batched_inputs):
        assert len(batched_inputs) == 1
        images = self.preprocess_image(batched_inputs, "image")
        image_sizes = images.image_sizes
        pad_img_shape = images.tensor.shape

        backbone_outputs = self.backbone(images.tensor)

        features = [backbone_outputs[key] for key in self.feats]
        pred_seg_ = backbone_outputs["seg"]
        pred_seg_s = self.scratcher(pred_seg_)
        ret = self.cline_deformer(batched_inputs, features, pad_img_shape, [],  pred_seg_s)
        pred_cpoints = ret['pred_cline']['verts']
        
        dist_map = batched_dist_map(pred_seg_[0][0].shape, [pred_cpoints.to(self.device)])
        dist_map = torch.clip(dist_map, min=0, max=3).int()
        dist_map = 1. - dist_map.float() / 3.
        pred_seg = torch.cat([pred_seg_, dist_map], dim=1)
        
        pred_seg = self.merger(pred_seg)
        pred_seg = pred_seg.sigmoid().gt(0.5)
        pred_seg = pred_seg[0, 0, :image_sizes[0][0], :image_sizes[0][1], :image_sizes[0][2]]

        output = dict(seg=pred_seg)
        output.update(ret)

        return [output]

    def preprocess_image(self, batched_inputs, key="image"):
        """
        Normalize, pad and batch the input images.
        """
        images = [x[key].to(self.device) for x in batched_inputs]
        if self.training:
            images = ImageList3d.from_tensors(images, (16, 16, 16))
        else:
            images = ImageList3d.from_tensors(images, (16, 16, 16))
        return images

