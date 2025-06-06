import torch
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from torch import nn
from .layers.structures import ImageList3d
from .layers.conv_blocks import ShapeSpec3d, get_dice_coeff

__all__ = ["VesselPatchSegmentor3d", ]

@META_ARCH_REGISTRY.register()
class VesselPatchSegmentor3d(nn.Module):

    def __init__(self, cfg):
        super(VesselPatchSegmentor3d, self).__init__()
        self.backbone = build_backbone(cfg, ShapeSpec3d(channels=1))
        self.slide_window = (224, 144, 144)  # (h, w, d)
        self.slide_stride = (112, 72, 72)
        self.loss_type = cfg.MODEL.LOSS
    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs, "image")
        seg_labels = self.preprocess_image(batched_inputs, "seg")
        target = (seg_labels.tensor.gt(0)).float()

        outputs = self.backbone(images.tensor)
        pred_seg = outputs["seg"]
        losses = {"loss_dice": 1 - get_dice_coeff(pred_seg[:, 0].sigmoid(), target[:, 0])}
        return losses

    def inference(self, batched_inputs):
        images = self.preprocess_image(batched_inputs, "image")
        image_sizes = images.image_sizes
        x = images.tensor
        pred_seg = self.slide_inference(x, dist_map=None)
        outputs = {"seg": pred_seg}
        outputs = [{n: o[i_data, :, :s[0], :s[1], :s[2]].gt(0.5) for n, o in outputs.items()}
                   for i_data, s in enumerate(image_sizes)]
        return outputs
    
    def slide_inference(self, image, dist_map=None):
        batch_size, _, h_img, w_img, d_img = image.size()
        h_stride, w_stride, d_stride = self.slide_stride
        h_crop, w_crop, d_crop = self.slide_window
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        d_grids = max(d_img - d_crop + d_stride - 1, 0) // d_stride + 1

        preds = image.new_zeros((batch_size, 1, h_img, w_img, d_img))
        count_mat = image.new_zeros((batch_size, 1, h_img, w_img, d_img))
        

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                for d_idx in range(d_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    z1 = d_idx * d_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    z2 = min(z1 + d_crop, d_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    z1 = max(z2 - d_crop, 0)
                    crop_img = image[:, :, y1:y2, x1:x2, z1:z2]
                    if dist_map is not None:
                        crop_dist_map = dist_map[:, :, y1:y2, x1:x2, z1:z2]
                        crop_seg_logit = self.backbone(crop_img)['seg']
                        crop_seg_logit = torch.cat([crop_seg_logit, crop_dist_map], dim=1)
                        crop_seg_logit = self.merger(crop_seg_logit).sigmoid()
                    else:
                        crop_seg_logit = self.backbone(crop_img)['seg'].sigmoid()
                    preds[:, :, y1:y2, x1:x2, z1:z2] += crop_seg_logit
                    count_mat[:, :, y1:y2, x1:x2, z1:z2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds
    
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