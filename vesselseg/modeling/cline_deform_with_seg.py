from typing import List
import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch import nn
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import (
    chamfer_distance, 
)
import networkx as nx

from .layers.coord_transform import normalize_vertices, unnormalize_vertices
from .layers.transformer_encoder import TransformerEncoderLayer, TransformerEncoder
from .layers import BasicFeatureSampling
from  skimage.morphology import skeletonize_3d
import torch.nn.functional as F
from scipy import interpolate
logger = logging.getLogger('ClineDeform')

def get_dice_coeff(pred, target):
    smooth = 1.
    m1 = pred.float()
    m2 = target.float()
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def BezierInterpo(pts, N, k=3):
    # make sure pts are all distinct
    pts = pts + 0.0001 * np.random.rand(*pts.shape)
    smoothness = len(pts) * 0.1
    w = np.ones(len(pts))
    interpo = interpolate.splprep([pts[:, i] for i in range(3)], k=k, w=w,  s=smoothness)
    tck, u = interpo
    u_new = np.linspace(0, 1, N)
    x_new, y_new, z_new = interpolate.splev(u_new, tck)
    ret = np.stack([x_new, y_new, z_new], axis=1)
    return ret

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # res = x
        x = F.relu(self.input_layer(x))
        for i in range(self.n_layers - 2):
            x = F.relu(self.hidden_layers[i](x))
        x = self.output_layer(x)
        return x

class f2v(nn.Module):
    def __init__(self):
        super(f2v, self).__init__()
        self.linear = nn.Linear(24, 3)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.enc = nn.Linear(24, 24)
    def forward(self, x):
        x = self.enc(x)
        # x = F.relu(x)
        return self.linear(x)
    
class TransformerEnc6(nn.Module):
    def __init__(self, input_dim):
        super(TransformerEnc6, self).__init__()
        self.enc = TransformerEncoder(
                            TransformerEncoderLayer(input_dim, 4, 
                                dim_feedforward=24, dropout=0.1, activation='relu'),
                            num_layers=6,
                    )
    def forward(self, x, src_key_padding_mask=None, pos=None):
        res = x
        return self.enc(x, src_key_padding_mask=src_key_padding_mask, pos=pos) + res

class Cline_Deformer(nn.Module):

    def __init__(self, cfg):
        super(Cline_Deformer, self).__init__()
        self.num_classes = 1
        self.num_steps = cfg.MODEL.DEFORM.NUM_STEPS
        self.norm = cfg.MODEL.DEFORM.NORM
        self.loss_edge_weight = cfg.MODEL.DEFORM.LOSS_EDGE_WEIGHT
        self.sdf_loss_weight = cfg.MODEL.DEFORM.SDF_LOSS_WEIGHT
        self.control_pts_num = cfg.MODEL.N_CONTROL_POINTS
        self.lower_thres = cfg.MODEL.DEFORM.LOWER_THRES
        self.pts_num = cfg.MODEL.DEFORM.PTS_NUM
        self.use_local_chamfer = cfg.MODEL.DEFORM.USE_LOCAL_CHAMFER
        self.chamfer_weight = cfg.MODEL.DEFORM.CHAMFER_LOSS_WEIGHT
        self.use_adaptive_tpl = cfg.MODEL.DEFORM.ADPTPL

        f2f_res_layers = [] # Residual feature to feature blocks
        f2v_layers = [] # Features to vertices
        lns_layers = [] # Learnt neighborhood sampling
        intra_relation_layers = [] # Intra relation layers
    
        for i in range(self.num_steps):
            irl_blocks_classes = []
            f2f_res_layers_classes = []
            f2v_layers_classes = []
            for _ in range(self.num_classes):
                res_blocks = []
                irl_blocks = []
                if i != 0:
                    input_dim = 24
                else:
                    input_dim = 24
                irl_blocks.append(
                    TransformerEnc6(input_dim)    
                )
                if (i != 0):
                    input_dim = 48
                else:
                    input_dim = 24
                
                res_blocks.append(nn.Linear(input_dim, 24))

                irl_blocks = nn.ModuleList(irl_blocks)
                irl_blocks_classes.append(irl_blocks)
                
                res_blocks = nn.ModuleList(res_blocks)
                f2f_res_layers_classes.append(res_blocks)
                f2v_layers_classes.append(nn.ModuleList([
                    f2v(),
                ])
                )
            
            irl_blocks_classes = nn.ModuleList(irl_blocks_classes)
            f2f_res_layers_classes = nn.ModuleList(f2f_res_layers_classes)
            f2v_layers_classes = nn.ModuleList(f2v_layers_classes)
            intra_relation_layers.append(irl_blocks_classes)
            f2f_res_layers.append(f2f_res_layers_classes)
            f2v_layers.append(f2v_layers_classes)

            # Feature sampling layer
            lns_layers.append(BasicFeatureSampling())
        
        self.intra_relation_layers = nn.ModuleList(intra_relation_layers)
        self.f2f_res_layers = nn.ModuleList(f2f_res_layers)
        self.f2v_layers = nn.ModuleList(f2v_layers)
        self.lns_layers = nn.ModuleList(lns_layers)
        self.features_fuse = nn.Linear(24 * 1, 24) 
        self.pos_embed = nn.Linear(3, 24)
        self.relative_pos_embed = nn.Linear(1, 24)


    @property
    def device(self):
        return list(self.parameters())[0].device
        
    def forward(self, batched_inputs, features: List[torch.FloatTensor], pad_img_shape, gt_seg, pred_seg):
        """
        Initialize the tube cline points.
        Deform the vertices through image features.

        Args:
            features: the downsampled feature map, shape (B, C, H, D, W)
        Returns:
            pred_mesh: the deformed mesh
        """
        pad_img_shape = torch.tensor(pad_img_shape[2:], device=features[0].device)
        segmentation = pred_seg
        estimate_dice = None
        if self.training:
            estimate_dice = get_dice_coeff((segmentation[:, 0] > 0.5).float(), gt_seg[:, 0]) 
        
        if self.use_adaptive_tpl:
            if self.training:
                if estimate_dice > 0.65:
                    initial_cline = self.initial_cline(from_seg = True, seg = segmentation,
                                                    img_shape = pad_img_shape)
                else:
                    initial_cline = self.initial_cline(from_seg = True, seg = gt_seg.float(),
                                                    img_shape = pad_img_shape)
            else:
                initial_cline = self.initial_cline(from_seg = True, seg = segmentation,
                                                    img_shape = pad_img_shape)
        else:
            initial_cline = self.initial_cline()
        
        
        preds = [initial_cline]
        preds_before_deform = [None]  # None: aligned to pred
        verts_feats_all_classes = [None] * self.num_classes

        for i, (f2f_res_layer_classes, f2v_layer_classes, lns_layer, irl_layer) in enumerate(zip(self.f2f_res_layers, 
                self.f2v_layers, self.lns_layers, self.intra_relation_layers)):
            cline_i = preds[i]
            verts_i = cline_i['verts']
            edges_i = cline_i['edges']
            verts_old = list()  # verts of next layer before deformation for all cls
            edges_old = list()  # faces of next layer before deformation for all cls
            verts_new = list()  # verts of next layer for all cls
            edges_new = list()  # faces of next layer for all cls

            for cls_idx in range(self.num_classes):
                f2f_res_layer = f2f_res_layer_classes[cls_idx]
                f2v_layer = f2v_layer_classes[cls_idx]
                irl_layer = irl_layer[cls_idx]
                
                verts = verts_i[cls_idx][None]
                edges = edges_i[cls_idx].clone()  # reinitialize edges
                verts_feats = verts_feats_all_classes[cls_idx]
                if i != 0:
                    last_verts_feats = verts_feats.clone()
                
                verts_old.append(verts.squeeze(0))
                edges_old.append(edges.squeeze(0))

                grid_verts = verts.flip(dims=[-1])
                verts_feats = []
                for feature in features:
                    verts_feats.append(lns_layer(feature, grid_verts, pad_img_shape))  # voxel feature sampling
                verts_feats = torch.cat(verts_feats, dim=2)
                verts_feats = self.features_fuse(verts_feats)
                if i != 0:
                    verts_feats = torch.cat([verts_feats, last_verts_feats[None]], dim=2)
                verts_feats = verts_feats.squeeze(0) # L, C
                
                verts_feats =  f2f_res_layer[0] (verts_feats)
                pos = self.pos_embed(grid_verts) # 1 L 24
                relative_pos = torch.arange(grid_verts.shape[1], device=grid_verts.device).float().unsqueeze(0).unsqueeze(-1)
                
                relative_pos_embed = self.relative_pos_embed(relative_pos)
                verts_feats = irl_layer[0](verts_feats.unsqueeze(0), 
                                        src_key_padding_mask=None, pos=pos + relative_pos_embed)
                
                verts_feats = verts_feats.squeeze(0) # L, C
                delta_verts = f2v_layer[0](verts_feats.unsqueeze(0)) * 0.2
                grid_verts = grid_verts + torch.tanh(delta_verts)
                verts = grid_verts.flip(dims=[-1])

                verts_new.append(verts.squeeze(0))
                edges_new.append(edges.squeeze(0))
                verts_feats_all_classes[cls_idx] = verts_feats  # update verts_feats
            
            cline_i_old = dict(verts=verts_old, edges=edges_old)
            preds_before_deform.append(cline_i_old)
            cline_i = dict(verts=verts_new, edges=edges_new)
            preds.append(cline_i)
        
        if self.training:
            gt_cline = batched_inputs[0]['cline'][0]
            assert len(gt_cline.shape) == 3
            gt_verts = torch.nonzero(gt_cline > 0.5).float()
            loss = self.loss(gt_verts, preds, gt_seg, pad_img_shape)

        ret = dict(pred_cline=dict())
        verts = preds[-1]['verts'][0]
        ret['pred_cline']['verts'] = unnormalize_vertices(verts, pad_img_shape)
        edges = preds[-1]['edges'][0]
        ret['pred_cline']['edges'] = edges
        ret['initial_cline'] = initial_cline['verts'][0]

        if self.training:
            return ret, loss
        else:
            return ret

    def initial_cline(self, from_seg=False, seg = None, img_shape = None):
        pts_num = self.pts_num
        if not from_seg:
            z_min, z_max = -0.8, 0.8
            pts_z = np.arange(z_min, z_max, (z_max-z_min)/pts_num)
            pts_z = - pts_z
            pts_x = np.zeros_like(pts_z)
            pts_y = np.zeros_like(pts_z)
            pts = np.stack([pts_y, pts_x, pts_z], axis=1)
            pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
        else:
            seg = (seg[0][0] > 0.5).float()
            cline = skeletonize_3d(seg.cpu().numpy())
            cline_pts = np.stack(np.where(cline), axis=1)
            cline_pts = cline_pts.astype(np.float64) * 1.
            if len(cline_pts) > self.lower_thres and len(cline_pts) < 600:
                cline_sample, _ = build_tree(cline_pts, n_p = pts_num, thres = 30)
                dist = np.sqrt(((cline_sample[1:] - cline_sample[:-1])**2).sum(axis=-1))
                all_dist = np.sum(dist)
                cumsum_dist = np.cumsum(dist)
                control_pts_num = self.control_pts_num
                remain_pts_num = (pts_num - control_pts_num) // (control_pts_num - 1)
                control_dist = all_dist / (control_pts_num - 1)
                control_pts = [cline_sample[0]]
                for i in range(control_pts_num - 2):
                    index = np.argmin(np.abs(cumsum_dist - control_dist * (i + 1)))
                    control_pts.append(cline_sample[index])
                control_pts.append(cline_sample[-1])
                control_pts = np.stack(control_pts, axis=0)
                # Use 2-order polynomial interpolation to get the control points
                cline_sample_ = BezierInterpo(control_pts, pts_num)
                pts = torch.tensor(cline_sample_, dtype=torch.float32, device=self.device)
                pts = normalize_vertices(pts, img_shape).to(self.device)
            else:
                z_min, z_max = -0.8, 0.8
                pts_z = np.arange(z_min, z_max, (z_max-z_min)/pts_num)
                pts_z = - pts_z
                pts_x = np.zeros_like(pts_z)
                pts_y = np.zeros_like(pts_z)
                pts = np.stack([pts_y, pts_x, pts_z], axis=1)
                pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
            
        pre = np.arange(pts_num - 1)
        fol = np.arange(pts_num)[1:]
        edges = np.stack([pre, fol], axis=1)
        edges = torch.tensor(edges, dtype=torch.int64, device=self.device)

        cline = dict(verts=[pts], edges=[edges])
        return cline
    
    def sample_verts(self, verts, num=20):
        index = np.random.choice(verts.shape[0], num, replace=True).tolist()
        return verts[index]

    def loss(self, gt_verts, preds, seg_targets, img_shape):
        losses = dict()
        gt_verts = normalize_vertices(gt_verts, img_shape).to(self.device)
        n_p = np.random.randint(60, 80)
        gt_verts_sample, _ = build_tree(gt_verts.cpu().numpy(), n_p=n_p, use_as_loss=True)
        gt_verts_sample = torch.tensor(gt_verts_sample, dtype=torch.float32, device=self.device)
        
        weights = [0, .05, .6, .95, 1.0]

        for i, pred_cline in enumerate(preds[2:], start=2):  # ignore the initial mesh and the first deformed mesh
            pred_verts = pred_cline['verts'][0]
            pred_edges = pred_cline['edges'][0]

            loss_chamfer = chamfer_distance(pred_verts[None], gt_verts[None])[0]
            patch_size = 0.4
            rand_gt_verts = gt_verts_sample
            rand_patch = torch.cat([rand_gt_verts - patch_size / 2, rand_gt_verts + patch_size / 2], dim=1) # rand_num, 6
            weights_local = torch.zeros_like(gt_verts_sample[:, 0]) + 1.2
            weights_local[int(len(rand_gt_verts)*0.2) : 
                int(len(rand_gt_verts)*0.7)] = 1.
            
            if self.use_local_chamfer:
                loss_local_chamfer = 0.
                valid_num = 0
                for j in range(len(rand_patch)):
                    patch = rand_patch[j]
                    pred_mask = ((pred_verts > patch[:3]) & (pred_verts < patch[3:])).all(dim=1)
                    gt_mask = ((gt_verts > patch[:3]) & (gt_verts < patch[3:])).all(dim=1)
                    if pred_mask.sum() < 15 or gt_mask.sum() < 10:
                        continue
                    valid_num += weights_local[j]
                    loss_local_chamfer += chamfer_distance(pred_verts[pred_mask][None], gt_verts[gt_mask][None])[0] \
                        * weights_local[j]
                
                if valid_num > 0:
                    loss_local_chamfer /= valid_num
                else:
                    loss_local_chamfer = loss_chamfer * 0.0
            else:
                loss_local_chamfer = loss_chamfer
            
            loss_edge = 0.
            target_length = 0.
            for edge in pred_edges:
                v0 = pred_verts[edge[0]][None]
                v1 = pred_verts[edge[1]][None]
                loss_edge += ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
            loss_edge = loss_edge / len(pred_edges)

            losses['loss_local_chamfer_' + str(i)] = loss_local_chamfer * 30. * weights[i] * self.chamfer_weight
            losses['loss_edge_' + str(i)] = loss_edge * 30. * 2 * weights[i] * \
                self.loss_edge_weight
            loss_sdf = self.sdf_loss(pred_verts, seg_targets[0][0], img_shape) * self.sdf_loss_weight
            losses['loss_sdf_' + str(i)] = loss_sdf * 0.5 * weights[i]

        
        return losses

    def sdf_loss(self, pred_verts, gt_seg, img_shape):
        # pred_verts: N, 3    original coord
        # gt_seg: H, W, D
        assert len(gt_seg.shape) == 3
        if gt_seg.sum() < 5:
            return pred_verts.sum() * 0.0
        
        pred_verts_ori = unnormalize_vertices(pred_verts, img_shape)

        sdf_signs = []
        for verts in pred_verts_ori:
            verts_ = verts.detach().cpu().numpy().astype(np.int32)
            verts_ = verts_.clip(min=0, max=img_shape.cpu().numpy() - 1)
            sdf_signs.append(gt_seg[verts_[0], verts_[1], verts_[2]])
        sdf_signs = torch.tensor(sdf_signs, dtype=torch.float32, device=pred_verts.device)
        sdf_signs[sdf_signs==0] = -1
        sdf_signs = - sdf_signs

        np_kernel = np.array([
                                [[0,0,0],[0,1,0],[0,0,0]],
                                [[0,1,0],[1,1,1],[0,1,0]],
                                [[0,0,0],[0,1,0],[0,0,0]],
                            ])

        torch_kernel = torch.tensor(np_kernel, dtype=torch.float32, device=pred_verts.device)
        temp = F.conv3d(gt_seg[None][None], torch_kernel[None, None], padding=1)[0][0]
        gt_surface = (temp < 7) & (gt_seg > 0)

        gt_pts = torch.nonzero(gt_surface > 0.).float()
        gt_pts = normalize_vertices(gt_pts, img_shape).to(pred_verts.device)
        pred_nn = knn_points(pred_verts[None, ...], gt_pts[None, ...], K=1)
        pred_k1_dist = pred_nn.dists.sqrt().squeeze(0).squeeze(-1)
        pred_sdf = pred_k1_dist * sdf_signs
        loss_sdf = pred_sdf.mean()

        return loss_sdf


def build_tree(pts, n_p=300, thres=0.12):
    G = nx.Graph()
    arr1, arr2 = pts.reshape((-1, 1, 3)), pts.reshape((1, -1, 3))
    dist_1 = np.sqrt(((arr1 - arr2)**2).sum(axis=-1))
    adj_bin = dist_1 < thres
    G = nx.from_numpy_array(adj_bin * dist_1)
    tree = nx.minimum_spanning_tree(G)
    length_connected_components = [len(elem) for elem in nx.connected_components(tree)]
    if len(length_connected_components) > 1:
        max_cc = max(nx.connected_components(tree), key=len)
        tree = tree.subgraph(max_cc)
    path_dict = nx.shortest_path_length(tree, list(tree.nodes)[0])
    sorted_paths = [(-v, k) for k, v in path_dict.items()]
    sorted_paths.sort()
    most_far_id1 = sorted_paths[0][1]

    path_dict = nx.shortest_path_length(tree, most_far_id1)
    sorted_paths = [(-v, k) for k, v in path_dict.items()]
    sorted_paths.sort()
    most_far_id2 = sorted_paths[0][1]
    
    if pts[most_far_id1, 2] < pts[most_far_id2, 2]:
        root = most_far_id2
    else:
        root = most_far_id1
    tree_ = nx.dfs_tree(tree, root)
    path_list = []
    find_all_path(tree_, root, [], path_list)
    longest_path = max(path_list, key=lambda x: len(x))
    l_lp = len(longest_path)
    
    samples = np.arange(0, l_lp, l_lp / n_p)[:n_p]
    samples = np.floor(samples).astype(np.int32)
    samples_ = pts[np.array(longest_path)[samples]]
    return samples_, []

def find_all_path(tree, cu_id, prefix_nodes, path_list):
    if len(tree[cu_id]) == 0:
        path_list.append(prefix_nodes + [cu_id])
        return
    elif len(tree[cu_id]) == 1:
        next_ids = [elem for elem in tree[cu_id]]
        find_all_path(tree, next_ids[0], prefix_nodes + [cu_id], path_list)
        return
    else:
        next_ids = [elem for elem in tree[cu_id]]
        for next_id in next_ids:
            find_all_path(tree, next_id, prefix_nodes + [cu_id], path_list)
        return