import torch

def cline_unpool(verts, edges, verts_feats, thresh=0.02):
    # verts shape: (num_verts, 3)
    # edges shape: (num_edges, 2)
    # verts_feats shape: (num_verts, num_feats)
    if verts is None:
        return None, None
    new_verts = []
    new_edges = []
    new_verts_feats = []

    adj_verts = verts[edges] # (num_edges, 2, 3)
    adj_verts_feats = verts_feats[edges] # (num_edges, 2, num_feats)
    adj_dis = torch.norm(adj_verts[:, 0] - adj_verts[:, 1], dim=1) # (num_edges, )
    unpool_mask = adj_dis > thresh # (num_edges, )
    if unpool_mask.sum() == 0:
        return verts, edges, verts_feats

    add_num = 0
    for i, edge in enumerate(edges):
        if unpool_mask[i]:
            new_vert = torch.mean(adj_verts[i], dim=0) # (3, )
            new_verts += [new_vert]
            new_edge_1 = torch.tensor([edge[0], len(verts) + add_num]) # (2, )
            new_edge_2 = torch.tensor([len(verts) + add_num, edge[1]]) # (2, )
            new_edges += [new_edge_1, new_edge_2]
            new_verts_feat = torch.mean(adj_verts_feats[i], dim=0) # (num_feats, )
            new_verts_feats += [new_verts_feat] 
            add_num += 1
    new_verts = torch.stack(new_verts, dim=0) # (num_new_verts, 3)
    new_edges = torch.stack(new_edges, dim=0) # (num_new_edges, 2)
    new_verts_feats = torch.stack(new_verts_feats, dim=0) # (num_new_verts, num_feats)

    ret_verts = torch.cat([verts, new_verts], dim=0) # (num_verts + num_new_verts, 3)
    ret_edges = torch.cat([edges, new_edges.to(edges.device)], dim=0) # (num_edges + num_new_edges, 2)
    ret_verts_feats = torch.cat([verts_feats, new_verts_feats], dim=0) # (num_verts + num_new_verts, num_feats)

    return ret_verts, ret_edges, ret_verts_feats
