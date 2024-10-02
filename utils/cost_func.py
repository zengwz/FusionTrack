import numpy as np
from .utils import expand_dims, mask_between_boxes, logical_or_mask, loop_inter, loop_convex, union_vectors
from typing import Tuple
from shapely.geometry import Polygon
import math
import copy

def xyz_dist(all_detections, all_predictions, motion_model):
    # all_detections: np.array
    # all_predictions: np.array
    det_len = len(all_detections)
    pred_len = len(all_predictions)

    # caculate dist cost
    all_detections = all_detections.reshape((det_len,1,-1))
    all_predictions = all_predictions.reshape((1,pred_len,-1))

    all_detections = np.tile(all_detections,(1,pred_len,1))
    all_predictions = np.tile(all_predictions,(det_len,1,1))

    cost_dist = (all_detections[...,0:3] - all_predictions[...,0:3])**2
    cost_dist = cost_dist.sum(-1)
    # cost_dist = cost_dist * all_predictions[...,-1]
    return cost_dist

def xyzlwh_dist(all_detections, all_predictions, motion_model):
    # all_detections: np.array
    # all_predictions: np.array
    det_len = len(all_detections)
    pred_len = len(all_predictions)

    # caculate dist cost
    all_detections = all_detections.reshape((det_len,1,-1))
    all_predictions = all_predictions.reshape((1,pred_len,-1))

    all_detections = np.tile(all_detections,(1,pred_len,1))
    all_predictions = np.tile(all_predictions,(det_len,1,1))

    xyz_dist = (all_detections[...,0:3] - all_predictions[...,0:3])**2
    xyz_dist = np.sqrt(xyz_dist.sum(-1))

    ids = get_lwhyaw_ids(motion_model)
    l_ids = ids[0]
    lwh_dist = (all_detections[...,l_ids:l_ids + 3] - all_predictions[...,l_ids:l_ids + 3])**2
    lwh_dist = np.sqrt(lwh_dist.sum(-1))

    xyz_ratio = 0.4
    cost_dist = xyz_ratio * xyz_dist + (1.0 - xyz_ratio) * lwh_dist

    # cost_dist = cost_dist * all_predictions[...,-1]
    return cost_dist

def xyzlwhyaw_dist(all_detections, all_predictions, motion_model):
    # all_detections: np.array
    # all_predictions: np.array
    det_len = len(all_detections)
    pred_len = len(all_predictions)

    # caculate dist cost
    all_detections = all_detections.reshape((det_len,1,-1))
    all_predictions = all_predictions.reshape((1,pred_len,-1))

    all_detections = np.tile(all_detections,(1,pred_len,1))
    all_predictions = np.tile(all_predictions,(det_len,1,1))

    xyz_dist = (all_detections[...,0:3] - all_predictions[...,0:3])**2
    xyz_dist = np.sqrt(xyz_dist.sum(-1))

    ids = get_lwhyaw_ids(motion_model)
    l_ids = ids[0]
    lwh_dist = (all_detections[...,l_ids:l_ids + 3] - all_predictions[...,l_ids:l_ids + 3])**2
    lwh_dist = np.sqrt(lwh_dist.sum(-1))

    yaw_ids = ids[1]
    yaw_dist = 2 - np.cos(abs(all_detections[..., yaw_ids] - all_predictions[..., yaw_ids]))

    xyz_ratio = 0.4
    cost_dist = (xyz_ratio * xyz_dist + (1.0 - xyz_ratio) * lwh_dist) * yaw_dist

    # cost_dist = cost_dist * all_predictions[...,-1]
    return cost_dist

def get_lwhyaw_ids(motion_model):
    ids_dict = {'CV': [6, 9],
                'CA': [9, 12],
                'CTRV': [3, 7],
                'CTRA': [3, 8]}
    assert motion_model in ids_dict
    return ids_dict[motion_model]

def area_compute(boxes_a: np.array, boxes_b: np.array, motion_model: str, mode: str) -> Tuple[np.array, np.array]:
    """
    half-parallel implementation of 3d giou. why half? convexhull and intersection are still serial
    'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    'np_dets_bottom_corners': np.array, [det_num, 4, 2]
    :param boxes_a: dict, a collection of NuscBox info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :param boxes_b: dict, a collection of NuscBox info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :return: [np.array, np.array], 3d giou/bev giou between two boxes collections
    """
    # load info
    boxes_a, boxes_b = union_vectors(boxes_a, motion_model), union_vectors(boxes_b, motion_model)
    infos_a, infos_b = get_infos(copy.deepcopy(boxes_a)), get_infos(copy.deepcopy(boxes_b))  # [box_num, 14]
    bcs_a, bcs_b = get_bcs(copy.deepcopy(boxes_a)), get_bcs(copy.deepcopy(boxes_b))  # [box_num, 4, 2]

    # corner case, 1d array(collection only has one box) to 2d array
    # if infos_a.ndim == 1: infos_a, bcs_a = infos_a[None, :], bcs_a[None, :]
    # if infos_b.ndim == 1: infos_b, bcs_b = infos_b[None, :], bcs_b[None, :]
    # assert infos_a.shape[1] == 14 and infos_b.shape[1] == 14, "dim must be 14"

    # mask matrix, True denotes different(invalid), False denotes same(valid)
    bool_mask, seq_mask = mask_between_boxes(infos_a[:, -1], infos_b[:, -1])
    bool_mask, seq_mask = logical_or_mask(bool_mask, seq_mask, boxes_a, boxes_b)

    # process bottom corners, size and center for parallel computing
    rep_bcs_a, rep_bcs_b = expand_dims(bcs_a, len(bcs_b), 1), expand_dims(bcs_b, len(bcs_a), 0)  # [a_num, b_num, 4, 2]
    wlh_a, wlh_b = expand_dims(infos_a[:, 3:6], len(infos_b), 1), expand_dims(infos_b[:, 3:6], len(infos_a), 0)
    za, zb = expand_dims(infos_a[:, 2], len(infos_b), 1), expand_dims(infos_b[:, 2], len(infos_a), 0)  # [a_num, b_num]
    wa, la, ha = wlh_a[:, :, 0], wlh_a[:, :, 1], wlh_a[:, :, 2]
    wb, lb, hb = wlh_b[:, :, 0], wlh_b[:, :, 1], wlh_b[:, :, 2]

    # polygons
    polys_a, polys_b = [Polygon(bc_a) for bc_a in bcs_a], [Polygon(bc_b) for bc_b in bcs_b]

    # overlap and union height
    ohs = np.maximum(np.zeros_like(ha), np.minimum(za + ha / 2, zb + hb / 2) - np.maximum(za - ha / 2, zb - hb / 2))
    uhs = np.maximum((za + ha / 2), (zb + hb / 2)) - np.minimum((zb - hb / 2), (za - ha / 2))

    # overlap and union area/volume
    inter_areas = loop_inter(polys_a, polys_b, bool_mask)
    inter_volumes = inter_areas * ohs
    union_areas, union_volumes = wa * la + wb * lb - inter_areas, wa * la * ha + wb * lb * hb - inter_volumes

    # convexhull area/volume
    if 'giou' in mode:
        convex_areas = loop_convex(rep_bcs_a, rep_bcs_b, seq_mask)
        convex_volumes = convex_areas * uhs

    if mode == 'iou_3d':
        ret = inter_volumes / union_volumes
    elif mode == 'iou_bev':
        ret = inter_areas / union_areas
    elif mode == 'giou_3d':
        ret = inter_volumes / union_volumes - (convex_volumes - union_volumes) / convex_volumes
    elif mode == 'giou_bev':
        ret = inter_areas / union_areas - (convex_areas - union_areas) / convex_areas
    else:
        raise NotImplementedError
    
    ret[bool_mask] = -np.inf

    return -ret

def iou_3d(boxes_a: np.array, boxes_b: np.array, motion_model: str):
    return area_compute(boxes_a, boxes_b, motion_model, 'iou_3d')

def iou_bev(boxes_a: np.array, boxes_b: np.array, motion_model: str):
    return area_compute(boxes_a, boxes_b, motion_model, 'iou_bev')

def giou_3d(boxes_a: np.array, boxes_b: np.array, motion_model: str):
    return area_compute(boxes_a, boxes_b, motion_model, 'giou_3d')

def giou_bev(boxes_a: np.array, boxes_b: np.array, motion_model: str):
    return area_compute(boxes_a, boxes_b, motion_model, 'giou_bev')

def get_infos(inner_vectors):
    # yaw is not used in this infos
    infos = np.zeros((inner_vectors.shape[0], 14), dtype=np.float32)
    infos[:, :3] = inner_vectors[:, :3]
    infos[:, 3] = inner_vectors[:, 4]
    infos[:, 4] = inner_vectors[:, 3]
    infos[:, 5] = inner_vectors[:, 5]
    return infos

def get_bcs(inner_vectors):
    pts_all = np.zeros((0, 4, 3), dtype=np.float32)
    for inner_vector in inner_vectors:
        x, y, z, l, w, h, yaw = inner_vector
        pt1 = [l / 2, w / 2, 1]
        pt2 = [l / 2, - w / 2, 1]
        pt3 = [- l / 2, - w / 2, 1]
        pt4 = [- l / 2, w / 2, 1]
        pts = np.array([[pt1, pt2, pt3, pt4]])
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), x],
                                    [np.sin(yaw), np.cos(yaw), y],
                                    [0, 0, 1]])
        pts = np.matmul(pts, rotation_matrix.T)
        pts_all = np.concatenate((pts_all, pts), axis=0)  # (N, 4, 3)
    return pts_all[:, :, :2]