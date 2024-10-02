from .trajectory import CV, CA, CTRV, CTRA, without_motion
from .box_op import *
import numpy as np
import copy
import lap
from dataset.kitti_data_base import velo_to_cam
import cv2
import os
from utils.cost_func import *
from utils.utils import union_vectors
from utils.cost_func import area_compute

class Tracker3D:
    def __init__(self,
                 box_type='Kitti',
                 config = None):
        """
        initialize the the 3D tracker
        Args:
            box_type: str, box type, available box type "OpenPCDet", "Kitti", "Waymo"
        """
        self.config = config
        self.motion_model = config.motion_model
        self.match_method_name = f"{self.config.match_method}_match_{self.config.cost_input}{self.config.name_profix}"
        self.motion_cost = config.motion_cost

        self.all_ids = None
        self.cost_feat = None
        self.cost_dist = None
        self.cost_xyz = None
        self.ids = []

        self.current_timestamp = None
        self.current_pose = None
        self.current_bbs = None
        self.current_features = None
        self.box_type = box_type

        self.label_seed = 0

        self.active_trajectories = {}
        self.dead_trajectories = {}

    def tracking(self,
                 dets_3d=None,
                 dets_2d=None,
                 features=None,
                 scores=None,
                 pose=None,
                 P2=None,
                 V2C=None,
                 seq_frame_name=None,
                 timestamp=None,
                 ):
        """
        tracking the objects at the given timestamp
        Args:
            bbs: array(N,7) or array(N, 7*k), 3D bounding boxes or 3D tracklets
                for tracklets, the boxes should be organized to [[box_t; box_t-1; box_t-2;...],...]
            features: array(N,k), the features of boxes or tracklets
            scores: array(N,), the detection score of boxes or tracklets
            pose: array(4,4), pose matrix to global scene
            timestamp: int, current timestamp, note that the timestamp should be consecutive

        Returns:
            bbs: array(M,7), the tracked bbs
            ids: array(M,), the assigned IDs for bbs
        """
        self.current_bbs = dets_3d
        self.P2 = P2
        self.V2C =V2C
        self.seq_frame_name = seq_frame_name
        self.current_features = features
        self.current_scores = scores
        self.current_pose = pose
        self.pose_inv = np.mat(self.current_pose).I
        self.current_timestamp = timestamp
        if dets_2d is not None:
            self.current_det_2d = dets_2d
        self.current_ori_bbs = copy.deepcopy(dets_3d)

        self.trajectores_prediction()

        if self.current_bbs is None:
            raise NotImplementedError
        
        self.ids, self.all_ids = [], []


        if len(self.current_bbs) == 0:
            if len(self.current_det_2d) == 0:
                return np.zeros(shape=(0,7)),np.zeros(shape=(0))
            else:
                if self.config.cost_input == "fuse" and \
                    ("tra_recover" in self.config.cost_input or self.config.cost_input == ""):
                    for key in self.active_trajectories.keys():
                        self.all_ids.append(key)
                    match_tra_ids = [False] * len(self.active_trajectories)
                    _ = self.update_unmatch_tra_by_det2d(match_tra_ids)
                    bbs, ids = self.trajectories_update_init()
                    return np.array(bbs), np.array(ids)
                else:
                    return np.zeros(shape=(0,7)),np.zeros(shape=(0))
        else:
            # self.current_bbs = convert_bbs_type(self.current_bbs,self.box_type)
            self.current_bbs = register_bbs(self.current_bbs, self.current_pose)

            self.association()
            bbs, ids = self.trajectories_update_init()
            return np.array(bbs), np.array(ids)

    def sigmoid(self, x):
        return np.array([1 / (1 + np.exp(-x_)) for x_ in x])

    def draw_img(self, bboxes, draw_path):
        img_root = '/home/zwz/FusionTrack/data/kitti_sfd_seguv_twise/training/image_02/'
        img = os.path.join(img_root, self.seq_frame_name[0], self.seq_frame_name[1] + '.png')
        img = cv2.imread(img)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        save_path = os.path.join(draw_path, self.seq_frame_name[0])
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, self.seq_frame_name[1] + '.png')
        cv2.imwrite(save_path, img)

    def trajectores_prediction(self):
        """
        predict the possible state of each active trajectories, if the trajectory is not updated for a while,
        it will be deleted from the active trajectories set, and moved to dead trajectories set
        Returns:

        """
        if len(self.active_trajectories) == 0 :
            return
        else:
            dead_track_id = []

            for key in self.active_trajectories.keys():
                if self.active_trajectories[key].consecutive_missed_num>=self.config.max_prediction_num:
                    dead_track_id.append(key)
                    continue
                self.active_trajectories[key].state_prediction(self.current_timestamp)

            for id in dead_track_id:
                tra = self.active_trajectories.pop(id)
                self.dead_trajectories[id]=tra

    def compute_cost_map(self):
        """
        compute the cost map between detections and predictions
        Returns:
              cost, array(N,M), where N is the number of detections, M is the number of active trajectories
              all_ids, list(M,), the corresponding IDs of active trajectories
        """
        all_ids = []

        all_predictions = []
        all_detections = []
        
        all_trajectory_feats = []
        all_detections_feats = []

        all_trajectory_feats_dict = []
        all_detections_feats_dict = []

        for key in self.active_trajectories.keys():
            all_ids.append(key)
            state = np.array(self.active_trajectories[key].trajectory[self.current_timestamp].predicted_state)
            state = state.reshape(-1)

            pred_score = np.array([self.active_trajectories[key].trajectory[self.current_timestamp].prediction_score])

            state = np.concatenate([state,pred_score])
            all_predictions.append(state)

            trajectory_feats = self.active_trajectories[key].trajectory[self.current_timestamp].features
            all_trajectory_feats.append(trajectory_feats)

            trajectory_feats_dict = self.active_trajectories[key].trajectory[self.current_timestamp].features_dict
            all_trajectory_feats_dict.append(trajectory_feats_dict)

        for i in range(len(self.current_bbs)):
            box = self.current_bbs[i]
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]
            label=1
            new_tra = globals()[self.motion_model](init_bb=box,
                                                   init_features=features,
                                                   init_score=score,
                                                   init_timestamp=self.current_timestamp,
                                                   label=label,
                                                   config = self.config)

            state = new_tra.trajectory[self.current_timestamp].predicted_state
            state = np.array(state).reshape(-1)
            all_detections.append(state)

            detection_feats = new_tra.trajectory[self.current_timestamp].features
            all_detections_feats.append(detection_feats)
            
            detection_feats_dict = new_tra.trajectory[self.current_timestamp].features_dict
            all_detections_feats_dict.append(detection_feats_dict)

        all_detections = np.array(all_detections)
        all_predictions = np.array(all_predictions)

        cost_dist = globals()[self.motion_cost](all_detections, all_predictions, self.config.motion_model)
        cost_xyz = xyz_dist(all_detections, all_predictions, self.config.motion_model)
        
        cost_feat = self.feats_cost_compute(all_detections_feats, all_trajectory_feats)

        return cost_dist, cost_feat, all_ids, cost_xyz

    def feats_cost_compute(self, all_detections_feats, all_trajectory_feats):
        all_detections_feats = np.array(all_detections_feats)
        all_trajectory_feats = np.array(all_trajectory_feats)
        cost_feats = all_detections_feats @ all_trajectory_feats.T
        cost_feats = 1.0 - self.softmax(cost_feats, dim=1)
        return cost_feats

    def softmax(self, x, dim):
        # subtract the maximum value to each row to avoid exponential overflow
        x -= np.max(x, axis=dim, keepdims=True)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        return softmax_x    

    def association(self):
        """
        greedy assign the IDs for detected state based on the cost map
        Returns:
            ids, list(N,), assigned IDs for boxes, where N is the input boxes number
        """
        self.ids = []
        if len(self.active_trajectories) == 0:
            if self.config.cost_input == "fuse" and \
                (self.config.name_profix == "" or "det_filter" in self.config.name_profix):
                if len(self.current_det_2d) == 0:
                    self.current_bbs = []
                    return []
                bboxes_3d = copy.deepcopy(self.current_bbs)
                self.clear_current_bbs(bboxes_3d)

            for i in range(len(self.current_bbs)):
                self.ids.append(self.label_seed)
                self.label_seed += 1
        else:
            self.cost_dist, self.cost_feat, self.all_ids, self.cost_xyz = self.compute_cost_map()
            match_func = getattr(self, self.match_method_name)
            match_func()

    def compute_dist_ratio(self, tol=20.0):
        dist_ratio = np.linalg.norm(self.current_ori_bbs[:, :3], axis=1)
        return dist_ratio < tol

    def hungarian_match_dist(self):
        _, row, col = lap.lapjv(self.cost_dist, extend_cost=True, cost_limit=self.config.dist_thres)
        first_match_tra_ids = col != -1
        for i, det_match_ids in enumerate(row):
            if det_match_ids != -1:
                self.ids.append(self.all_ids[det_match_ids])
            else:
                self.ids.append(self.label_seed)
                self.label_seed += 1

    def hungarian_match_feat(self):
        _, row, col = lap.lapjv(self.cost_feat, extend_cost=True, cost_limit=self.config.feat_thres)
        first_match_tra_ids = col != -1
        for i, det_match_ids in enumerate(row):
            if det_match_ids != -1:
                self.ids.append(self.all_ids[det_match_ids])
            else:
                self.ids.append(self.label_seed)
                self.label_seed += 1

    def hungarian_match_fuse(self):
        mask_cost_xyz = self.cost_xyz > self.config.dist_gate
        self.cost_feat[mask_cost_xyz] = np.inf

        _, row, col = lap.lapjv(self.cost_dist, extend_cost=True, cost_limit=self.config.dist_thres)
        first_unmatch_ids = []
        first_match_tra_ids = col != -1
        for i, det_match_ids in enumerate(row):
            if det_match_ids != -1:
                self.ids.append(self.all_ids[det_match_ids])
                self.cost_feat[i, :] = np.inf
                self.cost_dist[i, :] = np.inf
            else:
                if len(self.current_det_2d) != 0:
                    copy_bbs = copy.deepcopy(self.current_bbs[i]).reshape((1, 7))
                    copy_bbs_2d = copy.deepcopy(self.current_det_2d)
                    iou = self.filter_by_det_2d(copy_bbs, copy_bbs_2d[:, :4])
                    if iou > self.config.det_filter_thres:
                        first_unmatch_ids.append(i)
                self.ids.append(-2)

        if first_unmatch_ids == []:
            if len(self.current_det_2d) != 0:
                _ = self.update_unmatch_tra_by_det2d(first_match_tra_ids)
            return
        
        # second match using features
        self.cost_feat = np.array([self.cost_feat[i] for i in first_unmatch_ids])
        self.cost_feat[:, first_match_tra_ids] = np.inf
    
        _, row, col = lap.lapjv(self.cost_feat, extend_cost=True, cost_limit=self.config.feat_thres)
        second_match_tra_ids = col != -1
        for first_unmatch_id, det_match_ids in zip(first_unmatch_ids, row):
            if det_match_ids != -1:
                self.ids[first_unmatch_id] = self.all_ids[det_match_ids]
            else:
                self.ids[first_unmatch_id] = self.label_seed
                self.label_seed += 1

        if len(self.current_det_2d) != 0:
            matched_tra_ids = second_match_tra_ids | first_match_tra_ids
            _ = self.update_unmatch_tra_by_det2d(matched_tra_ids)

    def update_unmatch_tra_by_det2d(self, matched_tra_ids):
        recover_ids = []
        copy_det_2d = copy.deepcopy(self.current_det_2d)
        tra_bbs_3d, tra_bbs_2d, predicted_state_list, vaild_tras = self.get_trajectory_bbs()
        ious = compute_overlap(copy.deepcopy(tra_bbs_2d), copy_det_2d[:, :4], 'iou')
        current_bbs = copy.deepcopy(self.current_bbs)
        ious_bev = area_compute(tra_bbs_3d, current_bbs, 'without_motion', 'iou_bev')
        for i in range(len(self.all_ids)):
            if not vaild_tras[i] or matched_tra_ids[i] or max(ious_bev[i]) > 0.01:
                continue
            if np.max(ious[i]) > self.config.tra_recover_thres:
                recover_ids.append(i)
                self.ids.append(self.all_ids[i])
                self.current_bbs = np.concatenate((self.current_bbs, \
                                                    predicted_state_list[i * 3 + 1]), axis=0)
                self.current_features = np.concatenate((self.current_features, \
                                                        predicted_state_list[i * 3].reshape((1, -1))), axis=0)
                # score = np.array(predicted_state_list[i * 3 + 2]).reshape(1, )
                fake_score = np.array([1.0])
                self.current_scores = np.concatenate((self.current_scores, fake_score), axis=0)
        return recover_ids

    def filter_by_det_2d(self, bbox_3d, bboxes_det_2d):
        bbs_2d = self.inv_register_bbox_and_convert_2d(bbox_3d)
        iou = compute_overlap(bbs_2d, bboxes_det_2d, 'iou')
        return np.max(iou, axis=1)[0]

    def clear_current_bbs(self, bboxes_3d):
        bboxes_2d = np.zeros((0, 4), dtype=np.float32)
        for bbox_3d in bboxes_3d:
            bbox_3d = bbox_3d.reshape((1, -1))
            bbox_2d = self.inv_register_bbox_and_convert_2d(bbox_3d)
            bboxes_2d = np.concatenate((bboxes_2d, bbox_2d), axis=0)
        ious = compute_overlap(bboxes_2d, self.current_det_2d[:, :4], 'iou')
        ious_max = np.max(ious, axis=1)
        mask_ids = ious_max > self.config.det_filter_thres
        self.current_bbs = self.current_bbs[mask_ids]
        self.current_features = self.current_features[mask_ids]
        self.current_scores = self.current_scores[mask_ids]
        self.current_ori_bbs = self.current_ori_bbs[mask_ids]

    def get_trajectory_bbs(self):
        """
        Returns: tra_bboxes_2d: array, shape(n, 4), [x1, y1, x2, y2]
                 predicted_state_list: list, [n * 2], [feats, predicted_state] stack n times
        """
        tra_bboxes_2d = np.zeros((0, 4), dtype=np.float32)
        tra_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
        predicted_state_list = []
        vaild_tras = []
        for key in self.active_trajectories.keys():
            # ignore tras miss too long 
            if self.active_trajectories[key].consecutive_missed_num >= 5:
                vaild_tras.append(False)
            else:
                vaild_tras.append(True)
            trajactory_predict_state = copy.deepcopy(self.active_trajectories[key].\
                                                     trajectory[self.current_timestamp].predicted_state)
            trajactory_predict_feats = copy.deepcopy(self.active_trajectories[key].\
                                                     trajectory[self.current_timestamp].features)
            predicted_state_list.append(trajactory_predict_feats)
            predicted_score = copy.deepcopy(self.active_trajectories[key].\
                                                     trajectory[self.current_timestamp].prediction_score)
            trajactory_predict_state = np.array(trajactory_predict_state).reshape((1, -1))
            box_template = union_vectors(trajactory_predict_state, self.motion_model)
            tra_bboxes_3d = np.concatenate((tra_bboxes_3d, box_template), axis=0)
            predicted_state_list.append(copy.deepcopy(box_template))
            predicted_state_list.append(predicted_score)
            tra_bbox = self.inv_register_bbox_and_convert_2d(box_template)
            tra_bboxes_2d = np.concatenate((tra_bboxes_2d, tra_bbox), axis=0)
        return tra_bboxes_3d, tra_bboxes_2d, predicted_state_list, vaild_tras

    def inv_register_bbox_and_convert_2d(self, bbox_3d):
        """
        Args: bbox_3d : array, shape(1, 7), one bbox
        Returns: bbox_2d: array, shape(1, 4), [x1, y1, x2, y2]
        """
        bbox_3d = register_bbs(bbox_3d, self.pose_inv)
        bbox_3d[:, 6] = -bbox_3d[:, 6] - np.pi / 2
        bbox_3d[:, 2] -= bbox_3d[:, 5] / 2
        bbox_3d[:,0:3] = velo_to_cam(bbox_3d[:,0:3], self.V2C)[:,0:3]
        bbox_3d = bbox_3d[0]
        bbox_2d = bb3d_2_bb2d(bbox_3d, self.P2)
        return bbox_2d

    def trajectories_update_init(self):
        """
        update a exiting trajectories based on the association results, or init a new trajectory
        Args:
            ids: list or array(N), the assigned ids for boxes
        """
        assert len(self.ids) == len(self.current_bbs)

        valid_bbs = []
        valid_ids = []

        for i in range(len(self.current_bbs)):
            label = self.ids[i]
            box = self.current_bbs[i]
            features = None
            if self.current_features is not None:
                features = self.current_features[i]
            score = self.current_scores[i]

            if label == -2:
                continue

            if label in self.active_trajectories.keys() and score > self.config.update_score:
                track = self.active_trajectories[label]
                track.state_update(
                     bb=box,
                     features=features,
                     score=score,
                     timestamp=self.current_timestamp)
                valid_bbs.append(box)
                valid_ids.append(label)
            elif score>self.config.init_score:
                new_tra = globals()[self.motion_model](init_bb=box,
                                                       init_features=features,
                                                       init_score=score,
                                                       init_timestamp=self.current_timestamp,
                                                       label=label,
                                                       config = self.config)
                self.active_trajectories[label] = new_tra
                valid_bbs.append(box)
                valid_ids.append(label)
            else:
                continue
        if len(valid_bbs) == 0:
            return np.zeros(shape=(0,7)), np.zeros(shape=(0))
        else:
            return np.array(valid_bbs), np.array(valid_ids)


    def format_convert(self):
        tra = {}
        for key in self.dead_trajectories.keys():
            track = self.dead_trajectories[key]
            tra[key] = track
        for key in self.active_trajectories.keys():
            track = self.active_trajectories[key]
            tra[key] = track
        return tra

def compute_overlap(bboxes1, bboxes2, mode):
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    lt = np.maximum(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
    rb = np.minimum(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    wh = np.clip(rb - lt, a_min=0., a_max=None)
    overlap = wh[..., 0] * wh[..., 1]
    union = area1[..., None] + area2[..., None, :] - overlap
    eps = np.full_like(union, 1e-6)
    union = np.maximum(union, eps)
    if mode == 'iou':
        return overlap / union

    enclosed_lt = np.minimum(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
    enclosed_br = np.maximum(bboxes1[..., :, None, 2:4], bboxes2[..., None, :, 2:4])
    enclosed_wh = np.clip(enclosed_br - enclosed_lt, a_min=0., a_max=None)
    enclosed_area = enclosed_wh[..., 0] * enclosed_wh[..., 1]

    enclosed_area = np.maximum(enclosed_area, eps)

    ious = overlap / union
    gious = ious - (enclosed_area - union) / enclosed_area
    return gious