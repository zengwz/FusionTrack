from dataset.kitti_dataset import KittiTrackingDataset
from dataset.kitti_data_base import velo_to_cam
from tracker.tracker import Tracker3D
import time
import tqdm
import os
from tracker.config import cfg, cfg_from_yaml_file
from tracker.box_op import *
import numpy as np
import argparse
from utils.utils import union_vectors

from evaluation_HOTA.scripts.run_kitti import eval_kitti
import matplotlib.pyplot as plt

def track_one_seq(seq_id, config):

    """
    tracking one sequence
    Args:
        seq_id: int, the sequence id
        config: config
    Returns: dataset: KittiTrackingDataset
             tracker: Tracker3D
             all_time: float, all tracking time
             frame_num: int, num frames
    """
    root_load_path = f'data/3d_detections/{config.detector3d}/{config.train_or_test}'

    det_2d_path = f'data/2d_detections/{config.detector2d}/{config.train_or_test}'

    dataset_path = os.path.join(config.dataset_path, config.train_or_test)

    tracker = Tracker3D(box_type="Kitti", config = config)
    dataset = KittiTrackingDataset(dataset_path,
                                   root_load_path=root_load_path,
                                   det_2d_path=det_2d_path,
                                   seq_id=seq_id, 
                                   type=[config.tracking_type],
                                   SF_thres_2d=config.SF_thres_2d,
                                   SF_thres_3d=config.SF_thres_3d)

    all_time = 0
    frame_num = 0

    for i in range(len(dataset)):
        P2, V2C, pose, dets_3d, dets_2d, features, det_scores, seq_frame_name, _, _ = dataset[i]

        start = time.time()

        tracker.tracking(dets_3d=dets_3d,
                         dets_2d=dets_2d,
                         features=features,
                         scores=det_scores,
                         pose=pose,
                         P2=P2,
                         V2C=V2C,
                         seq_frame_name=seq_frame_name,
                         timestamp=i)
        end = time.time()
        all_time+=end-start
        frame_num+=1

    return dataset, tracker, all_time, frame_num

def convert_bbs_type(boxes, input_box_type):
    boxes = np.array(boxes)

    assert input_box_type in ["Kitti", "OpenPCDet", "Waymo"], 'unsupported input box type!'

    if input_box_type in ["OpenPCDet", "Waymo"]:
        return boxes

    if input_box_type == "Kitti":  # (h,w,l,x,y,z,yaw) -> (x,y,z,l,w,h,yaw)

        t_id = boxes.shape[1] // 7
        new_boxes = np.zeros(shape=boxes.shape)
        new_boxes[:, :] = boxes[:, :]
        for i in range(t_id):
            b_id = i * 7
            new_boxes[:, b_id + 0:b_id + 3] = boxes[:, b_id + 3:b_id + 6]
            new_boxes[:, b_id + 3] = boxes[:, b_id + 2]
            new_boxes[:, b_id + 4] = boxes[:, b_id + 1]
            new_boxes[:, b_id + 5] = boxes[:, b_id + 0]
            new_boxes[:, b_id + 6] = (np.pi - boxes[:, b_id + 6]) + np.pi / 2
            new_boxes[:, b_id + 2] += boxes[:, b_id + 0] / 2
        return new_boxes


def save_one_seq(dataset,
                 seq_id,
                 tracker,
                 config):
    """
    saving tracking results
    Args:
        dataset: KittiTrackingDataset, Iterable dataset object
        seq_id: int, sequence id
        tracker: Tracker3D
    """

    save_path = config.save_path
    tracking_type = config.tracking_type
    s =time.time()
    tracks = tracker.format_convert(config)
    proc_time = s-time.time()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = os.path.join(save_path,str(seq_id).zfill(4)+'.txt')
    global_xy_save_name = os.path.join(config.global_xy_save_path,str(seq_id).zfill(4)+'.txt')
    os.makedirs(os.path.dirname(global_xy_save_name), exist_ok=True)

    frame_first_dict = {}
    for ob_id in tracks.keys():
        track = tracks[ob_id]

        for frame_id in track.trajectory.keys():

            ob = track.trajectory[frame_id]
            if ob.updated_state is None:
                continue

            if frame_id in frame_first_dict.keys():
                frame_first_dict[frame_id][ob_id]=(np.array(ob.updated_state.T),ob.score)
            else:
                frame_first_dict[frame_id]={ob_id:(np.array(ob.updated_state.T),ob.score)}

    with open(save_name,'w+') as f1, open(global_xy_save_name, 'w+') as f2:
        for i in range(len(dataset)):
            P2, V2C, pose, _, _, _, _, _, _, _ = dataset[i]
            new_pose = np.mat(pose).I
            if i in frame_first_dict.keys():
                objects = frame_first_dict[i]

                for ob_id in objects.keys():
                    updated_state,score = objects[ob_id]

                    box_template = union_vectors(updated_state, config.motion_model)

                    print('%d %d %s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' % (i,ob_id,tracking_type,\
                        box_template[0][0],box_template[0][1],box_template[0][2],\
                        box_template[0][3], box_template[0][4], box_template[0][5], box_template[0][6],score),file = f2)

                    # global to lidar
                    box = register_bbs(box_template,new_pose)

                    # lidar to camera 
                    box[:, 6] = -(box[:, 6] + np.pi / 2) # rotate yaw from lidar to camera
                    box[:, 2] -= box[:, 5] / 2 # top_z to middle_z
                    box[:,0:3] = velo_to_cam(box[:,0:3],V2C)[:,0:3]

                    box = box[0]
                    # camera to pixel
                    box2d = bb3d_2_bb2d(box,P2)

                    print('%d %d %s -1 -1 -10 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                          % (i,ob_id,tracking_type,box2d[0][0],box2d[0][1],box2d[0][2],
                             box2d[0][3],box[5],box[4],box[3],box[0],box[1],box[2],box[6],score),file = f1)

    return proc_time
  
def tracking_val_seq(arg):

    yaml_file = arg.cfg_file

    config = cfg_from_yaml_file(yaml_file,cfg)

    print("\nconfig file:", yaml_file)
    print("data path: ", config.dataset_path)

    save_path = config.save_path

    os.makedirs(save_path, exist_ok=True)

    seq_list = config.tracking_seqs_val if config.train_or_test == 'training' else config.tracking_seqs_test

    print("tracking seqs: ", seq_list)

    all_time,frame_num = 0,0

    for id in tqdm.trange(len(seq_list)):
        seq_id = seq_list[id]
        dataset,tracker, this_time, this_num = track_one_seq(seq_id, config)
        proc_time = save_one_seq(dataset, seq_id, tracker, config)

        all_time+=this_time
        all_time+=proc_time
        frame_num+=this_num

    print("Tracking time: ", all_time)
    print("Tracking frames: ", frame_num)
    print("Tracking FPS:", frame_num/all_time)
    print("Tracking ms:", all_time/frame_num)

    eval_kitti()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="config/fusion_mot.yaml")
    args = parser.parse_args()
    tracking_val_seq(args)

