import numpy as np
from .object import Object
from utils.utils import warp_to_pi

class Basetra:
    def warpStateYawToPi(self, state: np.mat) -> np.mat:
        pass
    
    def warpMearureYawToPi(self, state: np.mat) -> np.mat:
        pass

    def yaw_recover(self, det_box, timestamp, min_numbers=2):
        yaw_history_list = []

        t = 0
        while (True):
            t += 1
            if (timestamp - t) not in self.trajectory:
                break
            if self.trajectory[timestamp - t].updated_state is None:
                continue 
            yaw_history_list.append(self.trajectory[timestamp - t].updated_state[self.yaw_ids, 0])
            if len(yaw_history_list) == min_numbers:
                break

        if len(yaw_history_list) < min_numbers:
            return 

        for i in range(len(yaw_history_list) - 1):
            res = (yaw_history_list[i] - yaw_history_list[i + 1])
            res = warp_to_pi(res)
            if res > (np.pi / 2) or res < (-np.pi / 2):
                return
        
        res = (yaw_history_list[0] - det_box[6, 0])
        res = warp_to_pi(res)
        if res > (np.pi / 2.) or res < (-np.pi / 2. ):
            det_box[6, 0] -= np.pi 
            self.warpMearureYawToPi(det_box)

class CV(Basetra):
    def __init__(self,
                 init_bb=None,
                 init_features=None,
                 init_score=None,
                 init_timestamp=None,
                 label=None,
                 config = None
                 ):
        """

        Args:
            init_bb: array(7) or array(7*k), 3d box or tracklet
            init_features: array(m), features of box or tracklet
            init_score: array(1) or float, score of detection
            init_timestamp: int, init timestamp
            label: int, unique ID for this trajectory
        """
        assert init_bb is not None

        self.init_bb = init_bb
        self.init_features = init_features
        self.init_score = init_score
        self.init_timestamp = init_timestamp
        self.label = label

        self.config = config
        self.feat_momentum = 0.9

        self.scanning_interval = 1. / self.config.LiDAR_scanning_frequency

        self.trajectory = {}

        # state vector：[x, y, z, xv, yv, zv, l, w, h, yaw]
        # measure vector：[x, y, z, l, w, h, yaw]
        self.state_dim = 10
        self.measure_dim = 7
        self.yaw_ids = 9

        self.init_parameters()
        self.init_trajectory()

        self.consecutive_missed_num = 0
        self.first_updated_timestamp = init_timestamp
        self.last_updated_timestamp = init_timestamp

    def __len__(self):
        return len(self.trajectory)

    def warpStateYawToPi(self, state: np.mat) -> np.mat:
        state[-1, 0] = warp_to_pi(state[-1, 0])
    
    def warpMearureYawToPi(self, state: np.mat) -> np.mat:
        state[-1, 0] = warp_to_pi(state[-1, 0])

    def init_trajectory(self):
        """
        first initialize the object state with the input boxes info,
        then initialize the trajectory with the initialized object.
        :return:
        """

        detected_state_template = np.zeros(shape=(self.measure_dim))

        update_covariance_template = self.init_P

        detected_state_template[:self.measure_dim] = self.init_bb[:self.measure_dim]

        detected_state_template = np.mat(detected_state_template).T
        update_covariance_template = np.mat(update_covariance_template).T

        update_state_template = self.H.T * detected_state_template

        object = Object()

        object.updated_state = update_state_template
        object.predicted_state = update_state_template
        object.detected_state = detected_state_template
        object.updated_covariance =update_covariance_template
        object.predicted_covariance = update_covariance_template
        object.prediction_score = 1
        object.score=self.init_score
        object.features = self.init_features
        object.features_dict[str(self.init_timestamp)] = self.init_features

        self.trajectory[self.init_timestamp] = object

    def init_parameters(self):
        """
        initialize KF tracking parameters
        :return:
        """
        self.A = np.mat(np.eye(self.state_dim))
        # state vector：[x, y, z, xv, yv, zv, l, w, h, yaw]
        # measure vector：[x, y, z, l, w, h, yaw]
        self.Q = np.mat(np.eye(self.state_dim)) * 1.
        self.Q[3:6, 3:6] *= 0.01
        self.R = np.mat(np.eye(self.measure_dim)) * 1.
        self.init_P = np.mat(np.diag([10., 10., 10., 10000., 10000., 10000., 10., 10., 10., 10.]))
        self.H = np.mat(np.zeros(shape=(self.measure_dim, self.state_dim)))
        self.H[0:3,:] = self.A[0:3,:]
        self.H[3:,:] = self.A[6:,:]

        self.velo = np.mat(np.eye(3)) * self.scanning_interval

        self.A[0:3,3:6] = self.velo

    def getInitCovP(self) -> np.mat:
        vector_p = [4, 4, 4, 1000, 1000, 1000, 4, 4, 4, 1]
        return np.mat(np.diag(vector_p))

    def state_prediction(self,timestamp):
        """
        predict the object state at the given timestamp
        """

        previous_timestamp = timestamp-1

        assert previous_timestamp in self.trajectory.keys()

        previous_object = self.trajectory[previous_timestamp]

        if previous_object.updated_state is not None:
            previous_state = previous_object.updated_state
            previous_covariance = previous_object.updated_covariance
        else:
            previous_state = previous_object.predicted_state
            previous_covariance = previous_object.predicted_covariance

        previous_prediction_score = previous_object.prediction_score
        current_prediction_score = previous_prediction_score

        self.warpStateYawToPi(previous_state)
        current_predicted_state = self.A * previous_state
        current_predicted_covariance = self.A * previous_covariance * self.A.T + self.Q
        self.warpStateYawToPi(current_predicted_state)

        new_ob = Object()

        new_ob.predicted_state = current_predicted_state
        new_ob.predicted_covariance = current_predicted_covariance
        new_ob.prediction_score = current_prediction_score
        new_ob.features = previous_object.features
        new_ob.features_dict = previous_object.features_dict

        self.trajectory[timestamp] = new_ob
        self.consecutive_missed_num += 1

    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-float(x)))

    def state_update(self,
                     bb=None,
                     features=None,
                     score=None,
                     timestamp=None
                     ):
        """
        update the trajectory
        Args:
            bb: array(7) or array(7*k), 3D box or tracklet
            features: array(m), features of box or tracklet
            score:
            timestamp:
        """
        assert bb is not None
        assert timestamp in self.trajectory.keys()

        detected_state_template = np.zeros(shape=(self.measure_dim))

        detected_state_template[:self.measure_dim] = bb[:self.measure_dim]

        detected_state_template = np.mat(detected_state_template).T
        self.yaw_recover(detected_state_template, timestamp)

        current_ob = self.trajectory[timestamp]

        predicted_state = current_ob.predicted_state
        predicted_covariance = current_ob.predicted_covariance

        RES = detected_state_template - self.H * predicted_state
        self.warpMearureYawToPi(RES)
        S = self.H * predicted_covariance * self.H.T + self.R
        KF_GAIN = predicted_covariance * self.H.T * S.I

        updated_state = predicted_state + KF_GAIN * RES
        updated_covariance = (np.mat(np.eye(self.state_dim)) - KF_GAIN * self.H) * predicted_covariance
        self.warpStateYawToPi(updated_state)

        current_ob.updated_state = updated_state
        current_ob.updated_covariance = updated_covariance
        current_ob.detected_state = detected_state_template
        if self.consecutive_missed_num > 1:
            current_ob.prediction_score = 1
        else:
            current_ob.prediction_score = current_ob.prediction_score
        current_ob.score = score
        if current_ob.features is None:
            current_ob.features = features
        else:
            current_ob.features = self.feat_momentum * features + (1 - self.feat_momentum) * current_ob.features

        current_ob.features_dict[str(timestamp)] = features

        self.consecutive_missed_num = 0
        self.last_updated_timestamp = timestamp

class CA(Basetra):
    def __init__(self,
                 init_bb=None,
                 init_features=None,
                 init_score=None,
                 init_timestamp=None,
                 label=None,
                 config = None
                 ):
        """

        Args:
            init_bb: array(7) or array(7*k), 3d box or tracklet
            init_features: array(m), features of box or tracklet
            init_score: array(1) or float, score of detection
            init_timestamp: int, init timestamp
            label: int, unique ID for this trajectory
        """
        assert init_bb is not None

        self.init_bb = init_bb
        self.init_features = init_features
        self.init_score = init_score
        self.init_timestamp = init_timestamp
        self.label = label

        self.config = config
        self.feat_momentum = 0.9

        self.scanning_interval = 1. / self.config.LiDAR_scanning_frequency

        self.trajectory = {}

        # state vector：[x, y, z, xv, yv, zv, ax, ay, az, l, w, h, yaw]
        # measure vector：[x, y, z, l, w, h, yaw]
        self.state_dim = 13
        self.measure_dim = 7
        self.yaw_ids = 12

        self.init_parameters()
        self.init_trajectory()

        self.consecutive_missed_num = 0
        self.first_updated_timestamp = init_timestamp
        self.last_updated_timestamp = init_timestamp

    def __len__(self):
        return len(self.trajectory)

    def warpStateYawToPi(self, state: np.mat) -> np.mat:
        state[-1, 0] = warp_to_pi(state[-1, 0])
    
    def warpMearureYawToPi(self, state: np.mat) -> np.mat:
        state[-1, 0] = warp_to_pi(state[-1, 0])

    def init_trajectory(self):
        """
        first initialize the object state with the input boxes info,
        then initialize the trajectory with the initialized object.
        :return:
        """

        detected_state_template = np.zeros(shape=(self.measure_dim))

        update_covariance_template = self.init_P

        detected_state_template[:self.measure_dim] = self.init_bb[:self.measure_dim]

        detected_state_template = np.mat(detected_state_template).T
        update_covariance_template = np.mat(update_covariance_template).T

        update_state_template = self.H.T * detected_state_template

        object = Object()

        object.updated_state = update_state_template
        object.predicted_state = update_state_template
        object.detected_state = detected_state_template
        object.updated_covariance =update_covariance_template
        object.predicted_covariance = update_covariance_template
        object.prediction_score = 1
        object.score=self.init_score
        object.features = self.init_features
        object.features_dict[str(self.init_timestamp)] = self.init_features

        self.trajectory[self.init_timestamp] = object

    def init_parameters(self):
        """
        initialize KF tracking parameters
        :return:
        """

        # state vector：[x, y, z, xv, yv, zv, ax, ay, az, l, w, h, yaw]
        # measure vector：[x, y, z, l, w, h, yaw]

        self.A = np.mat(np.eye(self.state_dim))
        self.Q = np.mat(np.eye(self.state_dim)) * 1.
        self.Q[6:9, 6:9] *= 0.01
        self.R = np.mat(np.eye(self.measure_dim)) * 1.
        self.init_P = np.mat(np.diag([10., 10., 10., 10000., 10000., 10000., 10000., 10000., 10000., 10., 10., 10., 10.]))
        self.H = np.mat(np.zeros(shape=(self.measure_dim, self.state_dim)))
        self.H[0:3,:] = self.A[0:3,:]
        self.H[3:,:] = self.A[9:,:]

        self.velo = np.mat(np.eye(3)) * self.scanning_interval
        self.acce = np.mat(np.eye(3)) * 0.5 * self.scanning_interval ** 2

        self.A[0:3,3:6] = self.velo
        self.A[3:6,6:9] = self.velo
        self.A[0:3,6:9] = self.acce

    def state_prediction(self,timestamp):
        """
        predict the object state at the given timestamp
        """

        previous_timestamp = timestamp-1

        assert previous_timestamp in self.trajectory.keys()

        previous_object = self.trajectory[previous_timestamp]

        if previous_object.updated_state is not None:
            previous_state = previous_object.updated_state
            previous_covariance = previous_object.updated_covariance
        else:
            previous_state = previous_object.predicted_state
            previous_covariance = previous_object.predicted_covariance

        previous_prediction_score = previous_object.prediction_score
        current_prediction_score = previous_prediction_score

        self.warpStateYawToPi(previous_state)
        current_predicted_state = self.A * previous_state
        current_predicted_covariance = self.A * previous_covariance * self.A.T + self.Q
        self.warpStateYawToPi(current_predicted_state)

        new_ob = Object()

        new_ob.predicted_state = current_predicted_state
        new_ob.predicted_covariance = current_predicted_covariance
        new_ob.prediction_score = current_prediction_score
        new_ob.features = previous_object.features
        new_ob.features_dict = previous_object.features_dict

        self.trajectory[timestamp] = new_ob
        self.consecutive_missed_num += 1

    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-float(x)))

    def state_update(self,
                     bb=None,
                     features=None,
                     score=None,
                     timestamp=None
                     ):
        """
        update the trajectory
        Args:
            bb: array(7) or array(7*k), 3D box or tracklet
            features: array(m), features of box or tracklet
            score:
            timestamp:
        """
        assert bb is not None
        assert timestamp in self.trajectory.keys()

        detected_state_template = np.zeros(shape=(self.measure_dim))

        detected_state_template[:self.measure_dim] = bb[:self.measure_dim]

        detected_state_template = np.mat(detected_state_template).T
        self.yaw_recover(detected_state_template, timestamp)

        current_ob = self.trajectory[timestamp]

        predicted_state = current_ob.predicted_state
        predicted_covariance = current_ob.predicted_covariance

        RES = detected_state_template - self.H * predicted_state
        self.warpMearureYawToPi(RES)
        S = self.H * predicted_covariance * self.H.T + self.R
        KF_GAIN = predicted_covariance * self.H.T * S.I

        updated_state = predicted_state + KF_GAIN * RES
        updated_covariance = (np.mat(np.eye(self.state_dim)) - KF_GAIN * self.H) * predicted_covariance
        self.warpStateYawToPi(updated_state)

        current_ob.updated_state = updated_state
        current_ob.updated_covariance = updated_covariance
        current_ob.detected_state = detected_state_template
        if self.consecutive_missed_num > 1:
            current_ob.prediction_score = 1
        else:
            current_ob.prediction_score = current_ob.prediction_score
        current_ob.score = score
        if current_ob.features is None:
            current_ob.features = features
        else:
            current_ob.features = self.feat_momentum * features + (1 - self.feat_momentum) * current_ob.features

        current_ob.features_dict[str(timestamp)] = features

        self.consecutive_missed_num = 0
        self.last_updated_timestamp = timestamp

class CTRV(Basetra):
    def __init__(self,
                 init_bb=None,
                 init_features=None,
                 init_score=None,
                 init_timestamp=None,
                 label=None,
                 config = None
                 ):
        """

        Args:
            init_bb: array(7) or array(7*k), 3d box or tracklet
            init_features: array(m), features of box or tracklet
            init_score: array(1) or float, score of detection
            init_timestamp: int, init timestamp
            label: int, unique ID for this trajectory
        """
        assert init_bb is not None

        self.init_bb = init_bb
        self.init_features = init_features
        self.init_score = init_score
        self.init_timestamp = init_timestamp
        self.label = label

        self.config = config
        self.feat_momentum = 0.9

        self.scanning_interval = 1./self.config.LiDAR_scanning_frequency

        self.trajectory = {}

        # state vector：[x, y, z, l, w, h, v, theta, omega]
        # measure vector：[x, y, z, l, w, h, theta]
        self.state_dim = 9
        self.measure_dim = 7
        self.yaw_ids = 7

        self.init_parameters()
        self.init_trajectory()

        self.consecutive_missed_num = 0
        self.first_updated_timestamp = init_timestamp
        self.last_updated_timestamp = init_timestamp

    def __len__(self):
        return len(self.trajectory)

    def init_trajectory(self):
        """
        first initialize the object state with the input boxes info,
        then initialize the trajectory with the initialized object.
        :return:
        """

        detected_state_template = np.zeros(shape=(self.measure_dim))

        update_covariance_template = self.init_P

        detected_state_template[:self.measure_dim] = self.init_bb[:self.measure_dim]

        detected_state_template = np.mat(detected_state_template).T
        update_covariance_template = np.mat(update_covariance_template).T

        update_state_template = self.H.T * detected_state_template

        object = Object()

        object.updated_state = update_state_template
        object.predicted_state = update_state_template
        object.detected_state = detected_state_template
        object.updated_covariance =update_covariance_template
        object.predicted_covariance = update_covariance_template
        object.prediction_score = 1
        object.score=self.init_score
        object.features = self.init_features
        object.features_dict[str(self.init_timestamp)] = self.init_features

        self.trajectory[self.init_timestamp] = object

    def init_parameters(self):
        """
        initialize KF tracking parameters
        :return:
        """
        self.A = None
        self.H = self.getMeasureStateH()

        # state vector：[x, y, z, l, w, h, v, theta, omega]
        # measure vector：[x, y, z, l, w, h, theta]
        self.Q = np.mat(np.eye(self.state_dim)) * 1.
        self.Q[6, 6] *= 0.01
        self.Q[8, 8] *= 0.01
        self.R = np.mat(np.eye(self.measure_dim)) * 1.
        self.init_P = np.mat(np.diag([10., 10., 10., 10., 10., 10., 10000., 10., 10000.]))

    def getMeasureStateH(self) ->np.mat:
        H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0]])
        return H

    def get_transition_matrix(self, state: np.mat) ->np.mat:
        """obtain matrix in the script/CTRA_kinect_jacobian.ipynb
        d(stateTransition) / d(state) at previous_state
        """
        dt = self.scanning_interval
        _, _, _, _, _, _, v, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        
        # corner case, tiny turn rate
        if abs(omega) < 0.001:
            A = np.mat([[1, 0, 0, 0, 0, 0,  dt*yaw_cos,                -dt*v*yaw_sin,  0],
                        [0, 1, 0, 0, 0, 0,  dt*yaw_sin,                 dt*v*yaw_cos,  0],
                        [0, 0, 1, 0, 0, 0,           0,                            0,  0],
                        [0, 0, 0, 1, 0, 0,           0,                            0,  0],
                        [0, 0, 0, 0, 1, 0,           0,                            0,  0],
                        [0, 0, 0, 0, 0, 1,           0,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           1,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           0,                            1, dt],
                        [0, 0, 0, 0, 0, 0,           0,                            0,  1]])
        else:
            ry_rate_inv, ry_rate_inv_square = 1.0 / omega, 1.0 / (omega * omega)
            next_ry = theta + omega * dt
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            A = np.mat([[1, 0, 0, 0, 0, 0,  -ry_rate_inv*(yaw_sin-next_yaw_sin),   -v*ry_rate_inv*(yaw_cos-next_yaw_cos),        ry_rate_inv*dt*v*next_yaw_cos + v*yaw_sin*ry_rate_inv_square - v*next_yaw_sin*ry_rate_inv_square],
                        [0, 1, 0, 0, 0, 0,   ry_rate_inv*(yaw_cos-next_yaw_cos),   -v*ry_rate_inv*(yaw_sin-next_yaw_sin),        ry_rate_inv*dt*v*next_yaw_sin - v*yaw_cos*ry_rate_inv_square + v*next_yaw_cos*ry_rate_inv_square],
                        [0, 0, 1, 0, 0, 0,           0,                 0,                            0],
                        [0, 0, 0, 1, 0, 0,           0,                 0,                            0],
                        [0, 0, 0, 0, 1, 0,           0,                 0,                            0],
                        [0, 0, 0, 0, 0, 1,           0,                 0,                            0],
                        [0, 0, 0, 0, 0, 0,           1,                 0,                            0],
                        [0, 0, 0, 0, 0, 0,           0,                 1,                           dt],
                        [0, 0, 0, 0, 0, 0,           0,                 0,                            1]])
                                
        return A 
    
    def state_transition(self, state: list) ->list:
        """state transition, 
        obtain analytical solutions in the motion_module/script/CTRA_kinect_jacobian.ipynb
        Args:
            state (np.mat): [state dim, 1] the estimated state of the previous frame

        Returns:
            np.mat: [state dim, 1] the predict state of the current frame
        """
        assert state.shape == (self.state_dim, 1), "state vector number in CTRA must equal to 10"
        
        dt = self.scanning_interval
        x, y, z, l, w, h, v, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        next_ry = theta + omega * dt
        
        # corner case(tiny yaw rate), prevent divide-by-zero overflow
        if abs(omega) < 0.001:
            predict_state = [x + v * dt * yaw_cos,
                             y + v * dt * yaw_sin,
                             z, l, w, h, 
                             v, 
                             next_ry, omega]
        else:
            ry_rate_inv = 1.0 / omega
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            predict_state = [x + ry_rate_inv * v * (-yaw_sin + next_yaw_sin),
                             y + ry_rate_inv * v * (yaw_cos - next_yaw_cos),
                             z, l, w, h,
                             v, 
                             next_ry, omega]
        
        return np.mat(predict_state).T

    def warpStateYawToPi(self, state: np.mat) -> np.mat:
        state[-2, 0] = warp_to_pi(state[-2, 0])
    
    def warpMearureYawToPi(self, state: np.mat) -> np.mat:
        state[-1, 0] = warp_to_pi(state[-1, 0])

    def state_prediction(self,timestamp):
        """
        predict the object state at the given timestamp
        """

        previous_timestamp = timestamp - 1

        assert previous_timestamp in self.trajectory.keys()

        previous_object = self.trajectory[previous_timestamp]

        if previous_object.updated_state is not None:
            previous_state = previous_object.updated_state
            previous_covariance = previous_object.updated_covariance
        else:
            previous_state = previous_object.predicted_state
            previous_covariance = previous_object.predicted_covariance

        previous_prediction_score = previous_object.prediction_score
        current_prediction_score = previous_prediction_score

        self.warpStateYawToPi(previous_state)
        self.A = self.get_transition_matrix(previous_state)
        current_predicted_state = self.state_transition(previous_state)
        current_predicted_covariance = self.A * previous_covariance * self.A.T + self.Q
        self.warpStateYawToPi(current_predicted_state)

        new_ob = Object()

        new_ob.predicted_state = current_predicted_state
        new_ob.predicted_covariance = current_predicted_covariance
        new_ob.prediction_score = current_prediction_score
        new_ob.features = previous_object.features
        new_ob.features_dict = previous_object.features_dict

        self.trajectory[timestamp] = new_ob
        self.consecutive_missed_num += 1

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-float(x)))

    def state_update(self,
                     bb=None,
                     features=None,
                     score=None,
                     timestamp=None
                     ):
        """
        update the trajectory
        Args:
            bb: array(7) or array(7*k), 3D box or tracklet
            features: array(m), features of box or tracklet
            score:
            timestamp:
        """
        assert bb is not None
        assert timestamp in self.trajectory.keys()

        detected_state_template = np.zeros(shape=(self.measure_dim))

        detected_state_template[:self.measure_dim] = bb[:self.measure_dim]

        detected_state_template = np.mat(detected_state_template).T
        self.yaw_recover(detected_state_template, timestamp)

        current_ob = self.trajectory[timestamp]

        predicted_state = current_ob.predicted_state
        predicted_covariance = current_ob.predicted_covariance

        RES = detected_state_template - self.H * predicted_state
        self.warpMearureYawToPi(RES)
        S = self.H * predicted_covariance * self.H.T + self.R
        KF_GAIN = predicted_covariance * self.H.T * S.I

        updated_state = predicted_state + KF_GAIN * RES
        updated_covariance = (np.mat(np.eye(self.state_dim)) - KF_GAIN * self.H) * predicted_covariance
        self.warpStateYawToPi(updated_state)

        current_ob.updated_state = updated_state
        current_ob.updated_covariance = updated_covariance
        current_ob.detected_state = detected_state_template
        if self.consecutive_missed_num > 1:
            current_ob.prediction_score = 1
        else:
            current_ob.prediction_score = current_ob.prediction_score
        current_ob.score = score
        if current_ob.features is None:
            current_ob.features = features
        else:
            current_ob.features = self.feat_momentum * features + (1 - self.feat_momentum) * current_ob.features

        current_ob.features_dict[str(timestamp)] = features

        self.consecutive_missed_num = 0
        self.last_updated_timestamp = timestamp

class CTRA(Basetra):
    def __init__(self,
                 init_bb=None,
                 init_features=None,
                 init_score=None,
                 init_timestamp=None,
                 label=None,
                 config = None
                 ):
        """

        Args:
            init_bb: array(7) or array(7*k), 3d box or tracklet
            init_features: array(m), features of box or tracklet
            init_score: array(1) or float, score of detection
            init_timestamp: int, init timestamp
            label: int, unique ID for this trajectory
        """
        assert init_bb is not None

        self.init_bb = init_bb
        self.init_features = init_features
        self.init_score = init_score
        self.init_timestamp = init_timestamp
        self.label = label

        self.config = config
        self.feat_momentum = 0.9

        self.scanning_interval = 1./self.config.LiDAR_scanning_frequency

        self.trajectory = {}

        # state vector：[x, y, z, l, w, h, v, a, theta, omega]
        # measure vector：[x, y, z, l, w, h, theta]
        self.state_dim = 10
        self.measure_dim = 7
        self.yaw_ids = 8

        self.init_parameters()
        self.init_trajectory()

        self.consecutive_missed_num = 0
        self.first_updated_timestamp = init_timestamp
        self.last_updated_timestamp = init_timestamp

    def __len__(self):
        return len(self.trajectory)

    def init_trajectory(self):
        """
        first initialize the object state with the input boxes info,
        then initialize the trajectory with the initialized object.
        :return:
        """

        detected_state_template = np.zeros(shape=(self.measure_dim))

        update_covariance_template = self.init_P

        detected_state_template[:self.measure_dim] = self.init_bb[:self.measure_dim]

        detected_state_template = np.mat(detected_state_template).T
        update_covariance_template = np.mat(update_covariance_template).T

        update_state_template = self.H.T * detected_state_template

        object = Object()

        object.updated_state = update_state_template
        object.predicted_state = update_state_template
        object.detected_state = detected_state_template
        object.updated_covariance =update_covariance_template
        object.predicted_covariance = update_covariance_template
        object.prediction_score = 1
        object.score=self.init_score
        object.features = self.init_features
        object.features_dict[str(self.init_timestamp)] = self.init_features

        self.trajectory[self.init_timestamp] = object

    def init_parameters(self):
        """
        initialize KF tracking parameters
        :return:
        """
        self.A = None
        self.H = self.getMeasureStateH()

        # state vector：[x, y, z, l, w, h, v, a, theta, omega]
        # measure vector：[x, y, z, l, w, h, theta]
        self.Q = np.mat(np.eye(self.state_dim)) * 1.
        self.Q[7, 7] *= 0.01
        self.Q[9, 9] *= 0.01
        self.R = np.mat(np.eye(self.measure_dim)) * 1.
        self.init_P = np.mat(np.diag([10., 10., 10., 10., 10., 10., 10000., 10000., 10., 10000.]))

    def getInitCovP(self) -> np.mat:
        vector_p = [4, 4, 4, 4, 4, 4, 1000, 4, 1, 0.1]
        return np.mat(np.diag(vector_p))

    def getMeasureStateH(self) ->np.mat:
        H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
        return H

    def get_transition_matrix(self, state: np.mat) ->np.mat:
        """obtain matrix in the script/CTRA_kinect_jacobian.ipynb
        d(stateTransition) / d(state) at previous_state
        """
        dt = self.scanning_interval
        _, _, _, _, _, _, v, a, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        
        # corner case, tiny turn rate
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt ** 2 / 2
            A = np.mat([[1, 0, 0, 0, 0, 0,  dt*yaw_cos,   dt**2*yaw_cos/2,        -displacement*yaw_sin,  0],
                        [0, 1, 0, 0, 0, 0,  dt*yaw_sin,   dt**2*yaw_sin/2,         displacement*yaw_cos,  0],
                        [0, 0, 1, 0, 0, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 1, 0, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 1, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 0, 1,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           1,                dt,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           0,                 1,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           0,                 0,                            1, dt],
                        [0, 0, 0, 0, 0, 0,           0,                 0,                            0,  1]])
        else:
            ry_rate_inv, ry_rate_inv_square, ry_rate_inv_cube = 1 / omega, 1 / (omega * omega), 1 / (omega * omega * omega)
            next_v, next_ry = v + a * dt, theta + omega * dt
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            A = np.mat([[1, 0, 0, 0, 0, 0,  -ry_rate_inv*(yaw_sin-next_yaw_sin),   -ry_rate_inv_square*(yaw_cos-next_yaw_cos)+ry_rate_inv*dt*next_yaw_sin,        ry_rate_inv_square*a*(yaw_sin-next_yaw_sin)+ry_rate_inv*(next_v*next_yaw_cos-v*yaw_cos),  ry_rate_inv_cube*2*a*(yaw_cos-next_yaw_cos)+ry_rate_inv_square*(v*yaw_sin-v*next_yaw_sin-2*a*dt*next_yaw_sin)+ry_rate_inv*dt*next_v*next_yaw_cos ],
                        [0, 1, 0, 0, 0, 0,   ry_rate_inv*(yaw_cos-next_yaw_cos),   -ry_rate_inv_square*(yaw_sin-next_yaw_sin)-ry_rate_inv*dt*next_yaw_cos,        ry_rate_inv_square*a*(-yaw_cos+next_yaw_cos)+ry_rate_inv*(next_v*next_yaw_sin-v*yaw_sin), ry_rate_inv_cube*2*a*(yaw_sin-next_yaw_sin)+ry_rate_inv_square*(v*next_yaw_cos-v*yaw_cos+2*a*dt*next_yaw_cos)+ry_rate_inv*dt*next_v*next_yaw_sin ],
                        [0, 0, 1, 0, 0, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 1, 0, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 1, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 0, 1,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           1,                dt,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           0,                 1,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           0,                 0,                            1, dt],
                        [0, 0, 0, 0, 0, 0,           0,                 0,                            0,  1]])
                                
        return A 
    
    def state_transition(self, state: list) ->list:
        """state transition, 
        obtain analytical solutions in the motion_module/script/CTRA_kinect_jacobian.ipynb
        Args:
            state (np.mat): [state dim, 1] the estimated state of the previous frame

        Returns:
            np.mat: [state dim, 1] the predict state of the current frame
        """
        assert state.shape == (self.state_dim, 1), "state vector number in CTRA must equal to 10"
        
        dt = self.scanning_interval
        x, y, z, l, w, h, v, a, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        next_v, next_ry = v + a * dt, theta + omega * dt
        
        # corner case(tiny yaw rate), prevent divide-by-zero overflow
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt ** 2 / 2
            predict_state = [x + displacement * yaw_cos,
                             y + displacement * yaw_sin,
                             z, l, w, h, 
                             next_v, a, 
                             next_ry, omega]
        else:
            ry_rate_inv_square = 1.0 / (omega * omega)
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            predict_state = [x + ry_rate_inv_square * (next_v * omega * next_yaw_sin + a * next_yaw_cos - v * omega * yaw_sin - a * yaw_cos),
                             y + ry_rate_inv_square * (-next_v * omega * next_yaw_cos + a * next_yaw_sin + v * omega * yaw_cos - a * yaw_sin),
                             z, l, w, h,
                             next_v, a, 
                             next_ry, omega]
        
        return np.mat(predict_state).T

    def warpStateYawToPi(self, state: np.mat) -> np.mat:
        state[-2, 0] = warp_to_pi(state[-2, 0])
    
    def warpMearureYawToPi(self, state: np.mat) -> np.mat:
        state[-1, 0] = warp_to_pi(state[-1, 0])

    def state_prediction(self,timestamp):
        """
        predict the object state at the given timestamp
        """

        previous_timestamp = timestamp - 1

        assert previous_timestamp in self.trajectory.keys()

        previous_object = self.trajectory[previous_timestamp]

        if previous_object.updated_state is not None:
            previous_state = previous_object.updated_state
            previous_covariance = previous_object.updated_covariance
        else:
            previous_state = previous_object.predicted_state
            previous_covariance = previous_object.predicted_covariance

        previous_prediction_score = previous_object.prediction_score
        current_prediction_score = previous_prediction_score

        self.warpStateYawToPi(previous_state)
        self.A = self.get_transition_matrix(previous_state)
        current_predicted_state = self.state_transition(previous_state)
        current_predicted_covariance = self.A * previous_covariance * self.A.T + self.Q
        self.warpStateYawToPi(current_predicted_state)

        new_ob = Object()

        new_ob.predicted_state = current_predicted_state
        new_ob.predicted_covariance = current_predicted_covariance
        new_ob.prediction_score = current_prediction_score
        new_ob.features = previous_object.features
        new_ob.features_dict = previous_object.features_dict

        self.trajectory[timestamp] = new_ob
        self.consecutive_missed_num += 1

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-float(x)))

    def state_update(self,
                     bb=None,
                     features=None,
                     score=None,
                     timestamp=None
                     ):
        """
        update the trajectory
        Args:
            bb: array(7) or array(7*k), 3D box or tracklet
            features: array(m), features of box or tracklet
            score:
            timestamp:
        """
        assert bb is not None
        assert timestamp in self.trajectory.keys()

        detected_state_template = np.zeros(shape=(self.measure_dim))

        detected_state_template[:self.measure_dim] = bb[:self.measure_dim]

        detected_state_template = np.mat(detected_state_template).T
        self.yaw_recover(detected_state_template, timestamp)

        current_ob = self.trajectory[timestamp]

        predicted_state = current_ob.predicted_state
        predicted_covariance = current_ob.predicted_covariance

        RES = detected_state_template - self.H * predicted_state
        self.warpMearureYawToPi(RES)
        S = self.H * predicted_covariance * self.H.T + self.R
        KF_GAIN = predicted_covariance * self.H.T * S.I

        updated_state = predicted_state + KF_GAIN * RES
        updated_covariance = (np.mat(np.eye(self.state_dim)) - KF_GAIN * self.H) * predicted_covariance
        self.warpStateYawToPi(updated_state)

        current_ob.updated_state = updated_state
        current_ob.updated_covariance = updated_covariance
        current_ob.detected_state = detected_state_template
        if self.consecutive_missed_num > 1:
            current_ob.prediction_score = 1
        else:
            current_ob.prediction_score = current_ob.prediction_score
        current_ob.score = score
        if current_ob.features is None:
            current_ob.features = features
        else:
            current_ob.features = self.feat_momentum * features + (1 - self.feat_momentum) * current_ob.features

        current_ob.features_dict[str(timestamp)] = features

        self.consecutive_missed_num = 0
        self.last_updated_timestamp = timestamp

class without_motion:
    def __init__(self,
                 init_bb=None,
                 init_features=None,
                 init_score=None,
                 init_timestamp=None,
                 label=None,
                 config = None
                 ):
        """

        Args:
            init_bb: array(7) or array(7*k), 3d box or tracklet
            init_features: array(m), features of box or tracklet
            init_score: array(1) or float, score of detection
            init_timestamp: int, init timestamp
            label: int, unique ID for this trajectory
        """
        assert init_bb is not None

        self.init_bb = init_bb
        self.init_features = init_features
        self.init_score = init_score
        self.init_timestamp = init_timestamp
        self.label = label

        self.config = config
        self.feat_momentum = 0.9

        self.scanning_interval = 1. / self.config.LiDAR_scanning_frequency

        self.trajectory = {}

        # state vector：[x, y, z, xv, yv, zv, ax, ay, az, l, w, h, yaw]
        # measure vector：[x, y, z, l, w, h, yaw]
        self.state_dim = 7
        self.measure_dim = 7

        self.init_trajectory()

        self.consecutive_missed_num = 0
        self.first_updated_timestamp = init_timestamp
        self.last_updated_timestamp = init_timestamp

    def __len__(self):
        return len(self.trajectory)

    # def warpStateYawToPi(self, state: np.mat) -> np.mat:
    #     state[-1, 0] = warp_to_pi(state[-1, 0])
    
    # def warpMearureYawToPi(self, state: np.mat) -> np.mat:
    #     state[-1, 0] = warp_to_pi(state[-1, 0])

    def init_trajectory(self):
        """
        first initialize the object state with the input boxes info,
        then initialize the trajectory with the initialized object.
        :return:
        """

        detected_state_template = np.zeros(shape=(self.measure_dim))

        detected_state_template[:self.measure_dim] = self.init_bb[:self.measure_dim]

        detected_state_template = np.mat(detected_state_template).T

        object = Object()

        object.updated_state = detected_state_template
        object.predicted_state = detected_state_template
        object.detected_state = detected_state_template
        object.prediction_score = 1
        object.score=self.init_score
        object.features = self.init_features
        object.features_dict[str(self.init_timestamp)] = self.init_features

        self.trajectory[self.init_timestamp] = object

    def state_prediction(self,timestamp):
        """
        predict the object state at the given timestamp
        """

        previous_timestamp = timestamp - 1

        assert previous_timestamp in self.trajectory.keys()

        previous_object = self.trajectory[previous_timestamp]

        if previous_object.updated_state is not None:
            previous_state = previous_object.updated_state
        else:
            previous_state = previous_object.predicted_state

        current_prediction_score = previous_object.prediction_score

        # self.warpStateYawToPi(previous_state)
        current_predicted_state = previous_state
        # self.warpStateYawToPi(current_predicted_state)

        new_ob = Object()

        new_ob.predicted_state = current_predicted_state
        new_ob.prediction_score = current_prediction_score
        new_ob.features = previous_object.features
        new_ob.features_dict = previous_object.features_dict

        self.trajectory[timestamp] = new_ob
        self.consecutive_missed_num += 1

    def state_update(self,
                     bb=None,
                     features=None,
                     score=None,
                     timestamp=None
                     ):
        """
        update the trajectory
        Args:
            bb: array(7) or array(7*k), 3D box or tracklet
            features: array(m), features of box or tracklet
            score:
            timestamp:
        """
        assert bb is not None
        assert timestamp in self.trajectory.keys()

        detected_state_template = np.zeros(shape=(self.measure_dim))

        detected_state_template[:self.measure_dim] = bb[:self.measure_dim]

        detected_state_template = np.mat(detected_state_template).T

        current_ob = self.trajectory[timestamp]

        predicted_state = current_ob.predicted_state
        predicted_covariance = current_ob.predicted_covariance

        # self.warpMearureYawToPi(RES)
        # self.warpStateYawToPi(updated_state)

        current_ob.updated_state = detected_state_template
        current_ob.detected_state = detected_state_template
        if self.consecutive_missed_num > 1:
            current_ob.prediction_score = 1
        current_ob.score = score
        if current_ob.features is None:
            current_ob.features = features
        else:
            current_ob.features = self.feat_momentum * features + (1 - self.feat_momentum) * current_ob.features

        current_ob.features_dict[str(timestamp)] = features

        self.consecutive_missed_num = 0
        self.last_updated_timestamp = timestamp