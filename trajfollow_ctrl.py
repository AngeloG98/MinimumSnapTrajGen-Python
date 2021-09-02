import time
import copy
import airsim
import numpy as np
import scipy.io as scio
from airsim.utils import wait_key
from airsim.types import KinematicsState
from numpy.lib.shape_base import expand_dims
from minsnap_traj import minimum_snap_traj
from minsnap_traj import  mimimum_snap_traj_p2p_id


class traj_follow_ctrl:
    def __init__(self, client, set_points, setpoint_land) -> None:
        ## Connect AirSim
        self.client = client
        self.setpoints = copy.deepcopy(set_points)
        self.landing = copy.deepcopy(setpoint_land)
        self.landing.append(np.deg2rad(-90))
        self.way_points = copy.deepcopy(self.setpoints)
        self.way_points.insert(0,[0,0,0,np.deg2rad(-90)])
        self.way_points.append(self.landing)
        # Control Parameters
        # self.kp = [2, 2 ,3, 2]# in order x, y, z, yaw
        # self.ki =  [0, 0, 0.1, 0]
        # self.kd = [1, 1, 0, 0]
        self.kp = [2, 2 ,3, 2]# in order x, y, z, yaw
        self.ki =  [0, 0, 0.1, 0]
        self.kd = [0, 0, 1, 0]
        self.I_saturation = 0.2
        self.sampletime = 0.01
        # Log Data
        self.des_log = []
        # Trajectroy Sourse Option
        # self.load_mat_data()
        self.gen_traj()
        # Reset
        self.reset()

    def load_mat_data(self):
        data = scio.loadmat('/run/user/1000/gvfs/afp-volume:host=CatDisk.local,user=a,volume=home/毛鹏达/test_mao.mat')
        # data = scio.loadmat('stable_traj/test_mao_35_stable.mat')
        self.list_time = data['ts'][0]
        self.matrix_x = data['polys_x']
        self.matrix_y = data['polys_y']
        self.matrix_z = data['polys_z']

    def gen_traj(self):
        self.list_time ,self.matrix_x , self.matrix_y ,self.matrix_z = minimum_snap_traj(self.way_points)
    
    def gen_traj_single(self,point_id, p_i, v_i, a_i):
        self.matrix_x , self.matrix_y ,self.matrix_z = mimimum_snap_traj_p2p_id(self.way_points, point_id, p_i, v_i, a_i)
        
    def reset(self):
        self.last_p_err = np.array([0.0, 0.0, 0.0])
        self.p_err_i =  np.array([0.0, 0.0, 0.0])
        self.last_yaw_err = 0.0
        self.yaw_err_i = 0.0
        self.in_time = 0.0
        self.systime_last = time.time()

    def minAngleDiff(self, a, b):
        diff = a - b
        if diff < 0:
            diff += 2*np.pi
        if diff < np.pi:
            return diff
        else:
            return diff - 2*np.pi

    def update_in_time(self):
        self.in_time =self.in_time + time.time()-self.systime_last
        self.systime_last = time.time()

    def traj_pid_update(self, p_des, v_des, yaw_des):
        # Get multirotor states
        state = self.client.getMultirotorState()
        quaternionr = state.kinematics_estimated.orientation
        _, _, yaw_ref = airsim.to_eularian_angles(quaternionr)
        position = state.kinematics_estimated.position
        p_ref = np.array([position.x_val, position.y_val, position.z_val])

        # Position control
        v_cmd = np.array([0.0,0.0,0.0])
        p_err = p_des - p_ref
        vel = state.kinematics_estimated.linear_velocity
        vel_vec = [vel.x_val, vel.y_val, vel.z_val]
        for i in range(3):
            # v_cmd[i] = (v_des[i]-vel_vec[i]) + self.kp[i]*p_err[i] + self.ki[i]*self.p_err_i[i] + self.kd[i]*(p_err[i]-self.last_p_err[i])/self.sampletime
            v_cmd[i] = v_des[i] + self.kp[i]*p_err[i] + self.ki[i]*self.p_err_i[i] + self.kd[i]*(p_err[i]-self.last_p_err[i])/self.sampletime
        self.last_p_err = p_err
        self.p_err_i = self.p_err_i + self.last_p_err
        for i in range(3):
            self.p_err_i[i] = np.sign(self.p_err_i[i]) * min(abs(self.p_err_i[i]), self.I_saturation)
        # Yaw control
        yaw_rate_cmd = 0.0
        yaw_err = self.minAngleDiff(yaw_des, yaw_ref)
        yaw_rate_cmd = self.kp[3]*yaw_err + self.ki[3]*self.yaw_err_i + self.kd[3]*(yaw_err-self.last_yaw_err)/self.sampletime
        self.last_yaw_err = yaw_err
        self.yaw_err_i = self.yaw_err_i + self.last_yaw_err
        self.yaw_err_i = np.sign(self.yaw_err_i) * min(abs(self.yaw_err_i), self.I_saturation)

        # Send to AirSim
        self.client.moveByVelocityAsync(v_cmd[0],v_cmd[1],v_cmd[2],2,drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom, 
            yaw_mode = airsim.YawMode(True, yaw_rate_cmd*180/np.pi))
        return v_cmd, yaw_rate_cmd

    def get_desire_states(self, point_id):
        # Init desire states
        p_des = np.array([0.0,0.0,0.0])
        v_des = np.array([0.0,0.0,0.0])
        yaw_des =  self.setpoints[point_id][3]
        # Get poly order
        n = self.matrix_x.shape[0] - 1
        # Form time array
        t_0= self.list_time[point_id]
        t_i = self.in_time
        t = t_i + t_0
        t_array_p = []
        t_array_v = []
        for i in range(n+1):
            t_array_p.append(pow(t,i))
            t_array_v.append((i*pow(t,i-1)))
        # Get desire states
        p_des[0] = np.dot(np.array(t_array_p),self.matrix_x[ : , point_id])
        p_des[1] = np.dot(np.array(t_array_p),self.matrix_y[ : , point_id])
        p_des[2] = np.dot(np.array(t_array_p),self.matrix_z[ : , point_id])
        v_des[0] = np.dot(np.array(t_array_v),self.matrix_x[ : , point_id])
        v_des[1] = np.dot(np.array(t_array_v),self.matrix_y[ : , point_id])
        v_des[2] = np.dot(np.array(t_array_v),self.matrix_z[ : , point_id])
        return p_des, v_des, yaw_des

    def get_desire_states_by_t_full(self):
        t = self.in_time
        # Init desire states
        p_des = np.array([0.0,0.0,0.0])
        v_des = np.array([0.0,0.0,0.0])
        # Get poly order
        n = self.matrix_x.shape[0] - 1
        # Form time array
        point_id = 0
        for i in range(len(self.list_time)-1):
            if  t >= self.list_time[i] and t < self.list_time[i+1]:
                point_id = i
                break
            else:
                pass
        t_array_p = []
        t_array_v = []
        for i in range(n+1):
            t_array_p.append(pow(t,i))
            t_array_v.append((i*pow(t,i-1)))
        # Get desire states
        p_des[0] = np.dot(np.array(t_array_p),self.matrix_x[ : , point_id])
        p_des[1] = np.dot(np.array(t_array_p),self.matrix_y[ : , point_id])
        p_des[2] = np.dot(np.array(t_array_p),self.matrix_z[ : , point_id])
        v_des[0] = np.dot(np.array(t_array_v),self.matrix_x[ : , point_id])
        v_des[1] = np.dot(np.array(t_array_v),self.matrix_y[ : , point_id])
        v_des[2] = np.dot(np.array(t_array_v),self.matrix_z[ : , point_id])
        if point_id<len(self.list_time)-2:
            yaw_des =  self.setpoints[point_id][3]
        return p_des, v_des, yaw_des

    def get_desire_states_by_t_single(self, point_id):
        # Init desire states
        p_des = np.array([0.0,0.0,0.0])
        v_des = np.array([0.0,0.0,0.0])
        yaw_des =  self.setpoints[point_id][3]
        # Get poly order
        n = self.matrix_x.shape[0] - 1
        # Form time array
        t = self.in_time
        t_array_p = []
        t_array_v = []
        for i in range(n+1):
            t_array_p.append(pow(t,i))
            t_array_v.append((i*pow(t,i-1)))
        # Get desire states
        p_des[0] = np.dot(np.array(t_array_p),self.matrix_x)
        p_des[1] = np.dot(np.array(t_array_p),self.matrix_y)
        p_des[2] = np.dot(np.array(t_array_p),self.matrix_z)
        v_des[0] = np.dot(np.array(t_array_v),self.matrix_x)
        v_des[1] = np.dot(np.array(t_array_v),self.matrix_y)
        v_des[2] = np.dot(np.array(t_array_v),self.matrix_z)
        return p_des, v_des, yaw_des

    def moveByMinSnapAsync(self,point_id):
        self.update_in_time()
        p_des, v_des, yaw_des = self.get_desire_states(point_id)
        self.traj_pid_update(p_des, v_des, yaw_des)
        self.des_log.append(p_des)

    def moveByMinSnapFullAsync(self):
        if self.in_time<self.list_time[-2]:
            self.update_in_time()
            p_des, v_des, yaw_des = self.get_desire_states_by_t_full()
            self.traj_pid_update(p_des, v_des, yaw_des)
            self.des_log.append(p_des)
            return True
        else:
            return False

    def moveByMinSnapSingleAsync(self,point_id):
        self.update_in_time()
        p_des, v_des, yaw_des = self.get_desire_states_by_t_single(point_id)
        self.traj_pid_update(p_des, v_des, yaw_des)
        self.des_log.append(p_des)
