import time
import airsim
from airsim.types import KinematicsState
import numpy as np
import scipy.io as scio
from airsim.utils import wait_key
from numpy.lib.shape_base import expand_dims
from minsnap_traj import minimum_snap_traj

class traj_follow_ctrl:
    def __init__(self, client) -> None:
        ## Connect AirSim
        self.client = client
        self.setpoints = [[15.8, -22.6, -4.8, np.deg2rad(-90)], [22.8, -45.2, -4.5, np.deg2rad(-90)], 
                          [19.8, -65.4, -3.7, np.deg2rad(-100)], [ 8.0, -82.2, -2.5, np.deg2rad(-120)],
                          [-11.3,-93.0, -2.9, np.deg2rad(-170)], [-29.9,-98.6, -4.7, np.deg2rad(-170)],
                          [-52.1,-103.0,-5.7, np.deg2rad(-180)]]
 
        # Control Parameters
        self.kp = [2, 2 ,3, 2]# in order x, y, z, yaw
        self.ki =  [0, 0, 0.1, 0]
        self.kd = [0, 0, 1, 0]
        self.sampletime = 0.01
        self.des_log = []
        # self.matrix_x = np.array([[0,-55.105701385464954,12.945774603093204,588.87087005129706,-3851.5064082979861,10986.541327370745,-28147.710205135085,4708.6455000657625],[0,38.263433183038273,1.2147075991010898,-222.78885312153673,1149.689655380057,-2880.1323826417038,7023.7409148539946,-13603.927183202719],[0,-8.5154247035497317,-0.44737336356082247,34.410383238278854,-135.04730128001074,296.39551933098232,-665.89336942649038,1833.0645945989506],[0.66497669651840918,0.99940150064750455,0.12091798338131673,-2.5918361602085067,7.8559381429163384,-14.969258784333704,30.453692708745027,-94.380839325880345],[-0.16292454854413715,-0.058427708118262665,-0.010601261030457367,0.09498044947383294,-0.226694189810487,0.3712186802036801,-0.67803925364796347,2.1677834054403777],[0.011060465407887647,0.0013198345926718736,0.00027832770496465597,-0.0013657597607970046,0.0025911073470023119,-0.0036224009642652926,0.0059110663213108629,-0.018684516383628034]])
        # self.matrix_y = np.array([[0,-29.44479887120044,301.53582103136938,-660.20370060227617,1278.5138740107943,-6388.631840238535,23887.148291558417,-14155.121493635737],[0,29.560032983878788,-150.63071695371312,224.71611180238821,-363.888769192389,1530.62213701462,-4903.9447705832681,3968.8969069721124],[0,-11.870275352117805,27.369152117115746,-31.225391543255011,40.253763917224205,-146.90520235012735,399.49422986086631,-379.89440188330661],[-0.60588492781078773,1.7774556919665083,-2.4950501752742071,2.0783716717815643,-2.2616237880148531,6.9788327771704743,-16.194910288596503,16.734902852468892],[0.11700277607795197,-0.12226352690698684,0.11033813543914588,-0.068139929090306042,0.063611068884498967,-0.16439747440005451,0.32650284986467071,-0.35058388013593533],[-0.00644127355636818,0.0031668140628699441,-0.001898461317575522,0.00088751896070455346,-0.00071227180087007425,0.0015371864388707061,-0.0026181498681293429,0.0028403931331428419]])
        # self.matrix_z = np.array([[0,-10.519017827466135,135.99916033741627,-431.41729446622605,1257.2400775656777,-8361.4305296071161,42827.181332762077,-150687.81507343549],[0,10.560497775162075,-69.207370002673684,152.19729854194574,-360.5711233543401,2005.6472981422589,-8796.2029076669642,31507.546135915243],[0,-4.2407205818049762,13.130142500960414,-21.424514692698338,40.852811868986954,-191.93546912674523,719.40992506433076,-2555.3266647952455],[-0.1693986106424521,0.6820639332597408,-1.2093303799588075,1.4870679405181229,-2.294753666759322,9.154055150371379,-29.272350702354583,101.29180408693759],[0.036913159157471356,-0.048566143758805962,0.054404324860851234,-0.050799379947476589,0.064027717344126045,-0.21744947611764032,0.59228243296941052,-1.9726583236768651],[-0.0021487236893290422,0.0012838224785236445,-0.00095852720423513719,0.00068333500915357338,-0.00071124173766409175,0.0020563447641736327,-0.0047656565396717249,0.01515340524154303]])
        # self.list_time = [0,4.9805187538601272,9.1841511629217223,12.815550986844411,16.469257354941373,20.402128984925191,23.868043448176632,27.892731405893002,29.999999999999996]
        # self.load_mat_data()
        self.gen_traj()
        self.reset()

    def load_mat_data(self):
        # data = scio.loadmat('/run/user/1000/gvfs/afp-volume:host=CatDisk.local,user=a,volume=home/毛鹏达/test_mao.mat')
        data = scio.loadmat('stable_traj/test_mao_35_stable.mat')
        print(data)
        self.list_time = data['ts'][0]
        self.matrix_x = data['polys_x']
        self.matrix_y = data['polys_y']
        self.matrix_z = data['polys_z']

    def gen_traj(self):
        self.list_time ,self.matrix_x , self.matrix_y ,self.matrix_z = minimum_snap_traj()
        
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
            v_cmd[i] = (v_des[i]-vel_vec[i]) + self.kp[i]*p_err[i] + self.ki[i]*self.p_err_i[i] + self.kd[i]*(p_err[i]-self.last_p_err[i])/self.sampletime
        self.last_p_err = p_err
        self.p_err_i = self.p_err_i + self.last_p_err
        for i in range(3):
            self.p_err_i[i] = max(self.p_err_i[i],0.2)
        # Yaw control
        yaw_rate_cmd = 0.0
        yaw_err = self.minAngleDiff(yaw_des, yaw_ref)
        yaw_rate_cmd = self.kp[3]*yaw_err + self.ki[3]*self.yaw_err_i + self.kd[3]*(yaw_err-self.last_yaw_err)/self.sampletime
        self.last_yaw_err = yaw_err
        self.yaw_err_i = self.yaw_err_i + self.last_yaw_err

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

    def moveByMinSnapAsync(self,point_id):
        self.update_in_time()
        p_des, v_des, yaw_des = self.get_desire_states(point_id)
        self.traj_pid_update(p_des, v_des, yaw_des)
        self.des_log.append(p_des)


