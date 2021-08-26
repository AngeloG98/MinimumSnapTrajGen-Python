import airsim
import time
import numpy as np
import cv2
import multiprocessing

def objectDetect(circle_xyr):
    """detect objects, return ex, ey"""
    client = airsim.MultirotorClient('127.0.0.1')
    client.confirmConnection()
    print("Multiprocessing objectDetect Start!")

    while True:
        responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8) #get numpy array
        img_bgr = img1d.reshape(responses[0].height, responses[0].width, 3) #reshape array to 3 channel image array H X W X 3
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        low_range = np.array([2,80,80])
        high_range = np.array([12,220,150])
        img_th = cv2.inRange(img_hsv, low_range, high_range)

        rt = circleDetect(img_th)
        circle_xyr[0], circle_xyr[1], circle_xyr[2] = rt[0], rt[1], rt[2]

def circleDetect(img_th):
    """Hough Circle detect"""
    circles = cv2.HoughCircles(img_th, cv2.HOUGH_GRADIENT, 2, 
                                    30, param1=None, param2=40, minRadius=15, maxRadius=150)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        obj = circles[0, 0]
        cv2.circle(img_th, (obj[0], obj[1]), obj[2], (0,255,0), 2)
        cv2.circle(img_th, (obj[0], obj[1]), 2, (0,255,255), 3)

        cv2.imshow("img_th", img_th)
        cv2.waitKey(1)
        height, width = img_th.shape
        return obj[0]-width/2, obj[1]-height/2, obj[2]
    else:
        return -1,-1,-1


class airsim_client:
    def __init__(self, ip_addr='127.0.0.1') -> None:
        print("Try to connect {}...".format(ip_addr))
        self.client = airsim.MultirotorClient(ip_addr)
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.new_task = True
        self.th_loss = 20
        self.th_through = 20
        self.th_pos = 5
        self.th_R_min = 15
        self.th_R_max = 100

        self.k_position = 1.0
        self.k_yaw = 0.5
        self.cx = 160
        self.cy = 120
        self.f = 120
        self.v_position = 5
        self.v_through = 3
        self.k_ibvs_hor = 0.025
        self.k_ibvs_ver = 0.025

        self.n_cc = np.array([0,0,1])
        self.R_cb = np.array([[0,0,1],\
                             [1,0,0],\
                             [0,1,0]])
        self.R_be = np.identity(3)

        self.setpoints = [[15.8, -22.6, -4.8, np.deg2rad(-90)], [22.8, -45.2, -4.5, np.deg2rad(-90)], 
                          [19.8, -65.4, -3.7, np.deg2rad(-110)], [ 8.0, -82.2, -2.5, np.deg2rad(-120)],
                          [-11.3,-93.0, -2.9, np.deg2rad(-170)], [-29.9,-98.6, -4.7, np.deg2rad(-180)],
                          [-52.1,-103.0,-5.7, np.deg2rad(-180)]]
        self.setpoint_land = [-62.9, -102.3, -2]

        self.circle_xyr = multiprocessing.Manager().list([-1, -1, -1])
        self.p = multiprocessing.Process(target=objectDetect, args=(self.circle_xyr, ))
        self.p.start()

    def __del__(self):
        self.p.terminate()

    def moveByVelocityYawrateAsync(self, vx, vy, vz, wyaw):
        """
        - function: 线速度加偏航角速度接口
        - params:
            - vx (float): desired velocity in world (NED) X axis
            - vy (float): desired velocity in world (NED) Y axis
            - vz (float): desired velocity in world (NED) Z axis
            - wyaw (float): desired yaw rate
        - return:
            - 异步执行
        """
        self.client.moveByVelocityAsync(vx, vy, vz, 2, 
            drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom, 
            yaw_mode = airsim.YawMode(True, wyaw))

    def moveByVelocityYawrateBodyFrameAsync(self, vx, vy, vz, wyaw):
        """
        - function: 线速度加偏航角速度接口
        - params:
            - vx (float): desired velocity in the X axis of the vehicle's local NED frame.
            - vy (float): desired velocity in the Y axis of the vehicle's local NED frame.
            - vz (float): desired velocity in the Z axis of the vehicle's local NED frame.
            - wyaw (float): desired yaw rate
        - return:
            - 异步执行
        """
        self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, 2, 
            drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom, 
            yaw_mode = airsim.YawMode(True, wyaw))

    def moveToPositionOnceAsync(self, x, y, z, yaw_d, velocity=3):
        """
        - function: 向world NED坐标系下期望位置和偏航角飞行
        - params:
            - x (float): desired position in world (NED) X axis
            - y (float): desired position in world (NED) Y axis
            - z (float): desired position in world (NED) Z axis
            - yaw_d (float): desired yaw
            - velocity (float): velocity saturation
        - return:
            - 异步执行
        - usage:
            while d < th: update p_d, p  moveToPositionOnceAsync(x, y, z, yaw, v)  time.sleep(t)
        """
        p_d = np.array([x, y, z])
        state = self.client.getMultirotorState()
        quaternionr = state.kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(quaternionr)
        position = state.kinematics_estimated.position
        p = np.array([position.x_val, position.y_val, position.z_val])
        cmd_v = self.saturation(self.k_position * (p_d - p), velocity)
        cmd_yaw = self.k_yaw * self.minAngleDiff(yaw_d, yaw) * 180/np.pi
        self.moveByVelocityYawrateAsync(*cmd_v, cmd_yaw)

    def moveToPosition(self, x, y, z, yaw_d=None, velocity=3):
        p_d = np.array([x, y, z])
        position = self.client.getMultirotorState().kinematics_estimated.position
        p = np.array([position.x_val, position.y_val, position.z_val])
        while np.linalg.norm(p_d - p) > 0.5:
            position = self.client.getMultirotorState().kinematics_estimated.position
            p = np.array([position.x_val, position.y_val, position.z_val])
            quaternionr = self.client.getMultirotorState().kinematics_estimated.orientation
            _, _, yaw = airsim.to_eularian_angles(quaternionr)
            if yaw_d is None:
                self.moveToPositionOnceAsync(*p_d, yaw, velocity)
            else:
                self.moveToPositionOnceAsync(*p_d, self.k_yaw * self.minAngleDiff(yaw_d, yaw) * 180/np.pi, velocity)
            time.sleep(0.02)

    def moveByVisualServoOnceAsync(self, ex_i, ey_i, r_i):
        self.moveByVelocityYawrateBodyFrameAsync(self.v_through, self.k_ibvs_hor*ex_i, self.k_ibvs_ver*ey_i, 0)

    def moveByRotateVisualServoOnceAsync(self, ex_i, ey_i, r_i):
        #calacute nc,the first idex(c:camera,b:body,e:earth) represent the frmae, the second idex(c,o) represent the camera or obstacle
        quaternionr = self.client.getMultirotorState().kinematics_estimated.orientation
        q0, q1, q2, q3 = quaternionr.w_val, quaternionr.x_val, quaternionr.y_val, quaternionr.z_val
        self.R_be = np.array([[q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                      [2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],
                      [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2-q1**2-q2**2+q3**2]])


        n_bc = self.R_cb.dot(self.n_cc)
        n_ec = self.R_be.dot(n_bc)
        
        #calacute the no
        n_co = np.array([ex_i, ey_i, self.f], dtype=np.float64)
        n_co /= np.linalg.norm(n_co)
        n_bo = self.R_cb.dot(n_co)
        n_eo = self.R_be.dot(n_bo)

        cos_beta = n_bo.dot(n_bc)
        v_b = n_bo*cos_beta - n_bc
        
        # v_m[1] = v_b[1] * 0.1/(1.01-cos_beta) + self.sat(self.cnt * 0.1,10)
        v_m = np.array([0., 0., 0.])
        # case1: (0.02, 3, 10)
        # case2: (0.04, 3, 12)
        v_m[0] = 5
        v_m[1] = 6*v_b[1]
        v_m[2] = 7*v_b[2]

        v = self.saturation(v_m, 8)
        yaw_rate = 0.002*(ex_i)
        
        # print("v_b: {}\nv_m: {}\nv: {}".format(v_b, v_m, v))
        # print("yaw_rate: {}".format(yaw_rate))
        self.moveByVelocityYawrateBodyFrameAsync(*v, yaw_rate)

    def saturation(self, a, maxv):
        n = np.linalg.norm(a)
        if n > maxv:
            return a / n * maxv
        else:
            return a

    def minAngleDiff(self, a, b):
        diff = a - b
        if diff < 0:
            diff += 2*np.pi
        if diff < np.pi:
            return diff
        else:
            return diff - 2*np.pi

    def task_takeoff(self):
        self.client.armDisarm(True)
        self.client.takeoffAsync()#.join()
        time.sleep(0.5)

    def task_cross_circle(self, circle_id):
        cnt_loss = 0
        cnt_through = 0
        last_R = 0
        while True:
            # print("R: {}".format(self.circle_xyr[2]))
            # 大于一定距离先靠近
            state = self.client.getMultirotorState()
            # vel = state.kinematics_estimated.linear_velocity
            # print(np.linalg.norm([vel.x_val, vel.y_val, vel.z_val]))
            position = state.kinematics_estimated.position
            p = np.array([position.x_val, position.y_val, position.z_val])
            dis = np.linalg.norm(np.array(self.setpoints[circle_id][:3]) - p)
            # print(dis)
            # 新任务按照预设点飞行
            if self.new_task:
                self.moveToPositionOnceAsync(*self.setpoints[circle_id], self.v_position)
                time.sleep(0.02)

            # circle_xyr = self.objectDetect()
            # print(circle_xyr)
            # 半径大于阈值就直接飞，防止太近误检测
            if self.circle_xyr[2] > self.th_R_max and self.circle_xyr[2] > last_R:
                cnt_through = 0
            last_R = self.circle_xyr[2]
            if self.circle_xyr[2] > self.th_R_max and cnt_through == 0:
                cnt_through += 1
            elif cnt_through > 0:
                cnt_through += 1
                time.sleep(0.02)
                # print("through", cnt_through)
                if cnt_through > self.th_through:
                    self.new_task = True
                    break
                continue
            # 如果找到了目标，按照视觉饲服飞行
            if self.circle_xyr[2] > 0 and self.circle_xyr[2] > self.th_R_min:
                self.new_task = False
                cnt_loss = 0
                # self.moveByVisualServoOnceAsync(*self.circle_xyr)
                self.moveByRotateVisualServoOnceAsync(*self.circle_xyr)
                time.sleep(0.02)
                # print(circle_xyr)
            # 如果没找到目标，但是曾经找到过，而且丢失次数小于阈值，按照上次指令飞
            elif self.circle_xyr[2] <= 0 and cnt_loss < self.th_loss and self.new_task == False:
                cnt_loss += 1
                time.sleep(0.02)
                # print(cnt_loss)
            # 如果没找到目标，但是曾经找到过，而且丢失次数大于阈值，进入下一任务（环）
            elif self.circle_xyr[2] <= 0 and cnt_loss >= self.th_loss and self.new_task == False:
                cnt_loss = 0
                self.new_task = True
                # print("break")
                break
            else:
                pass
                # print("pass")

    
    def task_land(self):
        self.moveToPosition(*self.setpoint_land, None, 10)
        self.client.armDisarm(False)

    def begin_task(self):
        # airsim.wait_key('Press any key to takeoff.')
        print("=========================")
        print("Taking off...")
        start_time = time.time()
        self.task_takeoff()

        for circle_id in range(len(self.setpoints)):
            # time.sleep(3)
            print("=========================")
            print("Now try to pass circle {}...".format(circle_id+1))
            self.task_cross_circle(circle_id)
        
        print("=========================")
        print("Landing...")
        self.task_land()

        print("=========================")
        print("Task is finished.")
        print("Time: {}".format(time.time() - start_time))

if __name__ == "__main__":
    client = airsim_client('127.0.0.1')
    client.begin_task()