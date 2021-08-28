from os import error
import numpy as np
from numpy.core.fromnumeric import shape
from scipy import sparse
import scipy.linalg
import math
import scipy.io as scio
import osqp
import matplotlib.pyplot as plt
import time

def t_n_list(t,n_order):
    t_array_p = []
    for i in range(n_order+1):
            t_array_p.append(pow(t,i))
    return t_array_p

def plot_xy(x_traj,y_traj,ref_x,ref_y):
    fig = plt.figure()
    plt.scatter(x_traj, y_traj,marker = 'x',color = 'blue', s = 2 ,label = 'state')
    plt.scatter(ref_x,ref_y,marker = 'x',color = 'red', s = 2 ,label = 'state')
    plt.show()

def gen_traj(Matrix_x,Matrix_y,time_set,n_order):
    p_x_traj = []
    p_y_traj = []
    k = 0
    i = 0
    for t in range(0,math.floor(time_set[-1]*100)+1):
        while True:
            if  t/100>=time_set[i] and  t/100<=time_set[i+1]:
                break
            else:
                k =k+1
                i = i+1
                break
        p_x_traj.append(np.dot(np.array(t_n_list(t/100,n_order)),Matrix_x[ : , k]))
        p_y_traj.append(np.dot(np.array(t_n_list(t/100,n_order)),Matrix_y[ : , k]))
    return p_x_traj,p_y_traj

def load_mat_data():
    # data = scio.loadmat('/run/user/1000/gvfs/afp-volume:host=CatDisk.local,user=a,volume=home/毛鹏达/test_mao.mat')
    data = scio.loadmat('/run/user/1000/gvfs/afp-volume:host=CatDisk.local,user=a,volume=home/毛鹏达/test_for_gao.mat')
    # print(data)
    return data

class MinSnap:
    def __init__(self) -> None:
        pass
    def minsnap_trajectory_single(self, way_points, time_set, n_order, n_obj, v_i, a_i, v_e, a_e):
        # Set init and end point
        p_i = way_points[0]
        p_e = way_points[-1]
         
        # Set poly num and coeffient num
        n_poly = len(way_points) - 1
        n_coef = n_order  + 1

        # Compute QP cost function matrix
        q_i_cost = []
        for i in range(n_poly):
            q_i_cost.append(self.compute_q(n_order+1, n_obj, time_set[i], time_set[i+1]))
        q_cost = scipy.linalg.block_diag(*q_i_cost)
        p_cost = np.zeros(q_cost.shape[0])

        # Set equality constraints
        A_eq = np.zeros((4*n_poly + 2, n_coef*n_poly))
        b_eq = np.zeros((4*n_poly + 2, 1))
        # Init and End constraints: 6
        A_eq[0:3,0:n_coef] = [self.compute_t_vec(time_set[0], n_order, 0), \
                                                    self.compute_t_vec(time_set[0], n_order, 1), \
                                                    self.compute_t_vec(time_set[0], n_order, 2)]
        A_eq[3:6,n_coef*(n_poly-1):n_coef*n_poly] = \
                                            [self.compute_t_vec(time_set[-1], n_order, 0), \
                                            self.compute_t_vec(time_set[-1], n_order, 1), \
                                            self.compute_t_vec(time_set[-1], n_order, 2)]
        # beq = np.vstack((beq,[p_i]))
        b_eq[0:6] = np.transpose(np.array([[p_i, v_i, a_i, p_e, v_e, a_e] ]))
        # Points constraints: n_poly - 1
        n_eq = 6
        for i in range(0, n_poly-1):
            A_eq[n_eq, n_coef*(i+1):n_coef*(i+2)] = self.compute_t_vec(time_set[i+1], n_order, 0)
            b_eq[n_eq] = np.array([[way_points[i+1]]])
            n_eq = n_eq+1
        # Continous constraints: (n_poly - 1)*3
        for i in range(0, n_poly-1):
            t_vec_p = self.compute_t_vec(time_set[i+1], n_order, 0)
            t_vec_v = self.compute_t_vec(time_set[i+1], n_order, 1)
            t_vec_a = self.compute_t_vec(time_set[i+1], n_order, 2)
            # t_vec_p
            # A_eq[n_eq,n_coef*i:n_coef*(i+2)]=[t_vec_p, -t_vec_p]
            A_eq[n_eq,n_coef*i:n_coef*(i+1)]=t_vec_p
            A_eq[n_eq,n_coef*(i+1):n_coef*(i+2)]=-t_vec_p
            n_eq=n_eq+1
            A_eq[n_eq,n_coef*i:n_coef*(i+1)]=t_vec_v
            A_eq[n_eq,n_coef*(i+1):n_coef*(i+2)]=-t_vec_v
            n_eq=n_eq+1
            A_eq[n_eq,n_coef*i:n_coef*(i+1)]=t_vec_a
            A_eq[n_eq,n_coef*(i+1):n_coef*(i+2)]=-t_vec_a
            n_eq=n_eq+1
        
        # Set inequality constraints
        A_ieq = np.zeros((0, n_coef*n_poly))  # inequality constraints A matrix
        b_ieq = np.zeros((0, 1))

        # Solve qp problem
        A_eq_ieq = np.vstack((A_eq, -1 * A_eq))
        b_eq_ieq = np.vstack((b_eq, -1 * b_eq))
        # A_ieq = np.vstack((A_ieq, A_eq_ieq))
        # b_ieq = np.vstack((b_ieq, b_eq_ieq))
        A_ieq = A_eq_ieq
        b_ieq = b_eq_ieq


        m = osqp.OSQP()
        m.setup(P=sparse.csc_matrix(q_cost), q=None, l=None, A=sparse.csc_matrix(A_ieq), u=b_ieq, verbose=False)
        results = m.solve()
        x = results.x
        return x

    def compute_q(self, n, r , t_i, t_e):
        q = np.zeros((n,n))
        for i in range(r, n):
            for j in range(i, n):
                k1 = i - r 
                k2 = j - r 
                k = k1 + k2 +1
                q[i, j] = (math.factorial(i)/math.factorial(k1)) * (math.factorial(j)/math.factorial(k2)) * (pow(t_e,k)-pow(t_i,k)) / k
                q[j,i] = q[i,j]
        return q

    def compute_t_vec(self, t, n_order, k):
        t_vector = np.zeros(n_order+1)
        for i in range(k, n_order+1):
            t_vector[i] =  (math.factorial(i)/math.factorial(i-k)) * pow(t, i-k)
        return t_vector

    def time_arrange(self, way_points, T):
        dist_vec = way_points[:, 1:] - way_points[:, :-1]
        dist = []
        for i in range(dist_vec.shape[1]):
            dist_i = 0
            for j in range(dist_vec.shape[0]):
                dist_i += dist_vec[j][i] **2
            dist_i = dist_i **0.5
            dist.append(dist_i)
        k = T/sum(dist)
        time_set = [0]
        time_i = np.array(dist)*k
        for i in range(1,len(time_i)):
            time_i[i] = time_i[i] + time_i[i-1]
        time_set.extend(time_i)
        return time_set

def minimum_snap_traj():
    start_t = time.time()
    # ref = load_mat_data()
    traj = MinSnap()
    # way_points = [[0,0,0],[15.8, -22.6, -4.8], [22.8, -45.2, -4.5], 
    #                       [19.8, -65.4, -3.7], [ 8.0, -82.2, -2.5],
    #                       [-11.3,-93.0, -2.9], [-29.9,-98.6, -4.7],
    #                       [-52.1,-103.0,-5.7],[-62.9, -102.3, -2]]
    way_points = [[0,0,0],[15.8, -22.6, -4.8+1], [22.8, -45.2, -4.5+1], 
                          [19.8, -65.4, -3.7+1], [ 8.0, -82.2, -2.5+1],
                          [-11.3,-93.0, -2.9+1], [-29.9,-98.6, -4.7+1],
                          [-52.1,-103.0,-5.7+1],[-62.9, -102.3, -5.7+1]]
    way_points_n = np.array(way_points)
    way_points_n = np.transpose(way_points_n)
    T = 25
    n_order = 8
    n_obj = 3
    v_i = [0,0,0,0]
    a_i = [0,0,0,0]
    v_e = [0,0,0,0]
    a_e = [0,0,0,0]
    time_set = traj.time_arrange(way_points_n,T)
    p_x = traj.minsnap_trajectory_single(way_points_n[0,:], time_set, n_order, n_obj, v_i[0], a_i[0], v_e[0], a_e[0])
    p_y= traj.minsnap_trajectory_single(way_points_n[1,:], time_set, n_order, n_obj, v_i[1], a_i[1], v_e[1], a_e[1])
    p_z = traj.minsnap_trajectory_single(way_points_n[2,:], time_set, n_order, n_obj, v_i[2], a_i[2], v_e[2], a_e[2])
    # p_x= traj.minsnap_trajectory_single(way_points_n[0,:], ref["ts"][0], n_order, n_obj, v_i[0], a_i[0], v_e[0], a_e[0])
    # p_y= traj.minsnap_trajectory_single(way_points_n[1,:], ref["ts"][0], n_order, n_obj, v_i[1], a_i[1], v_e[1], a_e[1])
    end_t = time.time()
    Matrix_x = np.transpose(p_x.reshape(8,n_order+1))
    Matrix_y = np.transpose(p_y.reshape(8,n_order+1))
    Matrix_z = np.transpose(p_z.reshape(8,n_order+1))
    # Matrix_x_ref = ref["polys_x"]
    # Matrix_y_ref = ref["polys_y"]
    p_x_traj,p_y_traj = gen_traj(Matrix_x,Matrix_y,time_set,n_order)
    # ref_x,ref_y = gen_traj(Matrix_x_ref,Matrix_y_ref,ref['ts'][0],7)
    # plot_xy(p_x_traj,p_y_traj,ref_x,ref_y)
    # print(end_t-start_t)
    return time_set, Matrix_x, Matrix_y, Matrix_z

