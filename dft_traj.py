import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from minsnap_traj import minimum_snap_traj, minimum_snap_traj_p2p, get_traj



def plot_3D_xyz(traj):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(traj[0], traj[1], traj[2],marker = 'x',color = 'red', s = 2 ,label = 'desire')
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()


def plot_1D_time(D3_traj,t_list):
    fig = plt.figure()
    plt.scatter(t_list[1:], D3_traj[0], marker = 'x', color = 'blue', s = 2, label = '0')
    plt.scatter(t_list[1:], D3_traj[1], marker = 'x', color = 'red', s = 2, label = '1')
    plt.scatter(t_list[1:], D3_traj[2], marker = 'x', color = 'green', s = 2, label = '2')
    plt.show()    


def differential_flatness_transform(p, v, a):
    psi = 0 # no psi plan
    g = np.array([0, 0, -9.8])
    x_c = np.array([np.cos(psi), np.sin(psi), 0])
    R_list = []
    for i in range(a.shape[1]):
        t = a[:,i] - g
        t_norm = np.linalg.norm(t)
        z_b = t/t_norm
        y_b = np.cross(z_b, x_c)
        x_b = np.cross(y_b, z_b)
        R = np.matrix([x_b, y_b, z_b])
        R_list.append(R)
    return R_list 

def simple_dft(a):
    psi = 0
    g = np.array([0, 0, -9.8])
    x_c = np.array([np.cos(psi), np.sin(psi), 0])
    t = a - g
    t_norm = np.linalg.norm(t)
    z_b = t/t_norm
    y_b = np.cross(z_b, x_c)
    x_b = np.cross(y_b, z_b)
    R = np.matrix([x_b, y_b, z_b])
    return R
    
if __name__ == "__main__":
    way_points = np.array([[0,0,0,0],[1.5,0.125,-0.75,0]])
    time_set = np.array([0,0.5])
    n_order = 5
    n_obj = 3
    v_i = [3,0,0,0]
    a_i = [0,0,0,0]
    v_e = [3,0.5,-5,0]
    a_e = [0,0,-20,0]
    sample_rate = 100
    Matrix_x, Matrix_y, Matrix_z = minimum_snap_traj_p2p(way_points, time_set, n_order, n_obj, v_i, a_i, v_e, a_e)
    p, v, a, t_list= get_traj(Matrix_x, Matrix_y, Matrix_z, time_set, sample_rate)
    R = differential_flatness_transform(np.array(p), np.array(v), np.array(a))
    print(p[1])
    plot_1D_time(a,t_list) 
    plot_3D_xyz(p)
    