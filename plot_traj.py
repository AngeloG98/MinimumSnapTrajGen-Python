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


def plot_1D_time(D1_traj,t_list):
    fig = plt.figure()
    plt.scatter(t_list[1:], D1_traj, marker = 'x', color = 'blue', s = 2, label = 'state')
    plt.show()    

if __name__ == "__main__":
    way_points = np.array([[0,0,1.25,0],[1.5,0,0,0]])
    time_set = np.array([0,0.5])
    n_order = 5
    n_obj = 3
    v_i = [3,0,0,0]
    a_i = [0,0,-30,0]
    v_e = [3,0,-5,0]
    a_e = [0,0,-30,0]
    sample_rate = 100
    Matrix_x, Matrix_y, Matrix_z = minimum_snap_traj_p2p(way_points, time_set, n_order, n_obj, v_i, a_i, v_e, a_e)
    p, v, a, t_list= get_traj(Matrix_x, Matrix_y, Matrix_z, time_set, sample_rate)
    print(a[2])
    # plot_1D_time(v[2],t_list) 
    plot_3D_xyz(p)