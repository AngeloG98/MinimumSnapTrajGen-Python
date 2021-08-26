import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np

def plot_log(des, state):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(des[:, 0], des[:, 1], -des[:, 2],marker = 'x',color = 'red', s = 2 ,label = 'desire')
    ax.scatter(state[:, 0], state[:, 1], -state[:, 2],marker = 'x',color = 'blue', s = 2 ,label = 'state')
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()

def read_log():
    des = np.load("./data_log/des.npy")
    state =  np.load("./data_log/state.npy")
    return des, state

if __name__ == "__main__":
    des, state = read_log()
    plot_log(des, state)