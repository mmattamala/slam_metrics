import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_est_color = 'b'
_gt_color = 'r'

def plot_3d_xyz(gt_traj, est_traj, gt_color=_gt_color, est_color=_est_color, show=True, export=False, title='Plot XYZ', max_height=0.4):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    gt_x = gt_traj[0,:].A[0]
    gt_y = gt_traj[1,:].A[0]
    gt_z = gt_traj[2,:].A[0]

    est_x = est_traj[0,:].A[0]
    est_y = est_traj[1,:].A[0]
    est_z = est_traj[2,:].A[0]

    ax.plot(gt_x, gt_y, zs=gt_z, label='Ground Truth', color=gt_color, zdir='z')
    ax.plot(est_x, est_y, zs=est_z, label='Estimated', color=est_color, zdir='z')

    #ax.xlabel('x [m]')
    #ax.ylabel('y [m]')
    #ax.zlabel('z [m]')
    # set axes properties
    #ax.set_zlim([0, max_height])

    ax.legend()
    plt.show()

def plot_2d_traj_xyz(gt_time, gt_traj, est_time, est_traj, gt_color=_gt_color, est_color=_est_color):

    fig, axarr = plt.subplots(3, sharex=True)

    gt_time = np.array(gt_time) - gt_time[0]
    est_time = np.array(est_time) - est_time[0]

    gt_x = gt_traj[0,:].A[0]
    gt_y = gt_traj[1,:].A[0]
    gt_z = gt_traj[2,:].A[0]

    est_x = est_traj[0,:].A[0]
    est_y = est_traj[1,:].A[0]
    est_z = est_traj[2,:].A[0]

    axarr[0].plot(gt_time, gt_x, color=gt_color, label='Ground Truth')
    axarr[0].plot(est_time, est_x, color=est_color, label='Estimated')
    axarr[0].set_title('x trajectory')
    axarr[0].legend()

    axarr[1].plot(gt_time, gt_y, color=gt_color, label='Ground Truth')
    axarr[1].plot(est_time, est_y, color=est_color, label='Estimated')
    axarr[1].set_title('y trajectory')
    axarr[1].legend()

    axarr[2].plot(gt_time, gt_z, color=gt_color, label='Ground Truth')
    axarr[2].plot(est_time, est_z, color=est_color, label='Estimated')
    axarr[2].set_title('z trajectory')
    axarr[2].legend()

    plt.show()
