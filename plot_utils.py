import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SE3UncertaintyLib as SE3Lib

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

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    #ax.xlabel('x [m]')
    #ax.ylabel('y [m]')
    #ax.zlabel('z [m]')
    # set axes properties
    #ax.set_zlim([0, max_height])

    ax.legend()
    plt.show()

def plot_3d_xyz_with_cov(gt_traj, est_traj, gt_cov=None, est_cov=None, cov_step=100, gt_color=_gt_color, est_color=_est_color, show=True, export=False, title='Plot XYZ', max_height=0.4):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    gt_xyz  = np.matrix([gt_traj[a][0:3,3] for a in gt_traj]).transpose()
    est_xyz  = np.matrix([est_traj[a][0:3,3] for a in est_traj]).transpose()

    gt_x = gt_xyz[0,:].A[0]
    gt_y = gt_xyz[1,:].A[0]
    gt_z = gt_xyz[2,:].A[0]

    est_x = est_xyz[0,:].A[0]
    est_y = est_xyz[1,:].A[0]
    est_z = est_xyz[2,:].A[0]

    ax.plot(gt_x, gt_y, zs=gt_z, label='Ground Truth', color=gt_color, zdir='z')
    ax.plot(est_x, est_y, zs=est_z, label='Estimated', color=est_color, zdir='z')

    est_list = []
    est_cov_list = []
    gt_list = []
    gt_cov_list = []

    i = 0
    for key in est_traj:
        if not (i%cov_step):
            est_list.append(est_traj[key])
            est_cov_list.append(est_cov[key])
            i = i + 1
    i = 0
    for key in gt_traj:
        if not (i % cov_step):
            gt_list.append(gt_traj[key])
            gt_cov_list.append(gt_cov[key])
            i = i + 1

    SE3Lib.Visualize(gt_list, gt_cov_list, nsamples = 1000, plot_color=gt_color )
    SE3Lib.Visualize(est_list, est_cov_list, nsamples = 1000, plot_color=est_color )


    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

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
    axarr[0].set_title('Trajectory (x-axis)')
    axarr[0].legend()

    axarr[1].plot(gt_time, gt_y, color=gt_color, label='Ground Truth')
    axarr[1].plot(est_time, est_y, color=est_color, label='Estimated')
    axarr[1].set_title('Trajectory (y-axis)')
    axarr[1].legend()

    axarr[2].plot(gt_time, gt_z, color=gt_color, label='Ground Truth')
    axarr[2].plot(est_time, est_z, color=est_color, label='Estimated')
    axarr[2].set_title('Trajectory (z-axis)')
    axarr[2].legend()

    plt.show()
