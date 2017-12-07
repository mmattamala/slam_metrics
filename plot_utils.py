import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SE3UncertaintyLib as SE3Lib

# matplotlib config
plt.style.use('seaborn-paper')

_est_color = 'b'
_gt_color = 'k'

_en_labels={'ground_truth': 'Ground Truth', \
           'estimated': 'Estimated', \
           'traj': 'Trajectory', \
           'x_axis': 'x-axis', \
           'y_axis': 'y-axis', \
           'z_axis': 'z-axis' \
           }

_es_labels={'ground_truth': 'Ground Truth', \
           'estimated': 'Estimación', \
           'traj': 'Trayectoria', \
           'x_axis': 'eje x', \
           'y_axis': 'eje y', \
           'z_axis': 'eje z' \
           }

_lang_labels = {'EN' : _en_labels, \
                'ES' : _es_labels  \
               }


_fig_extension = 'pdf'
_lang = 'ES'
_labels = _lang_labels[_lang]

def set_file_extension(ext='pdf'):
    global _fig_extension
    _fig_extension = ext

def set_language(lang='EN'):
    global _lang
    global _labels
    global _lang_labels
    _lang = lang
    _labels = _lang_labels[_lang]

def plot_3d_xyz(gt_traj, est_traj, show=True, export=False, title='Plot XYZ', max_height=0.4):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    gt_x = gt_traj[0,:].A[0]
    gt_y = gt_traj[1,:].A[0]
    gt_z = gt_traj[2,:].A[0]

    est_x = est_traj[0,:].A[0]
    est_y = est_traj[1,:].A[0]
    est_z = est_traj[2,:].A[0]

    ax.plot(gt_x, gt_y, zs=gt_z, label=_labels['ground_truth'], color=_gt_color, zdir='z')
    ax.plot(est_x, est_y, zs=est_z, label=_labels['estimated'], zdir='z')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # set axes properties
    #ax.set_zlim([0, max_height])

    ax.legend(loc='upper right')
    plt.show()

def plot_3d_xyz_with_cov(gt_traj, est_traj, gt_cov=None, est_cov=None, cov_step=100, show=True, export=False, title='Plot XYZ', max_height=0.4):

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

    ax.plot(gt_x, gt_y, zs=gt_z, label=_labels['ground_truth'], color=_gt_color, zdir='z')
    ax.plot(est_x, est_y, zs=est_z, label=_labels['estimated'], zdir='z')

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

    SE3Lib.Visualize(gt_list, gt_cov_list, nsamples = 1000,)
    SE3Lib.Visualize(est_list, est_cov_list, nsamples = 1000)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    ax.legend(loc='upper right')
    plt.show()

def plot_2d_traj_xyz(gt_time, gt_traj, est_time, est_traj, save_fig=False, show_fig=False, percent=0.5):

    fig, axarr = plt.subplots(3, sharex=True)
    fig.tight_layout(pad=2.0, w_pad=0.0, h_pad=3.0)

    gt_time = np.array(gt_time) - gt_time[0]
    est_time = np.array(est_time) - est_time[0]

    gt_x = gt_traj[0,:].A[0]
    gt_y = gt_traj[1,:].A[0]
    gt_z = gt_traj[2,:].A[0]

    est_x = est_traj[0,:].A[0]
    est_y = est_traj[1,:].A[0]
    est_z = est_traj[2,:].A[0]

    axarr[0].plot(gt_time, gt_x, color=_gt_color, label=_labels['ground_truth'])
    axarr[0].plot(est_time, est_x, label=_labels['estimated'])
    axarr[0].set_title(_labels['traj'] + ' (' + _labels['x_axis'] + ')')
    axarr[0].legend(loc='upper right')
    gt_x_min = np.min(gt_x)
    gt_x_max = np.max(gt_x)
    gt_x_dx = (gt_x_max - gt_x_min)*percent
    axarr[0].set_ylim([gt_x_min - gt_x_dx, gt_x_max + gt_x_dx])

    axarr[1].plot(gt_time, gt_y, color=_gt_color, label=_labels['ground_truth'])
    axarr[1].plot(est_time, est_y, label=_labels['estimated'])
    axarr[1].set_title(_labels['traj'] + ' (' + _labels['y_axis'] + ')')
    axarr[1].legend(loc='upper right')
    gt_y_min = np.min(gt_y)
    gt_y_max = np.max(gt_y)
    gt_y_dx = (gt_y_max - gt_y_min)*percent
    axarr[1].set_ylim([gt_y_min - gt_y_dx, gt_y_max + gt_y_dx])

    axarr[2].plot(gt_time, gt_z, color=_gt_color, label=_labels['ground_truth'])
    axarr[2].plot(est_time, est_z, label=_labels['estimated'])
    axarr[2].set_title(_labels['traj'] + ' (' + _labels['z_axis'] + ')')
    axarr[2].legend(loc='upper right')
    gt_z_min = np.min(gt_z)
    gt_z_max = np.max(gt_z)
    gt_z_dx = (gt_z_max - gt_z_min)*percent
    axarr[2].set_ylim([gt_z_min - gt_z_dx, gt_z_max + gt_z_dx])

    if show_fig:
        plt.show()

    if save_fig:
        fig.savefig(_labels['traj'] + '-xyz' '.' + _fig_extension)
