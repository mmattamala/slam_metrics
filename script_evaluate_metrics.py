#!/usr/bin/python

"""

This script computes several SLAM metrics.
It is based on the TUM scripts

"""

import sys
import numpy as np
import argparse
import utils
import plot_utils
import slam_metrics
import SE3UncertaintyLib as SE3Lib

if __name__=="__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes different error metrics for SLAM from the ground truth trajectory and the estimated trajectory.
    ''')
    # Add argument options
    parser.add_argument('gt_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('est_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_pairs', help='maximum number of pose comparisons (default: 10000, set to zero to disable downsampling)', default=10000)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    parser.add_argument('--fixed_delta', help='only consider pose pairs that have a distance of delta delta_unit (e.g., for evaluating the drift per second/meter/radian)', action='store_true')
    parser.add_argument('--delta', help='delta for evaluation (default: 1.0)',default=1.0)
    parser.add_argument('--delta_unit', help='unit of delta (options: \'s\' for seconds, \'m\' for meters, \'rad\' for radians, \'f\' for frames; default: \'m\')',default='m')
    parser.add_argument('--alignment', help='type of trajectory alignment (options: \'man\' for manifold, \'horn\' for Horn\'s method; default: \'horn\')',default='horn')

    parser.add_argument('--compute_automatic_scale', help='ATE_Horn computes the absolute scale using the mod by Raul Mur', action='store_true')
    parser.add_argument('--show_plots', help='shows the trajectory plots', action='store_true')
    parser.add_argument('--no_metrics', help='not computes the metrics, used for plotting test only', action='store_true')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute will be printed)', action='store_true')

    #parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    #parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    #parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    args = parser.parse_args()

    # read files in TUM format or TUM modified format (with covariances)
    gt_dict  = utils.read_file_dict(args.gt_file)
    est_dict = utils.read_file_dict(args.est_file)

    # check file format
    gt_format = utils.check_valid_pose_format(gt_dict)
    est_format = utils.check_valid_pose_format(est_dict)

    # generate poses
    gt_poses, gt_cov = utils.convert_file_dict_to_pose_dict(gt_dict, file_format=gt_format)
    est_poses, est_cov = utils.convert_file_dict_to_pose_dict(est_dict, file_format=est_format)

    #for key in est_poses:
    #    print(est_poses[key][0:3,3])

    # apply scale
    gt_poses  = utils.scale_dict(gt_poses, scale_factor=1)
    gt_cov_   = utils.scale_dict(gt_cov, scale_factor=1, is_cov=True)
    est_poses = utils.scale_dict(est_poses, scale_factor=1)
    est_cov   = utils.scale_dict(est_cov, scale_factor=1, is_cov=True)

    # associate sequences according to timestamps
    gt_poses, est_poses = utils.associate_and_filter(gt_poses, est_poses, offset=float(args.offset), max_difference=float(args.max_difference))
    gt_cov, est_cov = utils.associate_and_filter(gt_cov, est_cov, offset=float(args.offset), max_difference=float(args.max_difference))

    # align poses
    #gt_poses_align_man, est_poses_align_man, T_align_man = utils.align_trajectories_manifold(gt_poses, est_poses, cov_est=est_cov, align_gt=False)
    #gt_poses_align_horn, est_poses_align_horn, T_align_horn = utils.align_trajectories_horn(gt_poses, est_poses, align_gt=False)
    gt_poses_align_first, est_poses_align_first = utils.align_trajectories_to_first(gt_poses, est_poses)

    gt_poses_align = gt_poses_align_first
    est_poses_align = est_poses_align_first


    #for key in est_poses_align_man:
    #    print(est_poses_align_man[key][0:3,3])

    if(not args.no_metrics):
        # Compute metrics
        # ATE (Absolute trajectory error)
        print('\nATE - Horn')
        ate_horn_error = slam_metrics.ATE_Horn(gt_poses_align, est_poses_align)
        slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0))

        print('\nATE - Horn - X')
        ate_horn_error = slam_metrics.ATE_Horn(gt_poses_align, est_poses_align, axes='X')
        slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0))

        print('\nATE - Horn - Y')
        ate_horn_error = slam_metrics.ATE_Horn(gt_poses_align, est_poses_align, axes='Y')
        slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0))

        print('\nATE - Horn - Z')
        ate_horn_error = slam_metrics.ATE_Horn(gt_poses_align, est_poses_align, axes='Z')
        slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0))



        # ATE (Absolute trajectory error, SE(3))
        print('\nATE - Manifold')
        ate_se3_error = slam_metrics.ATE_SE3(gt_poses_align,
                                             est_poses_align,
                                             offset=float(args.offset),
                                             max_difference=float(args.max_difference))
        slam_metrics.compute_statistics(np.linalg.norm(ate_se3_error[0:3,:], axis=0), variable='Translational', verbose=args.verbose)
        slam_metrics.compute_statistics(np.linalg.norm(ate_se3_error[3:6,:], axis=0), variable='Rotational', verbose=args.verbose)

        # RPE (Relative Pose Error)
        print('\nRPE - %s [%s]' % (args.delta, args.delta_unit))
        rpe_error, rpe_trans_error, rpe_rot_error, rpe_distance = slam_metrics.RPE(gt_poses_align,
                                                                   est_poses_align,
                                                                   param_max_pairs=int(args.max_pairs),
                                                                   param_fixed_delta=args.fixed_delta,
                                                                   param_delta=float(args.delta),
                                                                   param_delta_unit=args.delta_unit,
                                                                   param_offset=float(args.offset))

        slam_metrics.compute_statistics(np.linalg.norm(rpe_error[0:3,:], axis=0), variable='Translational', verbose=args.verbose)
        slam_metrics.compute_statistics(np.linalg.norm(rpe_error[3:6,:], axis=0), variable='Rotational', verbose=args.verbose)

        print('\nDDT')
        ddt = np.divide(rpe_error, rpe_distance)
        slam_metrics.compute_statistics(np.linalg.norm(ddt[0:3,:], axis=0), variable='Translational', verbose=args.verbose)
        slam_metrics.compute_statistics(np.linalg.norm(ddt[3:6,:], axis=0), variable='Rotational', verbose=args.verbose)


    if(args.show_plots):
        gt_data = gt_poses_align
        est_data = est_poses_align

        gt_stamps = list(gt_data.keys())
        gt_stamps.sort()
        est_stamps = list(est_data.keys())
        est_stamps.sort()

        #gt_t0 = gt_stamps[0]
        #est_t0 = est_stamps[0]

        #gt_T0 = np.linalg.inv(gt_data[gt_t0])
        #est_T0 = np.linalg.inv(est_data[est_t0])

        #gt_data  = dict( [(a, np.dot(gt_T0, gt_data[a])) for a in gt_data])
        #est_data  = dict( [(a, np.dot(est_T0, est_data[a])) for a in est_data])

        gt_xyz  = np.matrix([gt_data[a][0:3,3] for a in gt_data]).transpose()
        est_xyz  = np.matrix([est_data[a][0:3,3] for a in est_data]).transpose()

        gt_angles   = np.matrix([utils.rotm_to_rpy(gt_data[a][0:3,0:3]) for a in gt_data]).transpose()
        est_angles  = np.matrix([utils.rotm_to_rpy(est_data[a][0:3,0:3]) for a in est_data]).transpose()

        plot_utils.plot_2d_traj_xyz(gt_stamps, gt_xyz, est_stamps, est_xyz)
        #plot_utils.plot_2d_traj_xyz(gt_stamps, gt_angles, est_stamps, est_angles)
        #plot_utils.plot_3d_xyz(gt_xyz, est_xyz)
        #plot_utils.plot_3d_xyz_with_cov(gt_data, est_data, gt_cov=gt_cov, est_cov=est_cov)
        #plot_utils.plot_3d_xyz(gt_xyz, est_xyz)
