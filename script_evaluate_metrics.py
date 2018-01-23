#!/usr/bin/python

"""

This script computes several SLAM metrics.
It is based on the TUM scripts

"""

import sys
import numpy as np
import argparse
import csv
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
    parser.add_argument('--offset_initial', help='time offset to start the sequence analysis (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_pairs', help='maximum number of pose comparisons (default: 10000, set to zero to disable downsampling)', default=10000)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    parser.add_argument('--fixed_delta', help='only consider pose pairs that have a distance of delta delta_unit (e.g., for evaluating the drift per second/meter/radian)', action='store_true')
    parser.add_argument('--delta', help='delta for evaluation (default: 1.0)',default=1.0)
    parser.add_argument('--delta_unit', help='unit of delta (options: \'s\' for seconds, \'m\' for meters, \'rad\' for radians, \'f\' for frames; default: \'m\')',default='m')
    parser.add_argument('--alignment', help='type of trajectory alignment (options: \'first\' for first pose, \'manifold\' for manifold, \'horn\' for Horn\'s method; default: \'none\')',default='none')
    parser.add_argument('--plot_lang', help='language used to show the plots; default: \'EN\')',default='EN')
    parser.add_argument('--plot_format', help='format to export the plots; default: \'pdf\')',default='pdf')
    parser.add_argument('--gt_static_transform', help='a static transform for ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)', default=None)
    parser.add_argument('--est_static_transform', help='a static transform for ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)', default=None)

    parser.add_argument('--ate_manifold', help='computes the error using ATE on the manifold', action='store_true')
    parser.add_argument('--rpe', help='computes RPE', action='store_true')
    parser.add_argument('--ddt', help='computes DDT', action='store_true')
    parser.add_argument('--automatic_scale', help='ATE_Horn computes the absolute scale using the mod by Raul Mur', action='store_true')
    parser.add_argument('--show_plots', help='shows the trajectory plots', action='store_true')
    parser.add_argument('--save_plots', help='saves the trajectory plots', action='store_true')
    parser.add_argument('--no_metrics', help='not computes the metrics, used for plotting test only', action='store_true')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute will be printed)', action='store_true')
    parser.add_argument('--ignore_timestamp_match', help='ignores the timestamp to associate the sequences', action='store_true')
    parser.add_argument('--recommended_offset', help='ignores the given offset and uses the recommended offset obtained from the sequences', action='store_true')
    parser.add_argument('--save_translations', help='saves the translations in a csv file', action='store_true')
    parser.add_argument('--save_statistics', help='saves the statistics summary in a csv file', action='store_true')

    #parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    #parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    #parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    args = parser.parse_args()

    # configure the plotting stuff
    plot_utils.set_language(lang=args.plot_lang)
    plot_utils.set_file_extension(ext=args.plot_format)

    # read files in TUM format or TUM modified format (with covariances)
    gt_dict  = utils.read_file_dict(args.gt_file)
    est_dict = utils.read_file_dict(args.est_file)

    # check file format
    gt_format = utils.check_valid_pose_format(gt_dict)
    est_format = utils.check_valid_pose_format(est_dict)

    # generate poses
    if gt_format == 'tum_cov':
        gt_poses, gt_cov = utils.convert_file_dict_to_pose_dict(gt_dict, file_format=gt_format)
        est_poses, est_cov = utils.convert_file_dict_to_pose_dict(est_dict, file_format=est_format)
    else:
        gt_poses  = utils.convert_file_dict_to_pose_dict(gt_dict, file_format=gt_format)
        est_poses = utils.convert_file_dict_to_pose_dict(est_dict, file_format=est_format)

    # associate sequences according to timestamps
    if not args.ignore_timestamp_match:
        gt_poses, est_poses = utils.associate_and_filter(gt_poses, est_poses, offset=float(args.offset), max_difference=float(args.max_difference), offset_initial=float(args.offset_initial), recommended_offset=args.recommended_offset)
        if gt_format == 'tum_cov':
            gt_cov, est_cov = utils.associate_and_filter(gt_cov, est_cov, offset=float(args.offset), max_difference=float(args.max_difference), offset_initial=float(args.offset_initial), recommended_offset=args.recommended_offset)

    # apply scale
    scale = float(args.scale)
    if args.automatic_scale:
        scale = utils.compute_scale_from_trajectories(gt_poses, est_poses)
    print('Using scale: %f' % scale)
    gt_poses  = utils.scale_dict(gt_poses, scale_factor=1)
    est_poses = utils.scale_dict(est_poses, scale_factor=scale)
    if gt_format == 'tum_cov':
        gt_cov_   = utils.scale_dict(gt_cov, scale_factor=1, is_cov=True)
        est_cov   = utils.scale_dict(est_cov, scale_factor=scale, is_cov=True)

    # align poses
    if args.alignment == 'manifold':
        if gt_format == 'tum_cov':
            gt_poses, est_poses, T_align_man = utils.align_trajectories_manifold(gt_poses, est_poses, cov_est=est_cov, align_gt=False)
        else:
            gt_poses, est_poses_align, T_align_man = utils.align_trajectories_manifold(gt_poses, est_poses, align_gt=False)
    elif args.alignment == 'horn':
        gt_poses, est_poses, T_align_horn = utils.align_trajectories_horn(gt_poses, est_poses, align_gt=False)
    elif args.alignment == 'first':
        gt_poses, est_poses = utils.align_trajectories_to_first(gt_poses, est_poses)

    ## apply fixed transform
    #if args.gt_static_transform:
    #    gt_static_dict  = utils.read_file_dict(args.gt_static_transform)
    #    gt_static_format = utils.check_valid_pose_format(gt_static_dict)
    #    gt_static_poses = utils.convert_file_dict_to_pose_dict(gt_static_dict, file_format=gt_static_format)
    #    gt_static_poses = dict( [(a, gt_static_poses[a]) for a in gt_poses] )
    #    gt_poses = dict([ (a, np.dot(gt_poses[a], gt_static_poses[b])) for a,b in zip(gt_poses,gt_static_poses)])
    #if args.est_static_transform:
    #    est_static_dict  = utils.read_file_dict(args.est_static_transform)
    #    est_static_format = utils.check_valid_pose_format(est_static_dict)
    #    est_static_poses = utils.convert_file_dict_to_pose_dict(est_static_dict, file_format=est_static_format)
    #    est_static_poses = dict( [(a, est_static_poses[a]) for a in est_poses] )
    #    est_poses = dict([ (a, np.dot(est_poses[a], est_static_poses[b])) for a,b in zip(est_poses,est_static_poses)])


    if(not args.no_metrics):
        # Compute metrics
        # ATE (Absolute trajectory error)
        ate_horn_error = slam_metrics.ATE_Horn(gt_poses, est_poses)
        slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0), verbose=args.verbose, title='ATE - Horn - XYZ', save=args.save_statistics)

        ate_horn_error = slam_metrics.ATE_Horn(gt_poses, est_poses, axes='X')
        slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0), verbose=args.verbose, title='ATE - Horn - X', save=args.save_statistics)

        ate_horn_error = slam_metrics.ATE_Horn(gt_poses, est_poses, axes='Y')
        slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0), verbose=args.verbose, title='ATE - Horn - Y', save=args.save_statistics)

        ate_horn_error = slam_metrics.ATE_Horn(gt_poses, est_poses, axes='Z')
        slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0), verbose=args.verbose, title='ATE - Horn - Z', save=args.save_statistics)



        # ATE (Absolute trajectory error, SE(3))
        if(args.ate_manifold):
            ate_se3_error = slam_metrics.ATE_SE3(gt_poses,
                                                 est_poses,
                                                 offset=float(args.offset),
                                                 max_difference=float(args.max_difference))
            slam_metrics.compute_statistics(np.linalg.norm(ate_se3_error[0:3,:], axis=0), variable='Translational', verbose=args.verbose, title='ATE - Manifold', save=args.save_statistics)
            slam_metrics.compute_statistics(np.linalg.norm(ate_se3_error[3:6,:], axis=0), variable='Rotational', verbose=args.verbose, title='ATE - Manifold', save=args.save_statistics)

        # RPE (Relative Pose Error)
        if(args.rpe):
            rpe_error, rpe_trans_error, rpe_rot_error, rpe_distance = slam_metrics.RPE(gt_poses,
                                                                       est_poses,
                                                                       param_max_pairs=int(args.max_pairs),
                                                                       param_fixed_delta=args.fixed_delta,
                                                                       param_delta=float(args.delta),
                                                                       param_delta_unit=args.delta_unit,
                                                                       param_offset=float(args.offset))

            slam_metrics.compute_statistics(np.linalg.norm(rpe_error[0:3,:], axis=0), variable='Translational', verbose=args.verbose, title=('RPE - %s [%s]' % (args.delta, args.delta_unit)), save=args.save_statistics)
            slam_metrics.compute_statistics(np.linalg.norm(rpe_error[3:6,:], axis=0), variable='Rotational', verbose=args.verbose, title=('RPE - %s [%s]' % (args.delta, args.delta_unit)), save=args.save_statistics)

        # DDT (Drift per distance)
        if(args.ddt):
            ddt = np.divide(rpe_error, rpe_distance)
            slam_metrics.compute_statistics(np.linalg.norm(ddt[0:3,:], axis=0), variable='Translational', verbose=args.verbose, title='DDT', save=args.save_statistics)
            slam_metrics.compute_statistics(np.linalg.norm(ddt[3:6,:], axis=0), variable='Rotational', verbose=args.verbose, title='DDT', save=args.save_statistics)

    #if(args.show_plots or args.save_plots):
    gt_data = gt_poses
    est_data = est_poses

    gt_stamps, gt_xyz      = utils.get_translations_along_trajectory(gt_poses)
    gt_stamps, gt_angles   = utils.get_orientations_along_trajectory(gt_poses)
    est_stamps, est_xyz    = utils.get_translations_along_trajectory(est_poses)
    est_stamps, est_angles = utils.get_orientations_along_trajectory(est_poses)

    if args.save_translations:
        with open('translations_gt.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['time', 'x', 'y', 'z'])
            for t,d in zip(gt_stamps,gt_xyz):
                wr.writerow([t, d[0,0], d[0,1], d[0,2]])
        with open('translations_est.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['time', 'x', 'y', 'z'])
            for t,d in zip(est_stamps,est_xyz):
                wr.writerow([t, d[0,0], d[0,1], d[0,2]])

    if args.show_plots or args.save_plots:
        plot_utils.plot_2d_traj_xyz(gt_stamps, gt_xyz.transpose(), est_stamps, est_xyz.transpose(), show_fig=args.show_plots, save_fig=args.save_plots)
    #plot_utils.plot_2d_traj_xyz(gt_stamps, gt_angles, est_stamps, est_angles)
    #plot_utils.plot_3d_xyz(gt_xyz, est_xyz, show_fig=args.show_plots, save_fig=args.save_plots)
    #plot_utils.plot_3d_xyz_with_cov(gt_data, est_data, gt_cov=gt_cov, est_cov=est_cov)
    #plot_utils.plot_3d_xyz(gt_xyz, est_xyz)
