#!/usr/bin/python

"""

This script computes several SLAM metrics.
It is based on the TUM scripts

"""

import sys
import numpy as np
import argparse
import glob
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

    #parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    #parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    #parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    args = parser.parse_args()

    # configure the plotting stuff
    plot_utils.set_language(lang=args.plot_lang)
    plot_utils.set_file_extension(ext=args.plot_format)

    # read files in TUM format or TUM modified format (with covariances)
    # we use glob to handle the case of single files or a list of files that match some specified pattern
    print(args.gt_file)
    print(args.est_file)

    gt_files  = glob.glob(args.gt_file)
    est_files = glob.glob(args.est_file)

    if len(gt_files) == 0 or len(est_files) == 0:
        sys.exit("Either the ground truth or estimated trajectory don't have any files associated. Please check the file names")
    if len(gt_files) != 1:
        sys.exit("Please select only one ground truth trajectory file")

    # prepare temporal list to allocate the data
    gt_poses_list = []
    gt_cov_list = []
    est_poses_list = []
    est_cov_list = []

    # read estimated files
    for est in est_files:
        est_dict = utils.read_file_dict(est)
        est_format = utils.check_valid_pose_format(est_dict)
        if est_format == 'tum_cov':
            est_poses, est_cov = utils.convert_file_dict_to_pose_dict(est_dict, file_format=est_format)
            est_poses_list.append(est_poses)
            est_cov_list.append(est_cov)
        else:
            est_poses = utils.convert_file_dict_to_pose_dict(est_dict, file_format=est_format)
            est_poses_list.append(est_poses)
    num_files = len(est_files)

    # read ground truth
    gt_dict  = utils.read_file_dict(gt_files[0])
    # check file format
    gt_format = utils.check_valid_pose_format(gt_dict)
    # generate poses
    if gt_format == 'tum_cov':
        gt_poses, gt_cov = utils.convert_file_dict_to_pose_dict(gt_dict, file_format=gt_format)
        gt_poses_list = [gt_poses]*num_files
        gt_cov_list = [gt_cov]*num_files
    else:
        gt_poses = utils.convert_file_dict_to_pose_dict(gt_dict, file_format=gt_format)
        gt_poses_list = [gt_poses]*num_files
    print(len(gt_poses_list))
    print(len(est_poses_list))


    # associate sequences according to timestamps
    if not args.ignore_timestamp_match:
        for i in range(num_files):
        # aling each sequence with their corresponding ground truth
            gt_poses_list[i], est_poses_list[i] = utils.associate_and_filter(gt_poses_list[i], est_poses_list[i], offset=float(args.offset), max_difference=float(args.max_difference), offset_initial=float(args.offset_initial), recommended_offset=args.recommended_offset)
            if gt_format == 'tum_cov':
                gt_cov_list[i], est_cov_list[i] = utils.associate_and_filter(gt_cov_list[i], est_cov_list[i], offset=float(args.offset), max_difference=float(args.max_difference), offset_initial=float(args.offset_initial), recommended_offset=args.recommended_offset)

    # apply scale
    for i in range(num_files):
        scale = float(args.scale)
        if args.automatic_scale:
            scale = utils.compute_scale_from_trajectories(gt_poses_list[i], est_poses_list[i])
        print('Using scale: %f' % scale)
        #gt_poses_list[i]  = utils.scale_dict(gt_poses_list[i, scale_factor=1)
        est_poses_list[i] = utils.scale_dict(est_poses_list[i], scale_factor=scale)
        if gt_format == 'tum_cov':
            #gt_cov_list[i_   = utils.scale_dict(gt_cov_list[i, scale_factor=1, is_cov=True)
            est_cov_list[i]   = utils.scale_dict(est_cov_list[i], scale_factor=scale, is_cov=True)

    # align poses
    for i in range(num_files):
        if args.alignment == 'manifold':
            if gt_format == 'tum_cov':
                gt_poses_list[i], est_poses_list[i], T_align_man = utils.align_trajectories_manifold(gt_poses_list[i], est_poses_list[i], cov_est=est_cov_list[i], align_gt=False)
            else:
                gt_poses_list[i], est_poses_align_list[i], T_align_man = utils.align_trajectories_manifold(gt_poses_list[i], est_poses_list[i], align_gt=False)
        elif args.alignment == 'horn':
            gt_poses_list[i], est_poses_list[i], T_align_horn = utils.align_trajectories_horn(gt_poses_list[i], est_poses_list[i], align_gt=False)
        elif args.alignment == 'first':
            gt_poses_list[i], est_poses_list[i] = utils.align_trajectories_to_first(gt_poses_list[i], est_poses_list[i])
        print(len(gt_poses_list[i]))
        print(len(est_poses_list[i]))

    if(not args.no_metrics):
        ate_horn_stats_xyz = []
        ate_horn_stats_x = []
        ate_horn_stats_y = []
        ate_horn_stats_z = []
        for i in range(num_files):
            # Compute metrics
            # ATE (Absolute trajectory error)
            ate_horn_error = slam_metrics.ATE_Horn(gt_poses_list[i], est_poses_list[i])
            stats = slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0), verbose=args.verbose, title='ATE - Horn - XYZ')
            ate_horn_stats_xyz.append(stats)

            ate_horn_error = slam_metrics.ATE_Horn(gt_poses_list[i], est_poses_list[i], axes='X')
            stats = slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0), verbose=args.verbose, title='ATE - Horn - X')
            ate_horn_stats_x.append(stats)

            ate_horn_error = slam_metrics.ATE_Horn(gt_poses_list[i], est_poses_list[i], axes='Y')
            stats = slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0), verbose=args.verbose, title='ATE - Horn - Y')
            ate_horn_stats_y.append(stats)

            ate_horn_error = slam_metrics.ATE_Horn(gt_poses_list[i], est_poses_list[i], axes='Z')
            stats = slam_metrics.compute_statistics(np.linalg.norm(ate_horn_error, axis=0), verbose=args.verbose, title='ATE - Horn - Z')
            ate_horn_stats_z.append(stats)

        # ATE (Absolute trajectory error, SE(3))
        if(args.ate_manifold):
            ate_se3_error = slam_metrics.ATE_SE3(gt_poses,
                                                 est_poses,
                                                 offset=float(args.offset),
                                                 max_difference=float(args.max_difference))
            slam_metrics.compute_statistics(np.linalg.norm(ate_se3_error[0:3,:], axis=0), variable='Translational', verbose=args.verbose, title='ATE - Manifold')
            slam_metrics.compute_statistics(np.linalg.norm(ate_se3_error[3:6,:], axis=0), variable='Rotational', verbose=args.verbose, title='ATE - Manifold')

        # RPE (Relative Pose Error)
        if(args.rpe):
            rpe_error, rpe_trans_error, rpe_rot_error, rpe_distance = slam_metrics.RPE(gt_poses,
                                                                       est_poses,
                                                                       param_max_pairs=int(args.max_pairs),
                                                                       param_fixed_delta=args.fixed_delta,
                                                                       param_delta=float(args.delta),
                                                                       param_delta_unit=args.delta_unit,
                                                                       param_offset=float(args.offset))

            slam_metrics.compute_statistics(np.linalg.norm(rpe_error[0:3,:], axis=0), variable='Translational', verbose=args.verbose, title=('RPE - %s [%s]' % (args.delta, args.delta_unit)))
            slam_metrics.compute_statistics(np.linalg.norm(rpe_error[3:6,:], axis=0), variable='Rotational', verbose=args.verbose, title=('RPE - %s [%s]' % (args.delta, args.delta_unit)))

        # DDT (Drift per distance)
        if(args.ddt):
            ddt = np.divide(rpe_error, rpe_distance)
            slam_metrics.compute_statistics(np.linalg.norm(ddt[0:3,:], axis=0), variable='Translational', verbose=args.verbose, title='DDT')
            slam_metrics.compute_statistics(np.linalg.norm(ddt[3:6,:], axis=0), variable='Rotational', verbose=args.verbose, title='DDT')

    if(args.show_plots or args.save_plots):
        for i in range(num_files):
            gt_data = gt_poses_list[i]
            est_data = est_poses_list[i]

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

            plot_utils.plot_2d_traj_xyz(gt_stamps, gt_xyz, est_stamps, est_xyz, show_fig=args.show_plots, save_fig=args.save_plots)
            #plot_utils.plot_2d_traj_xyz(gt_stamps, gt_angles, est_stamps, est_angles)
            #plot_utils.plot_3d_xyz(gt_xyz, est_xyz, show_fig=args.show_plots, save_fig=args.save_plots)
            #plot_utils.plot_3d_xyz_with_cov(gt_data, est_data, gt_cov=gt_cov, est_cov=est_cov)
            #plot_utils.plot_3d_xyz(gt_xyz, est_xyz)
