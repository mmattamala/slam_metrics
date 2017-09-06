#!/usr/bin/python

"""

This script computes several SLAM metrics.
It is based on the TUM scripts

"""

import sys
import numpy as np
import argparse
import tum_utils
import slam_metrics


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
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    parser.add_argument('--delta', help='delta for evaluation (default: 1.0)',default=1.0)
    parser.add_argument('--delta_unit', help='unit of delta (options: \'s\' for seconds, \'m\' for meters, \'rad\' for radians, \'f\' for frames; default: \'s\')',default='s')
    #parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    #parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    #parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    #parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    args = parser.parse_args()

    # read files in TUM format or TUM modified format (with covariances)
    gt_list  = tum_utils.read_file_list(args.gt_file)
    est_list = tum_utils.read_file_list(args.est_file)

    # check file format, we use the first element
    gt_n_elements = len(gt_list[gt_list.keys()[0]])
    est_n_elements = len(est_list[est_list.keys()[0]])

    if(gt_n_elements != est_n_elements):
        sys.exit("The format of both files is different, please check the input files")
    if gt_n_elements == 7:
        dataset_format = 'TUM'
    elif gt_n_elements == 28:
        dataset_format = 'TUM_mod'
    else:
        sys.exit("The format of files don't match any supported, please check the input files")

    # associate sequences according to timestamps
    matches = tum_utils.associate(gt_list, est_list, float(args.offset), float(args.max_difference))
    if len(matches)<2:
        sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")

    #print(gt_list)
    #print(est_list)

    # generate numpy arrays for transformations, positions, and orientations
    # generate positions as a 3 x N matrix, where N is the number of poses
    gt_xyz  = np.matrix([[float(value) for value in gt_list[a][0:3]] for a,b in matches]).transpose()
    est_xyz = np.matrix([[float(value)*float(args.scale) for value in est_list[b][0:3]] for a,b in matches]).transpose()

    # generate orientations as a 4 x N matrix
    gt_quat  = np.matrix([[float(value) for value in gt_list[a][3:7]] for a,b in matches]).transpose()
    est_quat = np.matrix([[float(value) for value in est_list[b][3:7]] for a,b in matches]).transpose()

    # generate poses as a N x 4x4 dict
    gt_pose  = dict( [ (a,tum_utils.transform44(np.array(gt_list[a][0:7]))) for a,b in matches ] )
    est_pose = dict( [ (b,tum_utils.transform44(np.array(est_list[b][0:7]))) for a,b in matches ] )

    # if available, generate covariances as N x 6x6 dict
    if dataset_format == 'TUM_mod':
        gt_pose  = dict( [ (a,tum_utils.covariance66(np.array(gt_list[a][7:]))) for a,b in matches ] )
        est_pose = dict( [ (b,tum_utils.covariance66(np.array(est_list[b][7:]))) for a,b in matches ] )

    # Compute metrics
    # ATE (Absolute trajectory error)
    ate_horn_error, ate_horn_rot, ate_horn_trans, ate_horn_scale = slam_metrics.ATE_Horn(gt_xyz, est_xyz, show=True)
    slam_metrics.compute_statistics_per_axis(ate_horn_error)
    slam_metrics.compute_statistics_absolute(ate_horn_error, variable='translational')

    # ATE (Absolute trajectory error, SE(3))
    ate_se3_error = slam_metrics.ATE_SE3(gt_pose, est_pose, matches=matches, show=True)
    slam_metrics.compute_statistics_per_axis(ate_se3_error)
    slam_metrics.compute_statistics_absolute(ate_se3_error[0:3,:], variable='translational')
    slam_metrics.compute_statistics_absolute(ate_se3_error[3:6,:], variable='rotational')

    # RPE (Relative Pose Error)
    rpe_error, rpe_trans_error, rpe_rot_error, distance_travelled = slam_metrics.RPE(gt_pose, est_pose, args.delta, show=True)
    slam_metrics.compute_statistics_per_axis(rpe_error)
    slam_metrics.compute_statistics_absolute(rpe_error[0:3,:], variable='translational')
    slam_metrics.compute_statistics_absolute(rpe_error[3:6,:], variable='rotational')
