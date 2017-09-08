#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Implementation of main metrics used in Visual SLAM
'''

import random
import math
import numpy as np
import SE3UncertaintyLib as SE3Lib
import tum_utils

def compute_statistics(err, verbose=False, variable='Translational', use_deg=True):
        """
        Computes the mean, RMSE, standard deviation, median, min and max from a vector of errors

        @param err: a MxN np array.
        M is the number of components by sample (M=3 if SO(3), M=6 if SE(3)). N is the number of samples.

        """
        stats = {}

        abs_err = np.fabs(err)

        # RMSE
        stats['rmse'] = np.sqrt(np.dot(abs_err, abs_err) / len(abs_err))
        # Mean
        stats['mean'] = np.mean(abs_err) # computed by column
        # Standard Deviation
        stats['std']  = np.std(abs_err) # computed by column
        # Median
        stats['median'] = np.median(abs_err) # computed by column
        # Min
        stats['min'] = np.min(np.fabs(abs_err)) # computed by column
        # Max
        stats['max'] = np.max(abs_err) # computed by column


        if verbose:
            for key in stats:
                if variable == 'Rotational':
                    if use_deg:
                        print(' %s %s: %f deg' % (variable, key, tum_utils.rad_to_deg(stats[key])))
                    else:
                        print(' %s %s: %f rad' % (variable, key, stats[key]))
                else:
                    print(' %s %s: %f m' % (variable, key, stats[key]))
        else:
            if variable == 'Rotational':
                if use_deg:
                    print(' %s rmse: %f deg' % (variable, tum_utils.rad_to_deg(stats['rmse'])))
                else:
                    print(' %s rmse: %f rad' % (variable, stats['rmse']))
            else:
                print(' %s rmse: %f m' % (variable, stats['rmse']))


        return stats


def ATE_SE3(traj_gt, traj_est, show=True, matches=None, offset=0.0, max_difference=0.02, scale=1.0):
    """
    This method computes the Absolute Trajectory Error (ATE)
    Ref: Sturm et al. (2012)

    @param estimated: a dictionary of matrices representing estimated poses
    @param ground_truth: a dictionary with real poses

    """
    if show:
        print('\nATE: Absolute Trajectory Error, SE(3) error')

    if matches is None:
        matches = tum_utils.associate(traj_gt, traj_est, offset=offset, max_difference=max_difference)
        if len(matches)<2:
            sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")

    errors = np.matrix([SE3Lib.TranToVec(tum_utils.transform_diff(traj_gt[a], tum_utils.scale(traj_est[b], scale))) for a,b in matches]).transpose()

    return errors

def ATE_Horn(traj_gt, traj_est, compute_scale=False, show=True):
    """Align two trajectories using the method of Horn (closed-form).
    It includes the automatic scale recovery modification by Raul Mur-Artal

    Input:
    traj_xyz_gt -- first trajectory (3xn)
    traj_xyz_est -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """

    if show:
        print('\nATE: Absolute Trajectory Error - Horn alignment')

    #np.set_printoptions(precision=3,suppress=True)
    traj_gt_zerocentered = traj_gt - traj_gt.mean(1)
    traj_est_zerocentered = traj_est - traj_est.mean(1)

    W = np.zeros( (3,3) )
    for column in range(traj_gt.shape[1]):
        W += np.outer(traj_gt_zerocentered[:,column],traj_est_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    s = 1.0
    if compute_scale:
        rottraj_gt = rot*traj_gt_zerocentered
        dots = 0.0
        norms = 0.0

        for column in range(traj_est_zerocentered.shape[1]):
    	    dots += np.dot(traj_est_zerocentered[:,column].transpose(),rottraj_gt[:,column])
            normi = np.linalg.norm(traj_gt_zerocentered[:,column])
            norms += normi*normi

        s = float(dots/norms)

    #print "scale: %f " % s
    trans = traj_est.mean(1) - s*rot * traj_gt.mean(1)

    traj_gt_aligned = s*rot * traj_gt + trans
    alignment_error = traj_gt_aligned - traj_est

    #trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]

    return alignment_error, rot, trans, s

def RPE(traj_gt, traj_est, param_max_pairs=10000, param_fixed_delta=False, param_delta=1.00, param_delta_unit="m", param_offset=0.00, param_scale=1.00, show=True):
    """
    This method computes the Relative Pose Error (RPE) and Drift Per Distance Travelled (DDT)
    Ref: Sturm et al. (2012), Scona et al. (2017)

    Input:
    traj_gt -- the first trajectory (ground truth)
    traj_est -- the second trajectory (estimated trajectory)
    param_max_pairs -- number of relative poses to be evaluated
    param_fixed_delta -- false: evaluate over all possible pairs
                         true: only evaluate over pairs with a given distance (delta)
    param_delta -- distance between the evaluated pairs
    param_delta_unit -- unit for comparison:
                        "s": seconds
                        "m": meters
                        "rad": radians
                        "deg": degrees
                        "f": frames
    param_offset -- time offset between two trajectories (to traj_xyz_gt the delay)
    param_scale -- scale to be applied to the second trajectory

    Output:
    list of compared poses and the resulting translation and rotation error
    """

    if show:
        if param_fixed_delta:
            print('\nRPE: Relative Pose Error, delta=%f [%s]' % (param_delta, param_delta_unit))
        else:
            print('\nRPE: Relative Pose Error with respect to %s' % param_delta_unit)

    stamps_gt = list(traj_gt.keys())
    stamps_est = list(traj_est.keys())
    stamps_gt.sort()
    stamps_est.sort()

    stamps_est_return = []
    for t_est in stamps_est:
        t_gt = stamps_gt[tum_utils.find_closest_index(stamps_gt,t_est + param_offset)]
        t_est_return = stamps_est[tum_utils.find_closest_index(stamps_est,t_gt - param_offset)]
        t_gt_return = stamps_gt[tum_utils.find_closest_index(stamps_gt,t_est_return + param_offset)]
        if not t_est_return in stamps_est_return:
            stamps_est_return.append(t_est_return)
    if(len(stamps_est_return)<2):
        raise Exception("Number of overlap in the timestamps is too small. Did you run the evaluation on the right files?")

    if param_delta_unit=="s":
        index_est = list(traj_est.keys())
        index_est.sort()
    elif param_delta_unit=="m":
        index_est = tum_utils.distances_along_trajectory(traj_est)
    elif param_delta_unit=="rad":
        index_est = tum_utils.rotations_along_trajectory(traj_est,1)
    elif param_delta_unit=="deg":
        index_est = tum_utils.rotations_along_trajectory(traj_est,180/np.pi)
    elif param_delta_unit=="f":
        index_est = range(len(traj_est))
    else:
        raise Exception("Unknown unit for delta: '%s'"%param_delta_unit)

    if not param_fixed_delta:
        if(param_max_pairs==0 or len(traj_est)<np.sqrt(param_max_pairs)):
            pairs = [(i,j) for i in range(len(traj_est)) for j in range(len(traj_est))]
        else:
            pairs = [(random.randint(0,len(traj_est)-1),random.randint(0,len(traj_est)-1)) for i in range(param_max_pairs)]
    else:
        pairs = []
        for i in range(len(traj_est)):
            j = tum_utils.find_closest_index(index_est,index_est[i] + param_delta)
            if j!=len(traj_est)-1:
                pairs.append((i,j))
        if(param_max_pairs!=0 and len(pairs)>param_max_pairs):
            pairs = random.sample(pairs,param_max_pairs)

    gt_interval = np.median([s-t for s,t in zip(stamps_gt[1:],stamps_gt[:-1])])
    gt_max_time_difference = 2*gt_interval

    result = []
    diff_pose = []
    for i,j in pairs:
        stamp_est_0 = stamps_est[i]
        stamp_est_1 = stamps_est[j]

        stamp_gt_0 = stamps_gt[ tum_utils.find_closest_index(stamps_gt,stamp_est_0 + param_offset) ]
        stamp_gt_1 = stamps_gt[ tum_utils.find_closest_index(stamps_gt,stamp_est_1 + param_offset) ]

        if(abs(stamp_gt_0 - (stamp_est_0 + param_offset)) > gt_max_time_difference  or
           abs(stamp_gt_1 - (stamp_est_1 + param_offset)) > gt_max_time_difference):
            continue

        gt_delta = tum_utils.transform_diff( traj_gt[stamp_gt_1], traj_gt[stamp_gt_0])
        est_delta = tum_utils.transform_diff( traj_est[stamp_est_1], traj_est[stamp_est_0] )
        error44 = tum_utils.transform_diff(  tum_utils.scale( est_delta, param_scale), gt_delta)

        gt_distance_travelled = tum_utils.compute_distance(gt_delta)
        # check if the distance is not nan or inf
        gt_distance_travelled = gt_distance_travelled if (not 0) else tum_utils._EPS

        diff_pose.append(error44)

        trans = tum_utils.compute_distance(error44)
        rot = tum_utils.compute_angle(error44)

        result.append([stamp_est_0, stamp_est_1, stamp_gt_0, stamp_gt_1, trans, rot])

    if len(result)<2:
        raise Exception("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory!")

    stamps = np.array(result)[:,0]
    trans_error = np.array(result)[:,4]
    rot_error = np.array(result)[:,5]

    errors = np.matrix([SE3Lib.TranToVec(dT) for dT in diff_pose]).transpose()


    return errors, trans_error, rot_error, gt_distance_travelled
