#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This file concentrates many functions used to compute SLAM metrics
Many are based on the TUM scripts by Juergen Sturm (see license below)

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements:
# sudo apt-get install python-argparse
"""

import argparse
import sys
import os
import math
import numpy as np
import SE3UncertaintyLib as SE3Lib

_EPS = np.finfo(float).eps * 4.0

def deg_to_rad(angle):
    return angle * np.pi / 180.0

def rad_to_deg(angle):
    return angle * 180.0 / np.pi

def get_supported_file_formats():
    return {'tum': 7, 'tum_cov': 28}

def read_file_dict(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples

    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    file_list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    file_list = [(float(l[0]),l[1:]) for l in file_list if len(l)>1]
    return dict(file_list)

def check_valid_pose_format(file_dict):
    """
    Checks the format of the file dict and returns a string with the format

    It exits if the format of the file is not found

    Input:
    file_dict -- a dictionary of (stamp,data) tuples

    Output:
    format -- a string with the format of the file
    """

    # check if all the rows have the same lenght andsave the length
    num_items = -1
    for item in file_dict:
        num_curr = len(file_dict[item])  # the current number of items
        if num_items == -1:
            num_items = num_curr
        elif num_items != num_curr:
            sys.exit("There are error in the inputs files: Some rows have different length")

    # check the format depending on the number of counted rows
    supported_formats = get_supported_file_formats()
    for f in supported_formats:
        if supported_formats[f] == num_items:
            return f

    # if we don't find any supported format, exit
    sys.exit("The format of files don't match any supported, please check the input files")

def convert_file_dict_to_pose_dict(file_dict, file_format='tum'):
    """
    Converts the file dict into a new dictionary with SE(3) poses

    Input:
    file_dict -- dictionary of (stamp,data) tuples
    file_format -- a string with the format of data

    Output:
    pose_dict -- dictionary of (stamp,poses) tuples
    cov_dict -- (optional, depends on the format) dictionary of (stamp,covariance) tuples

    """

    pose_dict = dict( [(key, transform44(file_dict[key][0:7])) for key in file_dict] )
    if file_format == 'tum_cov':
        cov_dict = dict( [(key, covariance66(file_dict[key][7:])) for key in file_dict] )

    if file_format == 'tum':
        return pose_dict
    else:
        return pose_dict, cov_dict

def scale_dict(unscaled_dict, scale_factor=1, is_cov=False):
    """
    Scales a dictionary with poses or covariances

    Input:
    unscaled_dict -- dictionary of (stamp,pose) tuples
    scale -- scale factor
    scale_cov -- a boolean to enable the scale covariance matrices


    Output:
    dict -- dictionary of scaled (stamp,poses)/(scale,cov) tuples

    """

    for key in unscaled_dict:
        if is_cov:
            unscaled_dict[key] = scale_cov(unscaled_dict[key], scale_factor)
        else:
            unscaled_dict[key] = scale_pose(unscaled_dict[key], scale_factor)
    return unscaled_dict

def associate(first_list, second_list, offset=0.0, max_difference=0.02, offset_initial=0.0):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if (abs(a - (b + offset)) < max_difference)]
    potential_matches.sort()
    matches = []

    # get initial timestamp
    first_keys.sort()
    second_keys.sort()
    first_t0 = first_keys[0] + offset_initial
    second_t0 = second_keys[0] + offset_initial
    print(first_t0)
    print(second_t0)
    print(offset_initial)

    # generate the matches
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            if a > first_t0 and b > second_t0:
                first_keys.remove(a)
                second_keys.remove(b)
                matches.append((a, b))

    matches.sort()

    if(len(matches)<2):
        sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")

    return matches

def associate_and_filter(first_list, second_list, third_list=None, offset=0.0, max_difference=0.02, offset_initial=0.0):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    third_list -- (optional) third dictionary of (stamp,data) tuples that will be aligned with the second list
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """

    matches = associate(first_list, second_list, offset=offset, max_difference=max_difference, offset_initial=offset_initial)

    filt_first  = dict( [(a, first_list[a]) for a,b in matches] )
    filt_second = dict( [(b, second_list[b]) for a,b in matches] )

    if third_list is not None:
        filt_third = dict( [(b, first_list[b]) for a,b in matches] )
        return filt_first, filt_second, filt_third
    else:
        return filt_first, filt_second


def find_closest_index(L,t):
    """
    Find the index of the closest value in a list.

    Input:
    L -- the list
    t -- value to be found

    Output:
    index of the closest element
    """
    beginning = 0
    difference = abs(L[0] - t)
    best = 0
    end = len(L)
    while beginning < end:
        middle = int((end+beginning)/2)
        if abs(L[middle] - t) < difference:
            difference = abs(L[middle] - t)
            best = middle
        if t == L[middle]:
            return middle
        elif L[middle] > t:
            end = middle
        else:
            beginning = middle + 1
    return best

def scale_pose(a,scalar):
    """
    Scale the translational components of a 4x4 homogeneous matrix by a scale factor.
    """
    return np.array(
        [[a[0,0], a[0,1], a[0,2], a[0,3]*scalar],
         [a[1,0], a[1,1], a[1,2], a[1,3]*scalar],
         [a[2,0], a[2,1], a[2,2], a[2,3]*scalar],
         [a[3,0], a[3,1], a[3,2], a[3,3]]]
                       )
def scale_cov(a,scalar):
    """
    Scale the translational components of a 6x6 covariance matrix by a scale factor.
    """
    return np.array(
        [[a[0,0]*scalar*scalar, a[0,1]*scalar*scalar, a[0,2]*scalar*scalar, a[0,3]*scalar, a[0,4]*scalar, a[0,5]*scalar],
         [a[1,0]*scalar*scalar, a[1,1]*scalar*scalar, a[1,2]*scalar*scalar, a[1,3]*scalar, a[1,4]*scalar, a[1,5]*scalar],
         [a[2,0]*scalar*scalar, a[2,1]*scalar*scalar, a[2,2]*scalar*scalar, a[2,3]*scalar, a[2,4]*scalar, a[2,5]*scalar],
         [a[3,0]*scalar,        a[3,1]*scalar,        a[3,2]*scalar,        a[3,3],        a[3,4],        a[3,5]],
         [a[4,0]*scalar,        a[4,1]*scalar,        a[4,2]*scalar,        a[4,3],        a[4,4],        a[4,5]],
         [a[5,0]*scalar,        a[5,1]*scalar,        a[5,2]*scalar,        a[5,3],        a[5,4],        a[5,5]]]
                       )

def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return np.linalg.norm(transform[0:3,3])

def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos( min(1,max(-1, (np.trace(transform[0:3,0:3]) - 1)/2) ))

def distances_along_trajectory(traj):
    """
    Compute the translational distances along a trajectory.
    """
    keys = list(traj.keys())
    keys.sort()
    motion = [transform_diff(traj[keys[i+1]],traj[keys[i]]) for i in range(len(keys)-1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_distance(t)
        distances.append(sum)
    return distances

def rotations_along_trajectory(traj,scale):
    """
    Compute the angular rotations along a trajectory.
    """
    keys = list(traj.keys())
    keys.sort()
    motion = [transform_diff(traj[keys[i+1]],traj[keys[i]]) for i in range(len(keys)-1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_angle(t)*scale
        distances.append(sum)
    return distances


def transform_diff(a,b):
    """
    Compute the relative 3D transformation between a and b.

    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)

    Output:
    Relative 3D transformation from a to b.
    """
    return np.dot(np.linalg.inv(a),b)

def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[0:3]
    q = np.array(l[3:7], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q  = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def rotm_to_rpy(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return np.array([x, y, z])

def covariance66(l):
    """
    Generate a 6x6 covariance matrix from an upper-triangular representation

    Input:
    l -- tuple consisting of (Sxx, Sxy, Sxz, Sxr, Sxt, Sxp, Syy, Syz, Syr, Syt, Syp, Szz, Szr, Szt, Szp, Srr, Srt, Srp, Stt, Stp, Spp)
    where:
        x = x-axis
        y = y-axis
        z = x-axis
        r = roll-axis
        t = tilt(pitch)-axis
        p = pan(yaw)-axis

    Output:
    matrix -- 6x6 SE(3) covariance matrix
    """

    Sxx, Sxy, Sxz, Sxr, Sxt, Sxp, Syy, Syz, Syr, Syt, Syp, Szz, Szr, Szt, Szp, Srr, Srt, Srp, Stt, Stp, Spp = l

    return np.array((
        (Sxx, Sxy, Sxz, Sxr, Sxt, Sxp),
        (Sxy, Syy, Syz, Syr, Syt, Syp),
        (Sxz, Syz, Szz, Szr, Szt, Szp),
        (Sxr, Syr, Szr, Srr, Srt, Srp),
        (Sxt, Syt, Szt, Srt, Stt, Stp),
        (Sxp, Syp, Szp, Srp, Stp, Spp)
        ), dtype=np.float64)

def align_trajectories_manifold(traj_gt, traj_est, cov_est=None, verbose=False, align_gt=True, return_alignment=True):
    """
    Aligns two trajectories using the alignment on manifold.

    Salas, M., Latif, Y., Reid, I. D., & Montiel, J. M. M. (2015).
    Trajectory Alignment and Evaluation in SLAM : Horn’s Method vs Alignment on the Manifold.
    In RSS Workshop: The Problem of Mobile Sensors.

    Note: The Jacobian of the cost function is approximated by the SE(3) adjoint

    Input:
    traj_gt -- a dictionary of ground truth (stamp,pose)
    traj_est -- a dictionary of estimated (stamp,pose)
    cov_est -- (optional) a dictionary of estimated (stamp,covariance)

    Output:
    traj_gt_aligned -- a dictionary of aligned ground truth (stamp,pose)
    traj_est_aligned -- a dictionary of aligned ground truth (stamp,pose)
    """

    # gauss newton
    iterations = 100
    N = len(traj_gt)
    Vprev = 0   # previous value of the cost function

    # the optimal alignment transformation
    Talign = np.eye(4, dtype=float)

    for it in range(iterations):
        LHS = np.zeros(6)
        RHS = np.zeros(6)

        # align head frames
        for q,p in zip(traj_gt, traj_est):
            Q = traj_gt[q]
            P = traj_est[p]
            invQ = np.linalg.inv(Q)

            delta_k = SE3Lib.TranToVec(np.dot(np.dot(invQ, Talign), P)) # Eq. 2: Log(Qi^-1 * T * Pi)

            Jac = SE3Lib.TranAd(invQ)  # approximation of the Jacobian by the SE(3) Adjoint (Eq. 18)
            Jact = Jac.T
            if cov_est is not None:
                JactS = np.dot(Jact, np.linalg.inv(cov_est[p]))
            else:
                JactS = Jact
            LHS = LHS + np.dot(JactS, Jac)
            RHS = RHS + np.dot(JactS, delta_k)

        # update
        delta = -np.linalg.solve(LHS,RHS)

        # apply update
        Talign = np.dot(SE3Lib.VecToTran(delta),Talign)
        Sigma = np.linalg.inv(LHS)

        # Check the cost function
        V = 0.
        for q,p in zip(traj_gt, traj_est):
            Q = traj_gt[q]
            P = traj_est[p]
            delta_k = SE3Lib.TranToVec(np.dot(np.dot(invQ, Talign), P))
            V = V + np.dot(delta_k.T,delta_k)
        if abs(V - Vprev) < 1e-10:
            break
        Vprev = V

    # apply alignment
    if align_gt:
        Talign = np.linalg.inv(Talign)
        traj_gt_aligned = dict([ (a, np.dot(Talign, traj_gt[a])) for a in traj_gt])
        traj_est_aligned = traj_est
    else:
        traj_gt_aligned = traj_gt
        traj_est_aligned = dict([ (a, np.dot(Talign, traj_est[a])) for a in traj_est])

    if verbose:
        print('Alignment transformation:', Talign)

    # return alignment
    if return_alignment:
        return traj_gt_aligned, traj_est_aligned, Talign
    else:
        return traj_gt_aligned, traj_est_aligned

def align_trajectories_horn(traj_gt, traj_est, verbose=False, align_gt=True, return_alignment=True):
    """
    Aligns two trajectories using the method of Horn (closed-form).

    Horn, B. K. P. (1987).
    Closed-form solution of absolute orientation using unit quaternions.
    Journal of the Optical Society of America A, 4(4), 629.

    Input:
    traj_gt -- a dictionary of ground truth (stamp,pose)
    traj_est -- a dictionary of estimated (stamp,pose)

    Output:
    traj_gt_aligned -- a dictionary of aligned ground truth (stamp,pose)
    traj_est_aligned -- a dictionary of aligned ground truth (stamp,pose)

    """

    # recover a list with the translations only
    gt_xyz  = np.matrix([[float(value) for value in traj_gt[a][0:3,3]] for a in traj_gt]).transpose()
    est_xyz  = np.matrix([[float(value) for value in traj_est[a][0:3,3]] for a in traj_est]).transpose()

    # substract the mean
    gt_xyz_zerocentered = gt_xyz - gt_xyz.mean(1)
    est_xyz_zerocentered = est_xyz - est_xyz.mean(1)

    # estimate orientation
    W = np.zeros( (3,3) )
    for column in range(gt_xyz.shape[1]):
        W += np.outer(gt_xyz_zerocentered[:,column], est_xyz_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1

    # alignment orientation
    rot = U*S*Vh

    # alignment translation
    trans = est_xyz.mean(1) - rot * gt_xyz.mean(1)

    # create transformation to align ground truth
    Talign = np.eye(4, dtype=float)
    Talign[0:3,0:3] = rot
    Talign[0:3,3] = trans.reshape(3)

    # apply alignment
    if align_gt:
        traj_gt_aligned = dict([ (a, np.dot(Talign, traj_gt[a])) for a in traj_gt])
        traj_est_aligned = traj_est
    else:
        Talign = np.linalg.inv(Talign)
        traj_gt_aligned = traj_gt
        traj_est_aligned = dict([ (a, np.dot(Talign, traj_est[a])) for a in traj_est])

    if verbose:
        print('Alignment transformation:', Talign)

    # return alignment
    if return_alignment:
        return traj_gt_aligned, traj_est_aligned, Talign
    else:
        return traj_gt_aligned, traj_est_aligned

def align_trajectories_to_first(traj_gt, traj_est, verbose=False):
    """
    Aligns two trajectories by substracting the initial pose.

    Input:
    traj_gt -- a dictionary of ground truth (stamp,pose)
    traj_est -- a dictionary of estimated (stamp,pose)

    Output:
    traj_gt_aligned -- a dictionary of aligned ground truth (stamp,pose)
    traj_est_aligned -- a dictionary of aligned ground truth (stamp,pose)

    """

    #print(traj_gt)
    #print(traj_est)

    stamps_gt = list(traj_gt.keys())
    stamps_gt.sort()
    stamps_est = list(traj_est.keys())
    stamps_est.sort()

    t0_gt = stamps_gt[0]
    t0_est = stamps_est[0]

    traj_gt_0 = np.linalg.inv(traj_gt[t0_gt])
    traj_est_0 = np.linalg.inv(traj_est[t0_est])

    traj_gt_aligned  = dict( [(a, np.dot(traj_gt_0, traj_gt[a])) for a in traj_gt])
    traj_est_aligned  = dict( [(a, np.dot(traj_est_0, traj_est[a])) for a in traj_est])

    return traj_gt_aligned, traj_est_aligned
