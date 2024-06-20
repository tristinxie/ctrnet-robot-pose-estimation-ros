import numpy as np
import kornia
import torch
import cv2
from scipy import optimize
from scipy.stats import norm
from sklearn.metrics.pairwise import rbf_kernel


def sample_gaussian(std):
    m = np.zeros(std.shape)
    sample = norm.rvs(m, std)
    prob_individual = norm.pdf(sample, m, std)
    prob_individual[np.isnan(prob_individual)] = 1.0
    return sample, np.prod(prob_individual)

def additive_gaussian(state, std):
    sample, prob = sample_gaussian(std)
    return state + sample, prob

def axisAngleToRotationMatrix(axis_angle):
    angle = np.linalg.norm(axis_angle)
    
    if angle < 1e-5:
        return np.eye(3)
    
    axis  = axis_angle/angle
    cross_product_mat_axis = np.array([[0, -axis[2], axis[1]],
                                       [axis[2], 0, -axis[0]],
                                       [-axis[1], axis[0], 0]])
    return np.cos(angle)*np.eye(3) + np.sin(angle) * cross_product_mat_axis \
            + (1.0 - np.cos(angle))*np.outer(axis,axis)
# What is robot_arm.getPointFeatures()
# State is: [ori_x, ori_y, ori_z, pos_x, pos_y, pos_z]
def point_feature_obs(state, points_2d, ctrnet, joint_angles, cam, cTr, gamma):
    #convert state to angle axis
    T = np.eye(4)
    # print(state.shape)
    T[:-1, -1] = np.array(state[3:])
    T[:-1, :-1] = axisAngleToRotationMatrix(state[:3])
    # Want to have final result of project image plane coords
    _, t_list = ctrnet.robot.get_joint_RT(joint_angles)
    p_t = t_list[[0,2,3,4,6,7,8]] # remove 1 and 5 links as they are overlapping with 2 and 6
    K = np.float64(ctrnet.intrinsics)
    # only projects from camera coords to image coords
    # p_c = None
    # projected_points = cam.projectPoints(p_c)
    cTr = cTr.cpu().detach().numpy().squeeze()
    # print(cTr.shape)
    cTr_mat = np.eye(4)
    cTr_mat[:-1, -1] = np.array(cTr[3:])
    cTr_mat[:-1, :-1] = axisAngleToRotationMatrix(cTr[:3])
    p_cTr  = np.dot(cTr_mat, T)
    # print(p_cTr)
    rvec = np.float64(p_cTr[:3, :3])
    tvec = np.float64(p_cTr[:3, 3:])
    # print(rvec)
    # print(tvec)
    projected_points, _ = cv2.projectPoints(p_t, rvec, tvec, K, None)

    # Make association between detected and projected points to compute prob and use prob to update weights
    projected_points = projected_points.squeeze(1).reshape(1, 14)
    detected_points = points_2d.reshape(1, 14).cpu().detach().numpy()
    # print(projected_points, detected_points)
    # TODO Fix NaN problem
    prob = rbf_kernel(projected_points, Y=detected_points).squeeze()
    return prob