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

def projectPoints(points, K):
    # points is Nx3 np array
    points = np.transpose(points)
    projected_point = np.transpose(np.dot(K, points/points[-1,:]))[:,:-1]
    
    # points_homogeneous = np.concatenate( (points, np.ones((1, points.shape[1]))) )
    # points_r = np.dot(self.T, points_homogeneous)[:-1, :]
    
    
    return projected_point
# State is: [ori_x, ori_y, ori_z, pos_x, pos_y, pos_z]
def point_feature_obs(states, points_2d, ctrnet, joint_angles, cam, cTr, gamma):
    #convert state to angle axis
    num_particles = states.shape[0]
    T = np.eye(4)
    T = np.tile(T, (num_particles, 1, 1))

    T[:, :-1, -1] = states[:, 3:]
    print(axisAngleToRotationMatrix(states[:, :3]).shape)
    T[:, :-1, :-1] = axisAngleToRotationMatrix(states[:, :3])
    # Want to have final result of project image plane coords
    _, t_list = ctrnet.robot.get_joint_RT(joint_angles)
    p_t = t_list[[0,2,3,4,6,7,8]] # remove 1 and 5 links as they are overlapping with 2 and 6
    K = np.float64(ctrnet.intrinsics)
    # only projects from camera coords to image coords
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
    p_c = np.dot(np.dot(cTr_mat, T), np.transpose(np.concatenate((p_t, np.ones((p_t.shape[0], 1))), axis=1)))
    p_c = np.transpose(p_c)[:, :-1]
    projected_points = projectPoints(p_c, K)
    # projected_points_cv2, _ = cv2.projectPoints(p_t, rvec, tvec, K, None)
    # print(np.isclose(projected_points, projected_points_cv2.squeeze(1)))
    # print(projected_points)
    # print(projected_points_cv2)

    # Make association between detected and projected points to compute prob and use prob to update weights
    projected_points = projected_points.reshape(1, 14)
    detected_points = points_2d.reshape(1, 14).cpu().detach().numpy()
    # print(projected_points, detected_points)
    # TODO Fix NaN problem
    if np.any(np.isnan(projected_points)) or np.any(np.isnan(detected_points)):
        print(projected_points)
        print(detected_points)

    prob = rbf_kernel(projected_points, Y=detected_points).squeeze()
    return prob