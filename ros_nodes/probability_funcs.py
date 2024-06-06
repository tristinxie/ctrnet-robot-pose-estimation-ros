import numpy as np
from scipy import optimize
from scipy.stats import norm


def sample_gaussian(std):
    m = np.zeros(std.shape)
    sample = norm.rvs(m, std)
    prob_individual = norm.pdf(sample, m, std)
    prob_individual[np.isnan(prob_individual)] = 1.0
    return sample, np.prod(prob_individual)

def additive_gaussian(state, std):
    sample, prob = sample_gaussian(std)
    return state + sample, prob

# What is robot_arm.getPointFeatures()
def point_feature_obs(state, points_2d, ctrnet, joint_angles, cam, cTr, gamma, thresh):
    T = poseToMatrix(state)


    # Want to have final result of project image plane coords
    _, t_list = ctrnet.robot.get_joint_RT(joint_angles)
    p_t = t_list[[0,2,3,4,6,7,8]] # remove 1 and 5 links as they are overlapping with 2 and 6
    K = ctrnet.intrinsics
    # only projects from camera coords to image coords
    p_c = None
    projected_points = cam.projectPoints(p_c)

    # Make association between detected and projected points to compute prob and use prob to update weights
    prob = 1
    return prob