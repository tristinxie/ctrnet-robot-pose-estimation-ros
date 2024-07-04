#!/usr/bin/env python3
import sys
import os
base_dir = os.path.abspath(".")
sys.path.append(base_dir)

import numpy as np
import time
import rospy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs
import geometry_msgs
import kornia
import shutil

import torch
import torchvision.transforms as transforms
from multiprocessing.pool import Pool
from itertools import repeat

from PIL import Image as PILImage
from utils import *
from models.CtRNet import CtRNet

import cv2
bridge = CvBridge()

import transforms3d as t3d
import tf2_ros
from sklearn.metrics.pairwise import rbf_kernel
from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import stratified_resample

import matplotlib.patches as mpatches
#os.environ['ROS_MASTER_URI']='http://192.168.1.116:11311'
#os.environ['ROS_IP']='192.168.1.186'
from ros_nodes.particle_filter import *
from ros_nodes.probability_funcs import *
################################################################
import argparse
parser = argparse.ArgumentParser()

args = parser.parse_args("")

args.base_dir = "/home/workspace/src/ctrnet-robot-pose-estimation-ros/"
args.use_gpu = True
args.trained_on_multi_gpus = True
args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/panda/panda-3cam_azure/net.pth")
args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")
args.robot_name = 'Panda' # "Panda" or "Baxter_left_arm"
args.n_kp = 7
args.scale = 0.15625
args.height = 1536
args.width = 2048
args.fx, args.fy, args.px, args.py = 967.2597045898438, 967.2623291015625, 1024.62451171875, 772.18994140625
# scale the camera parameters
args.width = int(args.width * args.scale)
args.height = int(args.height * args.scale)
args.fx = args.fx * args.scale
args.fy = args.fy * args.scale
args.px = args.px * args.scale
args.py = args.py * args.scale

if args.use_gpu:
    device = "cuda"
else:
    device = "cpu"

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CtRNet = CtRNet(args)

def preprocess_img(cv_img,args):
    image_pil = PILImage.fromarray(cv_img)
    width, height = image_pil.size
    new_size = (int(width*args.scale),int(height*args.scale))
    image_pil = image_pil.resize(new_size)
    image = trans_to_tensor(image_pil)
    return image

shutil.rmtree("/home/workspace/src/ctrnet-robot-pose-estimation-ros/ros_nodes/visuals")
os.mkdir("/home/workspace/src/ctrnet-robot-pose-estimation-ros/ros_nodes/visuals")
#############################################################################3

#start = time.time()
visual_idx = 0

new_data = False
points_2d = None
image = None
joint_angles = None
cTr = None
joint_confidence = None
image_msg = None
def gotData(img_msg, joint_msg):
    #global start
    global new_data, points_2d, image, joint_angles, cTr, joint_confidence, image_msg
    image_msg = img_msg
    # print("Received data!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        image = preprocess_img(cv_img,args)

        joint_angles = np.array(joint_msg.position)[[0,1,2,3,4,5,6]]

        if args.use_gpu:
            image = image.cuda()

        cTr, points_2d, segmentation, joint_confidence = CtRNet.inference_single_image(image, joint_angles)

        
        #### visualization code ####
        #points_2d = points_2d.detach().cpu().numpy()
        #img_np = to_numpy_img(image)
        #img_np = overwrite_image(img_np,points_2d[0].astype(int), color=(1,0,0))
        #plt.imsave("test.png",img_np)
        ####
    except CvBridgeError as e:
        print(e)
    new_data = True
def visualize_panda(particles, joint_angles, cTr, image, points_2d, max_w_idx, points_2d_minus_one):
    global visual_idx
    base_dir = "/home/workspace/src/ctrnet-robot-pose-estimation-ros"
    mesh_files = [base_dir + "/urdfs/Panda/meshes/visual/link0/link0.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link1/link1.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link2/link2.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link3/link3.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link4/link4.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link5/link5.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link6/link6.obj",
              base_dir + "/urdfs/Panda/meshes/visual/link7/link7.obj",
              base_dir + "/urdfs/Panda/meshes/visual/hand/hand.obj",
             ]
    robot_renderer = CtRNet.setup_robot_renderer(mesh_files)
    robot_mesh = robot_renderer.get_robot_mesh(joint_angles)
    # rendered_image = CtRNet.render_single_robot_mask(cTr.squeeze().detach().cuda(), robot_mesh, robot_renderer)
    img_np = to_numpy_img(image)
    img_np = 0.0* np.ones(img_np.shape) + img_np * 0.6
    red = (1,0,0)
    green = (0,1,0)
    blue = (0,0,1)
    yellow = (1,1,0)
    img_np = overwrite_image(img_np, particles.reshape(-1, particles.shape[-1]), color=blue, point_size=1)
    img_np = overwrite_image(img_np, particles[max_w_idx, :, :], color=red, point_size=1)
    img_np = overwrite_image(img_np, points_2d.detach().cpu().numpy().squeeze().astype(int), color=green, point_size=3)
    img_np = overwrite_image(img_np, points_2d_minus_one.detach().cpu().numpy().squeeze().astype(int), color=yellow, point_size=3)

    plt.figure(figsize=(15,5))
    plt.title("keypoints")
    plt.imshow(img_np)
    colors = [blue, red, green, yellow]
    labels = ["Projected particles", "Max particle", "Current point2d", "Previous point2d"]
    patches = [ mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.grid(True)
    plt.savefig(f"/home/workspace/src/ctrnet-robot-pose-estimation-ros/ros_nodes/visuals/result{visual_idx}.png", dpi=800, format="png")
    visual_idx += 1
    if visual_idx == 100:
        quit()
    # input("Type Enter to continue")

def update_publisher(cTr, img_msg, qua, T):
    p = geometry_msgs.msg.PoseStamped()
    p.header = img_msg.header
    p.pose.position.x = T[0]
    p.pose.position.y = T[1]
    p.pose.position.z = T[2]
    p.pose.orientation.x = qua[0]
    p.pose.orientation.y = qua[1]
    p.pose.orientation.z = qua[2]
    p.pose.orientation.w = qua[3]
    #print(p)
    pose_pub.publish(p)

    # TODO: Not using the filtered output!
    # Rotating to ROS format
    cvTr= np.eye(4)
    # potentially problematic? https://github.com/kornia/kornia/issues/317
    cvTr[:3, :3] = kornia.geometry.conversions.angle_axis_to_rotation_matrix(cTr[:, :3]).detach().cpu().numpy().squeeze()
    cvTr[:3, 3] = np.array(cTr[:, 3:].detach().cpu())

    # ROS camera to CV camera transform
    cTcv = np.array([[0, 0 , 1, 0], [-1, 0, 0 , 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    T = cTcv@cvTr
    qua = t3d.quaternions.mat2quat(T[:3, :3]) # wxyz
    # Publish Transform
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "camera_base"
    t.child_frame_id = "panda_link0"
    t.transform.translation.x = T[0, 3]
    t.transform.translation.y = T[1, 3]
    t.transform.translation.z = T[2, 3]
    t.transform.rotation.x = qua[1]
    t.transform.rotation.y = qua[2]
    t.transform.rotation.z = qua[3]
    t.transform.rotation.w = qua[0]
    br.sendTransform(t)

if __name__ == "__main__":
    rospy.init_node('panda_pose')
    # Define your image topic
    image_topic = "/rgb/image_raw"
    robot_joint_topic = "/joint_states"
    robot_pose_topic = "robot_pose"
    # Set up your subscriber and define its callback
    #rospy.Subscriber(image_topic, sensor_msgs.msg.Image, gotData)

    image_sub = Subscriber(image_topic, sensor_msgs.msg.Image)
    robot_j_sub = Subscriber(robot_joint_topic, sensor_msgs.msg.JointState)
    pose_pub = rospy.Publisher(robot_pose_topic, geometry_msgs.msg.PoseStamped, queue_size=1)

    ats = ApproximateTimeSynchronizer([image_sub, robot_j_sub], queue_size=10, slop=5)
    ats.registerCallback(gotData)


    joint_confident_thresh = 7
    init_std = np.array([
                1.0e-2, 1.0e-2, 1.0e-2, # ori
                1.0e-3, 1.0e-3, 1.0e-3, # pos
            ])
    pf = ParticleFilter(num_states=6,
                        init_distribution=sample_gaussian,
                        motion_model=additive_gaussian,
                        obs_model=point_feature_obs,
                        num_particles=200)
    pf.init_filter(init_std)
    rospy.loginfo("Initailized particle filter")

    # Main loop:
    rate = rospy.Rate(30) # 30hz
    prev_cTr = None

    while not rospy.is_shutdown():
        if new_data:
            # print("here")
            # Copy all new data gotData
            # new_points_2d = torch.copy(points_2d)
            # new_image = torch.copy(image)
            # new_joint_angles = np.copy(joint_angles)
            new_cTr = torch.clone(cTr)
            new_joint_confidence = torch.clone(joint_confidence)
            new_image_msg = image_msg
            new_data = False

            if prev_cTr is None:
                prev_cTr = new_cTr
                continue

            # Skip if not CtRNet not confident in joints
            num_joint_confident = torch.sum(torch.gt(joint_confidence, 0.90))
            if num_joint_confident < joint_confident_thresh:
                print(f"Only confident with {num_joint_confident} joints, skipping...")
                continue
            
            # Predict Particle filter
            pred_std = np.array([1.0e-4, 1.0e-4, 1.0e-4,
                                2.5e-5, 2.5e-5, 2.5e-5])

            pf.predict(pred_std)

            # Update Particle filter
            cam = None
            gamma = 0.15
            pf.update(points_2d, CtRNet, joint_angles, cam, prev_cTr, gamma)

            mean_particle = pf.get_mean_particle()
            mean_particle_r = torch.from_numpy(mean_particle[:3])
            mean_particle_t = torch.from_numpy(mean_particle[3:])
            prev_cTr_r = prev_cTr[:, :3]
            prev_cTr_t = prev_cTr[:, 3:]
            pred_cTr = torch.zeros((1, 6))
            pred_cTr[0, :3] = prev_cTr_r.cpu() + mean_particle_r
            pred_cTr[0, 3:] = prev_cTr_t.cpu() + mean_particle_t
            # prev_cTr = pred_cTr
            pred_qua = kornia.geometry.conversions.angle_axis_to_quaternion(pred_cTr[:,:3]).detach().cpu() # xyzw
            pred_T = pred_cTr[:,3:].detach().cpu()
            update_publisher(cTr, new_image_msg, pred_qua.numpy().squeeze(), pred_T.numpy().squeeze())

        rate.sleep()
