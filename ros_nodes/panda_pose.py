#!/usr/bin/env python3
import sys
import os
import warnings
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
################################################################
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor
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
# args.fx, args.fy, args.px, args.py = 960.41357421875, 960.22314453125, 1021.7171020507812, 776.2381591796875
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

args.checkpoint = "/home/co-tracker/checkpoints/cotracker2.pth"
args.grid_size = 10
args.grid_query_frame = 0
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
window_frames = []
is_first_step = True
ctrnet_points = []
cotracker_query = None
pred_tracks = None
use_cotracker_for_ctrnet = True
def gotData(img_msg, joint_msg):
    global new_data, points_2d, image, joint_angles, cTr, joint_confidence, image_msg, window_frames, is_first_step, cotracker_query, window_frames, visual_idx, pred_tracks
    image_msg = img_msg
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        image = preprocess_img(cv_img,args)

        joint_angles = np.array(joint_msg.position)[[0,1,2,3,4,5,6]]

        if args.use_gpu:
            image = image.cuda()

        # CoTracker Start ##################################################################################
        # Start CoTracker on high confidence points

        if visual_idx % model.step == 0 and visual_idx != 0 and cotracker_query is not None:
            pred_tracks, pred_visibility = process_step_query(
                window_frames,
                is_first_step,
                query=cotracker_query
            )
            is_first_step = False
        frame = to_numpy_img(image)
        window_frames.append(frame)
        visual_idx += 1

        if pred_tracks is not None and use_cotracker_for_ctrnet:
            cotracker_points = pred_tracks[:, -1, :, :]
        else:
            cotracker_points = None
        # CoTracker End ##################################################################################
        cTr, points_2d, segmentation, joint_confidence = CtRNet.inference_single_image(image, joint_angles, cotracker_points)
        
        if cotracker_query is None:
            joint_confident_thresh = 7
            num_joint_confident = torch.sum(torch.gt(joint_confidence, 0.90))
            if num_joint_confident >= joint_confident_thresh:
                print("Created cotracker query")
                cotracker_query = torch.cat(((torch.ones((7,1))*visual_idx).to(device), points_2d.squeeze().to(device)), 1)
                print(len(window_frames))
                print(cotracker_query)
    
    except CvBridgeError as e:
        print(e)
    new_data = True

def update_publisher(cTr, img_msg, quaternion=True):
    cvTr= np.eye(4)
    if quaternion:
        cvTr[:3, :3] = kornia.geometry.conversions.quaternion_to_rotation_matrix(cTr[:, :4]).detach().cpu().numpy().squeeze()
        cvTr[:3, 3] = np.array(cTr[:, 4:].detach().cpu())
    else:
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

def process_step_query(window_frames, is_first_step, query):
    video_chunk = (
        torch.tensor(np.stack(window_frames[-model.step * 2 :]), device=device)
        .float()
        .permute(0, 3, 1, 2)[None]
    )  # (1, T, 3, H, W)
    return model(
        video_chunk,
        is_first_step=is_first_step,
        queries=query[None],
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="`XYZW` quaternion coefficient order is deprecated and will be removed after > 0.6. Please use `QuaternionCoeffOrder.WXYZ` instead.")
    # CoTracker
    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(device)

    # Particle filter
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


    init_std = np.array([
                1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, # ori
                1.0e-3, 1.0e-3, 1.0e-3, # pos
            ])
    pf = ParticleFilter(num_states=7,
                        init_distribution=sample_gaussian,
                        motion_model=additive_gaussian,
                        obs_model=point_feature_obs,
                        num_particles=2000)
    pf.init_filter(init_std)
    rospy.loginfo("Initailized particle filter")

    # Main loop:
    rate = rospy.Rate(30) # 30hz
    prev_cTr = None
    use_particle_filter = False
    visualize_cotracker = False
    while not rospy.is_shutdown():
        try:
            if new_data:
                # print("here")
                # Copy all new data gotData
                # new_image = torch.copy(image)
                # new_joint_angles = np.copy(joint_angles)
                new_points_2d = torch.clone(points_2d)
                new_cTr = torch.clone(cTr)
                new_joint_confidence = torch.clone(joint_confidence)
                new_image_msg = image_msg
                new_image = image
                new_data = False
                ctrnet_points.append(new_points_2d.detach().cpu().numpy())

                if use_particle_filter == False:
                    print("PARTICLE FILTER TURNED OFF")
                    # pred_qua = kornia.geometry.conversions.angle_axis_to_quaternion(cTr[:,:3]).detach().cpu() # xyzw
                    pred_T = cTr[:,3:].detach().cpu()
                    update_publisher(new_cTr, new_image_msg, quaternion=False)
                    # update_publisher(new_cTr, new_image_msg, pred_qua.cpu().detach().numpy().squeeze(), pred_T.cpu().detach().numpy().squeeze())
                    continue

                if prev_cTr is None:
                    prev_cTr = new_cTr
                    continue

                # Predict Particle filter
                pred_std = np.array([1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4,
                                    2.5e-5, 2.5e-5, 2.5e-5])

                pf.predict(pred_std)

                # Update Particle filter
                cam = None
                gamma = 0.15
                pf.update(new_points_2d, cotracker_points, CtRNet, joint_angles, cam, prev_cTr, gamma)

                mean_particle = pf.get_mean_particle()
                mean_particle_r = torch.from_numpy(mean_particle[:4])
                mean_particle_t = torch.from_numpy(mean_particle[4:])

                prev_cTr_r = kornia.geometry.conversions.angle_axis_to_quaternion(prev_cTr[:, :3])
                prev_cTr_t = prev_cTr[:, 3:]

                pred_cTr = torch.zeros((1, 7))
                pred_cTr[0, :4] = prev_cTr_r.cpu() + mean_particle_r
                pred_cTr[0, 4:] = prev_cTr_t.cpu() + mean_particle_t
                # good ctr: tensor([[ 0.3882,  0.7024, -0.5301,  0.2847,  0.1452,  0.3092,  1.0225]]
                print(pred_cTr)
                # pred_qua = kornia.geometry.conversions.angle_axis_to_quaternion(pred_cTr[:,:3]).detach().cpu() # xyzw
                # pred_T = pred_cTr[:,3:].detach().cpu()
                # update_publisher(pred_cTr, new_image_msg, pred_qua.cpu().detach().numpy().squeeze(), pred_T.cpu().detach().numpy().squeeze())
                update_publisher(pred_cTr, new_image_msg)
            rate.sleep()
        except KeyboardInterrupt:
            rospy.signal_shutdown("here")
    if visualize_cotracker:
        # Processing the final video frames in case video length is not a multiple of model.step
        pred_tracks, pred_visibility = process_step_query(
            window_frames[-(visual_idx % model.step) - model.step - 1 :],
            is_first_step,
            query=cotracker_query
        )
        print("Tracks are computed")
        # save a video with predicted tracks
        seq_name = "occlusion_panda"
        video = torch.tensor(np.stack(window_frames), device=device).permute(0, 3, 1, 2)
        video = (video*255)[None]
        vis = Visualizer(save_dir="/home/workspace/src/ctrnet-robot-pose-estimation-ros/ros_nodes/visuals", linewidth=1, tracks_leave_trace=-1)
        ctrnet_points = torch.tensor(np.stack(ctrnet_points), device=device).permute(1, 0, 2, 3)
        print(pred_tracks.shape)
        print(ctrnet_points.shape)
        vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename=f"cotracker_{seq_name}")
        vis.visualize(video=video, tracks=ctrnet_points, filename=f"ctrnet_{seq_name}")
        # vis.visualize(video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=seq_name)