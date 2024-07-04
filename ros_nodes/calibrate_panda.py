import sys
import os
import shutil
base_dir = os.path.abspath(".")
sys.path.append(base_dir)

import numpy as np
import rospy
from scipy.io import savemat
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs

import torch
import torchvision.transforms as transforms

from PIL import Image as PILImage
from utils import *
from models.CtRNet import CtRNet

import cv2
bridge = CvBridge()

import pickle
#os.environ['ROS_MASTER_URI']='http://192.168.1.116:11311'
#os.environ['ROS_IP']='192.168.1.186'

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

matlab_calib_dir = "/home/matlab_calibrate"
curr_dir = None
image_dir = None
#############################################################################3

image_idx = 0
all_bTe = None
all_joint_angles = None
all_cTr = None
all_points_2d = None
all_segmentation = None
all_joint_confidence = None
def gotData(img_msg, joint_msg):
    global all_bTe
    global all_joint_angles
    global all_cTr
    global all_points_2d
    global all_segmentation
    global all_joint_confidence
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
        r_list, t_list = CtRNet.robot.get_joint_RT(joint_angles)

        bTe = np.eye(4)
        bTe[:-1, :-1] = r_list[8]
        bTe[:-1, -1] = t_list[8]
        img_np = to_numpy_img(image)
        capture_data(img_np, bTe, joint_angles, cTr, points_2d, segmentation, joint_confidence)
        command = input(f"Enter to capture another image, {image_idx} so far. Any key to quit.") 
        if command != "":
            all_bTe_dict = {"armMat": all_bTe}
            all_data_dict = {"all_bTe": all_bTe, "all_joint_angles": all_joint_angles, "all_cTr": all_cTr, "all_points_2d": all_points_2d, "all_segmentation": all_segmentation, "all_joint_confidence": all_joint_confidence}
            pickle_path = os.path.join(curr_dir, "all_data.pkl")
            pickle.dump(all_data_dict, open(pickle_path, "wb"))
            mat_path = os.path.join(curr_dir, "arm_mat.mat")
            savemat(mat_path, all_bTe_dict)
            rospy.signal_shutdown("Done.")

    except CvBridgeError as e:
        print(e)

def capture_data(img_np, bTe, joint_angles, cTr, points_2d, segmentation, joint_confidence):
    global image_idx
    global all_bTe
    global all_joint_angles
    global all_cTr
    global all_points_2d
    global all_segmentation
    global all_joint_confidence
    
    cTr = cTr.detach().cpu().numpy()
    points_2d = points_2d.detach().cpu().numpy()
    segmentation = segmentation.detach().cpu().numpy()
    joint_confidence = joint_confidence.detach().cpu().numpy()
    if all_bTe is None:
        all_bTe = bTe.reshape(4,4,1)
        all_joint_angles = joint_angles.reshape(1,7)
        all_cTr = cTr.reshape(1,1,6)
        all_points_2d = points_2d.reshape(1,1,7,2)
        all_segmentation = segmentation.reshape(1,1,1,240,320)
        all_joint_confidence = joint_confidence.reshape(1,7)
    else:
        all_bTe = np.concatenate((all_bTe, bTe.reshape(4,4,1)), axis=2)
        all_joint_angles = np.concatenate((all_joint_angles, joint_angles.reshape(1,7)), axis=0)
        all_cTr = np.concatenate((all_cTr, cTr.reshape(1,1,6)), axis=0)
        all_points_2d = np.concatenate((all_points_2d, points_2d.reshape(1,1,7,2)), axis=0)
        all_segmentation = np.concatenate((all_segmentation, segmentation.reshape(1,1,1,240,320)))
        all_joint_confidence = np.concatenate((all_joint_confidence, joint_confidence.reshape(1,7)), axis=0)
    img_path = os.path.join(image_dir, f"image_{image_idx}.png")
    plt.imsave(img_path, img_np)
    image_idx += 1

def create_data_dir(data_dir_name):
    global curr_dir
    global image_dir
    curr_dir = os.path.join(matlab_calib_dir, data_dir_name)
    image_dir = os.path.join(curr_dir, "images")
    try:
        os.mkdir(curr_dir)
        os.mkdir(image_dir)
    except OSError as e:
        print(e)
        while True:
            command = input("Remove existing directory? y or n? ")
            if command == "y":
                shutil.rmtree(curr_dir)
                create_data_dir(data_dir_name)
                break
            elif command == "n":
                print("Stopping")
                sys.exit()
            else:
                print("Please enter y or n only")


if __name__ == "__main__":
    rospy.init_node('calibrate_panda')
    # Define your image topic
    image_topic = "/rgb/image_raw"
    robot_joint_topic = "/joint_states"

    image_sub = Subscriber(image_topic, sensor_msgs.msg.Image)
    robot_j_sub = Subscriber(robot_joint_topic, sensor_msgs.msg.JointState)

    data_dir_name = input("Name for data directory: ")
    create_data_dir(data_dir_name)

    ats = ApproximateTimeSynchronizer([image_sub, robot_j_sub], queue_size=10, slop=5)
    ats.registerCallback(gotData)


    # Main loop:
    rate = rospy.Rate(30) # 30hz

    while not rospy.is_shutdown():
        rate.sleep()