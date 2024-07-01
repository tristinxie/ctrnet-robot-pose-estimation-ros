import sys
import os
import shutil
base_dir = "/home/workspace/src/ctrnet-robot-pose-estimation-ros"
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
args.use_gpu = False
args.trained_on_multi_gpus = True
args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/panda/panda-3cam_azure/net.pth")
args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")
args.robot_name = 'Panda' # "Panda" or "Baxter_left_arm"
args.n_kp = 7
args.scale = 0.15625
args.height = 1536
args.width = 2048
args.fx, args.fy, args.px, args.py = 960.41357421875, 960.22314453125, 1021.7171020507812, 776.2381591796875
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
#############################################################################3

def read_data_dir(data_dir_name):
    curr_dir = os.path.join(matlab_calib_dir, data_dir_name)
    image_dir = os.path.join(curr_dir, "images")
    pickle_path = os.path.join(curr_dir, "all_data.pkl")
    try:
        all_data = pickle.load(open(pickle_path, "rb"))
    except OSError as e:
        print(e)
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
    print(base_dir)
    robot_renderer = CtRNet.setup_robot_renderer(mesh_files)
    robot_mesh = robot_renderer.get_robot_mesh(all_data["all_joint_angles"][0])
    rendered_image = CtRNet.render_single_robot_mask(torch.from_numpy(all_data["all_cTr"][0].squeeze()), robot_mesh, robot_renderer)
    plt.imsave("test_rendered_image.jpg", rendered_image.squeeze().detach().cpu().numpy())

if __name__ == "__main__":
    data_dir_name = input("Input matlab directory name: ")
    read_data_dir(data_dir_name)


    # r_list, t_list = CtRNet.robot.get_joint_RT(joint_angles)
