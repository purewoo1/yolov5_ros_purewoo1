#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# 初始化节点
rospy.init_node('stereo_vision_node')

# 初始化 cv_bridge
bridge = CvBridge()

# 相机参数
camera_matrix = np.array([
    [320., 0., 320.],
    [0., 320., 240.],
    [0., 0., 1.]
])
distortion = np.array([0, 0, 0, 0, 0])  # 没有畸变

# 立体相机设置
R = np.eye(3)  # 单位旋转矩阵
T = np.array([0, 0.095, 0])  # 平移矩阵
size = (640, 480)  # 图像尺寸

# 计算立体校正参数
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    camera_matrix, distortion, camera_matrix, distortion, size, R, T
)

# 计算去畸变校正映射
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    camera_matrix, distortion, R1, P1, size, cv2.CV_16SC2
)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    camera_matrix, distortion, R2, P2, size, cv2.CV_16SC2
)

# 初始化 SGBM 参数
blockSize = 9  # 增大以获得更好的平滑性
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*4,
    blockSize=blockSize,
    P1=8*blockSize**2,
    P2=32*blockSize**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,  # 增加
    speckleWindowSize=150,  # 增加以获得更连贯的视差区域
    speckleRange=2,  # 调整
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# 函数用于应用后处理滤波器
def filter_disparity(disparity):
    # 使用中值滤波器减少噪声
    return cv2.medianBlur(disparity, 5)

# 回调函数处理每个图像
left_image = None
right_image = None

def left_image_callback(msg):
    global left_image
    left_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def right_image_callback(msg):
    global right_image, depth_pub
    right_image = bridge.imgmsg_to_cv2(msg, "bgr8")

    if left_image is not None and right_image is not None:
        # 处理左右图像
        left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)

        # 计算视差
        raw_disparity = stereo.compute(left_rectified, right_rectified)

        # 应用后处理滤波器
        
        filtered_disparity = filter_disparity(raw_disparity)

        # 视差图标准化用于显示
        disp_display = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 显示深度图
        cv2.imshow('Disparity', disp_display)
        cv2.waitKey(1)

        # 将深度图转换为 ROS Image 消息并发布
        depth_image_msg = bridge.cv2_to_imgmsg(filtered_disparity, "16SC1")
        depth_pub.publish(depth_image_msg)

# 订阅立体相机话题
rospy.Subscriber("/airsim_node/drone_1/front_left/Scene", Image, left_image_callback)
rospy.Subscriber("/airsim_node/drone_1/front_right/Scene", Image, right_image_callback)

# 初始化深度图发布器
depth_pub = rospy.Publisher("/depth_image", Image, queue_size=10)

# ROS spin 以保持脚本运行
rospy.spin()




