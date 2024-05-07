#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# 初始化节点
rospy.init_node('image_hsv_segmentation', anonymous=True)

# 创建一个CvBridge实例
bridge = CvBridge()

# 创建发布者
image_pub = rospy.Publisher("/segmented_image_topic", Image, queue_size=10)

def image_callback(ros_image):
    try:
        # 将ROS图像消息转换为OpenCV图像
        current_frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)

    # 将BGR图像转换为HSV
    hsv_image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

    # 定义HSV中红色的范围
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([13, 255, 255])
    lower_red2 = np.array([150, 100, 108])
    upper_red2 = np.array([179, 255, 255])

    # 创建掩码以分离红色区域
    mask_red = cv2.inRange(hsv_image, lower_red1, upper_red1) | cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_not_red = cv2.bitwise_not(mask_red)

    # 创建一个全黑的底图
    black_background = np.zeros_like(current_frame)

    # 将红色区域设为白色，非红色区域设为黑色
    black_background[mask_red == 255] = [255, 255, 255]
    black_background[mask_not_red == 255] = [0, 0, 0]
    cv2.imshow("original Image", black_background)
    

    kernel = np.ones((2,2),np.uint8)

    ret1 = cv2.morphologyEx(black_background, cv2.MORPH_OPEN, kernel)
    # gaussian_blur = cv2.GaussianBlur(black_background, (5, 5), 0)

    # 显示经过高斯滤波的图像
    cv2.imshow("open Image", ret1)
    cv2.waitKey(3)

    # 将处理后的图像转换回ROS消息并发布
    try:
        image_pub.publish(bridge.cv2_to_imgmsg(ret1, "bgr8"))
    except CvBridgeError as e:
        print(e)

# 订阅裁剪后图像的话题
rospy.Subscriber("/cropped_image_topic", Image, image_callback)

# 开始循环
rospy.spin()



#全流程：
# 
# 
# 首先对detect.py中，新增分割函数。  分割后的小图：/cropped_image_topic
# 在hsv.py中，订阅/cropped_image_topic，实现色彩分割。输出分割后的黑白图：/segmented_image_topic
# 在corrd_point.py中，订阅/segmented_image_topic，结合yolo的/yolov5/detections输出的boundingboxes，输出原图中的坐标，发布话题：/original_pixel_coords
