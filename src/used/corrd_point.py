#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from detection_msgs.msg import BoundingBoxes
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# 初始化ROS节点
rospy.init_node('calculate_original_coordinate')

# 创建CvBridge实例
bridge = CvBridge()

# 全局变量存储bounding box的坐标
bbox_coord = None

def bbox_callback(data):
    global bbox_coord
    if data.bounding_boxes:
        bbox = data.bounding_boxes[0]  # 获取第一个bounding box
        bbox_coord = (bbox.xmin, bbox.ymin)
        # rospy.loginfo("%d,%d",bbox.xmin,bbox.ymin)



def image_callback(ros_image):
    global bbox_coord
    if bbox_coord is None:
        return

    try:
        # 将ROS图像消息转换为OpenCV图像
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    # 找到所有白色像素的坐标
    white_pixels = np.where(cv_image == 255)
    for y, x in zip(white_pixels[0], white_pixels[1]):
        # 考虑边界框的坐标
        original_x = x + bbox_coord[0]
        original_y = y + bbox_coord[1]

        # 发布每个像素的坐标
        pixel_coord = Point()
        pixel_coord.x = original_x
        pixel_coord.y = original_y
        
        #rospy.loginfo("%d,%d",original_x,original_y)
        coord_pub.publish(pixel_coord)

# 创建发布者，发布原始图像中白色像素的坐标
coord_pub = rospy.Publisher("/original_pixel_coords", Point, queue_size=10)

# 订阅分割后图像的话题
rospy.Subscriber("/segmented_image_topic", Image, image_callback)

# 订阅YOLO检测到的边界框的话题
rospy.Subscriber("/yolov5/detections", BoundingBoxes, bbox_callback)

# 开始ROS循环
rospy.spin()
