#!/usr/bin/env python3
import rospy
from detection_msgs.msg import BoundingBoxes
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Float64  # 导入 Float64 类型

class LeftCircleCenterNode:
    def __init__(self):
        rospy.init_node('left_circle_center_node', anonymous=True)

        # 订阅left_cropped_image_topic
        self.image_subscriber = rospy.Subscriber("/left_cropped_image_topic", Image, self.left_image_callback)

        # 订阅YOLO检测到的边界框的话题
        self.bbox_subscriber = rospy.Subscriber("/yolov5/left_detections", BoundingBoxes, self.bbox_callback)

        # 发布坐标的话题
        self.left_circle_center = rospy.Publisher("/left_circle_center", Point, queue_size=10)


        #发布半长轴
        self.circle_radius_publisher = rospy.Publisher("/circle_radius", Float64, queue_size=10)

        # 初始化椭圆检测相关参数
        self.got_circle_flag_L = False
        self.center_L = None

        self.circle_radius = 0.0  # 初始化半长轴长度


        self.bridge = CvBridge()

        # 全局变量存储bounding box的坐标
        self.bbox_coord = None

    def hsv_mask(self, img):
        # 转换hsv
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 红色掩膜
        lower_hsv_red1 = np.array([0, 100, 100])
        upper_hsv_red1 = np.array([13, 255, 255])
        mask_red1 = cv2.inRange(hsv_image, lower_hsv_red1, upper_hsv_red1)
        lower_hsv_red2 = np.array([150, 100, 108])
        upper_hsv_red2 = np.array([179, 255, 255])
        mask_red2 = cv2.inRange(hsv_image, lower_hsv_red2, upper_hsv_red2)
        # 合并掩膜
        mask_red_C = cv2.bitwise_or(mask_red1, mask_red2)
        # hsv掩膜
        hsv_red = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_red_C)

        # 黄色掩膜
        lower_hsv_yellow1 = np.array([20, 50, 120])
        upper_hsv_yellow1 = np.array([40, 235, 235])
        mask_yellow1 = cv2.inRange(hsv_image, lower_hsv_yellow1, upper_hsv_yellow1)
        # hsv掩膜
        hsv_yellow = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_yellow1)
        # 合并图像
        result_image = cv2.add(hsv_red, hsv_yellow)

        # 高斯滤波
        filtered_img = cv2.GaussianBlur(result_image, (5, 5), 0)

        return filtered_img

    def img_process(self, img):
        # 中值滤波
        image = cv2.medianBlur(img, 3)  # 5
        # 高斯滤波
        image = cv2.GaussianBlur(image, (5, 5), 0)
        # 转换灰度
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 阈值分割
        threshold_value = 90
        _, result_image = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
        # 进行开操作
        Result = cv2.morphologyEx(result_image, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
        return Result

    # boundingbox左上坐标
    def bbox_callback(self, data):
        if data.bounding_boxes:
            bbox = data.bounding_boxes[0]  # 获取第一个bounding box
            self.bbox_coord = (bbox.xmin, bbox.ymin)
            # rospy.loginfo("%d,%d",bbox.xmin,bbox.ymin)

    def left_image_callback(self, msg):
        try:
            # cv bridge
            left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 转换hsv
            left_image = self.hsv_mask(left_image)
            # 预处理
            left_image = self.img_process(left_image)
            # 寻找轮廓
            contours, _ = cv2.findContours(left_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            self.got_circle_flag_L = False

            # 对轮廓进行处理
            max_contour = max(contours, key=cv2.contourArea, default=None)

            if max_contour is not None and cv2.contourArea(max_contour) > 1000 and len(max_contour) >= 30:
                # 椭圆检测
                ellipse = cv2.fitEllipse(max_contour)
                # 算数平均值
                self.circle_radius =  ellipse[1][0]
                # 得到圆心
                self.center_L = (ellipse[0][0], ellipse[0][1])
                # 绘制椭圆
                cv2.ellipse(left_image, ellipse, (255, 255, 0), 1)
                cv2.circle(left_image, (int(ellipse[0][0]), int(ellipse[0][1])), 1, (255, 255, 0), 1)
                self.got_circle_flag_L = True
        except Exception as e:
            rospy.logerr("Error processing left image: %s", str(e))

        # 相加，发布
        if self.center_L is not None and self.bbox_coord is not None:
            left_circle_center = Point()
            left_circle_center.x = self.center_L[0] + self.bbox_coord[0]
            left_circle_center.y = self.center_L[1] + self.bbox_coord[1]
            self.left_circle_center.publish(left_circle_center)

            #发布半长轴
            self.circle_radius_publisher.publish(self.circle_radius)

if __name__ == '__main__':
    LeftCircleCenterNode()
    rospy.spin()
