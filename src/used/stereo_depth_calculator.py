import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

class StereoDepthCalculator:
    def __init__(self):
        # 初始化ROS节点和订阅者
        self.bridge = CvBridge()
        self.left_sub = rospy.Subscriber("/airsim_node/drone_1/front_left/Scene", Image, self.left_image_callback)
        self.right_sub = rospy.Subscriber("/airsim_node/drone_1/front_right/Scene", Image, self.right_image_callback)
        self.coords_sub = rospy.Subscriber("/original_pixel_coords", Point, self.coords_callback)
        self.depth_pub = rospy.Publisher("/average_depth", Float32, queue_size=10)
        self.left_image = None
        self.right_image = None
        self.pixel_coords = []

        # 投影矩阵参数
        self.mLeftM = np.array([[320, 0, 320, 0],
                                [0, 320, 240, 0],
                                [0, 0, 1, 0]])  # left P matrix
        self.mRightM = np.array([[320, 0, 320, -30.4],
                                 [0, 320, 240, 0],
                                 [0, 0, 1, 0]])  # right P matrix

    def left_image_callback(self, data):
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("Error converting left image: %s", e)

    def right_image_callback(self, data):
        try:
            self.right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("Error converting right image: %s", e)

    def coords_callback(self, data):
        # 更新像素坐标列表
        self.pixel_coords.append((data.x, data.y))

    def compute_depth(self):
        if self.left_image is not None and self.right_image is not None and self.pixel_coords:
            # ORB特征检测和匹配
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(self.left_image, None)
            kp2, des2 = orb.detectAndCompute(self.right_image, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # 计算深度
            depths = []
            for lx, ly in self.pixel_coords:
                match = min(matches, key=lambda m: np.linalg.norm(np.array(kp1[m.queryIdx].pt) - np.array([lx, ly])))
                rx, ry = kp2[match.trainIdx].pt
                depth = self.uvToXYZ(lx, ly, rx, ry)
                if depth is not None:
                    depths.append(depth)

            # 计算平均深度
            if depths:
                average_depth = np.mean(depths)
                self.depth_pub.publish(Float32(average_depth))
                self.pixel_coords = []  # 清空坐标列表以准备下一轮计算

    def uvToXYZ(self, lx, ly, rx, ry):
            # 使用第二个文件中的深度计算方法
            mLeftM = np.array([[320, 0, 320, 0],
                            [0, 320, 240, 0],
                            [0, 0, 1, 0]])  # left P矩阵

            mRightM = np.array([[320, 0, 320, -30.4],
                                [0, 320, 240, 0],
                                [0, 0, 1, 0]])  # right P矩阵

            A = np.zeros((4, 3))
            B = np.zeros((4, 1))
            for i in range(3):
                A[0][i] = lx * mLeftM[2, i] - mLeftM[0, i]
                A[1][i] = ly * mLeftM[2, i] - mLeftM[1, i]
                A[2][i] = rx * mRightM[2, i] - mRightM[0, i]
                A[3][i] = ry * mRightM[2, i] - mRightM[1, i]
                B[i][0] = mLeftM[i][3] - lx * mLeftM[2][3]
                B[i][0] = mRightM[i][3] - rx * mRightM[2][3]
            XYZ = np.zeros((3, 1))
            cv2.solve(A, B, XYZ, cv2.DECOMP_SVD)
            return XYZ[2][0]  # 返回计算出的深度数据

if __name__ == "__main__":
    rospy.init_node("stereo_depth_calculator")
    calculator = StereoDepthCalculator()
    rate = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown():
        calculator.compute_depth()
        rate.sleep()



