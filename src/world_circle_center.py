#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge          import CvBridge
from sensor_msgs.msg    import Image
from nav_msgs.msg       import Odometry
from geometry_msgs.msg  import PoseStamped
from sensor_msgs.msg    import Imu
from geometry_msgs.msg import Point
from yolov5_ros.msg import CircleCenterPoint
from std_msgs.msg import Float64 


def quaternion_to_rotation_matrix(w,x,y,z):
    """ 
    Convert a quaternion to a rotation matrix.
    
    Parameters:
    q : list or numpy array
        A quaternion [w, x, y, z]
        
    Returns:
    R : 3x3 numpy array
        Rotation matrix
    """

    tx = 2 * x
    ty = 2 * y
    tz = 2 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    res = np.zeros((3, 3))
    res[0, 0] = 1 - (tyy + tzz)
    res[0, 1] = txy - twz
    res[0, 2] = txz + twy
    res[1, 0] = txy + twz
    res[1, 1] = 1 - (txx + tzz)
    res[1, 2] = tyz - twx
    res[2, 0] = txz - twy
    res[2, 1] = tyz + twx
    res[2, 2] = 1 - (txx + tyy)
    return res

def append_with_limit(arr, value):
    """
    Appends a value to the array. 
    If the array length exceeds the limit, the first value is discarded.
    """
    arr = np.vstack((arr, value))
    while len(arr) > 5:
        arr = np.delete(arr, 0, axis=0)
    return arr


class CircleDetectNode:
    def __init__(self):

        self.flagCircle = False
        # 圆心标志位
        self.got_circle_flag_R = False
        self.got_circle_flag_L = False

        self.circle_radius = 0

        # 用于存储待计算的点的集合
        self.point_array = np.zeros((5, 3))

        # R_b_w 来自 IMU
        self.R_b_w          =   np.array([[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 1]])
        
        # R_c_b 固定
        self.R_c_b          =   np.array([[0, 0, 1],
                                          [1, 0, 0],
                                          [0, 1, 0]])
        
        # T_b_w来自Vins
        self.T_b_w          =   np.array([[0],
                                          [0],
                                          [0]])
        
        # 左右圆心
        self.center_R = (0,0)
        self.center_L = (0,0)   

        #半径初始化
        self.circle_radius = 0


        # 数据处理
        self.bridge    = CvBridge()
        self.vins_sub  = rospy.Subscriber("/vins_estimator/imu_propagate" ,           Odometry,          self.vins_callback       )
        self.pose_sub = rospy.Subscriber("/airsim_node/drone_1/debug/pose_gt",        PoseStamped,       self.pose_callback)
        self.imu_sub   = rospy.Subscriber("/airsim_node/drone_1/imu/imu",             Imu,               self.imu_callback        )


        self.circle_radius_sub = rospy.Subscriber("/circle_radius", Float64, self.circle_radius_callback)

        self.left_circle_sub = rospy.Subscriber("/left_circle_center",Point , self.left_circle_callback)
        self.right_circle_sub = rospy.Subscriber("/right_circle_center",Point , self.right_circle_callback)
        self.circle_pub= rospy.Publisher ("/circle_center_pos",           CircleCenterPoint, queue_size=10            )

        # self.gray_left_pub = rospy.Publisher("/gray_left_image", Image, queue_size=10)
        # self.gray_right_pub = rospy.Publisher("/gray_right_image", Image, queue_size=10)

        self.timer     = rospy.Timer     (rospy.Duration(0.01),                               self.timer_callback      )




    def vins_callback(self, msg):
        # Vins 获取 T_b_w
        self.T_b_w = np.array([[msg.pose.pose.position.x], [-msg.pose.pose.position.y], [-msg.pose.pose.position.z]])
        # pass

    def pose_callback(self, msg):
        # self.T_b_w = np.array([[msg.pose.position.x], [msg.pose.position.y], [msg.pose.position.z]])
        pass

    def imu_callback(self, msg): 
        # IMU 获取 R_b_w
        self.R_b_w = quaternion_to_rotation_matrix(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)
   
    def uvToXYZ(self,lx, ly, rx, ry):
            
            # Step 1:
            #      从左右目像素坐标得到一个深度数据

            mLeftM  = np.array([[320, 0,   320, 0   ],
                                [0,   320, 240, 0   ],
                                [0,   0,   1,   0   ]])  # left P矩阵

            mRightM = np.array([[320, 0,   320, -30.4],
                                [0,   320, 240, 0   ],
                                [0,   0,   1,   0   ]])  # right P矩阵
            
            A = np.zeros(shape=(4, 3))
            for i in range(0, 3):
                A[0][i] = lx * mLeftM[2, i] - mLeftM[0][i]
            for i in range(0, 3):
                A[1][i] = ly * mLeftM[2][i] - mLeftM[1][i]
            for i in range(0, 3):
                A[2][i] = rx * mRightM[2][i] - mRightM[0][i]
            for i in range(0, 3):
                A[3][i] = ry * mRightM[2][i] - mRightM[1][i]

            B = np.zeros(shape=(4, 1))
            for i in range(0, 2):
                B[i][0] = mLeftM[i][3] - lx * mLeftM[2][3]
            for i in range(2, 4):
                B[i][0] = mRightM[i - 2][3] - rx * mRightM[2][3]

            XYZ = np.zeros(shape=(3, 1))
            cv2.solve(A, B, XYZ, cv2.DECOMP_SVD)
            # 此处的  XYZ[2][0]  为计算出的深度数据

            # Step 2:
            #      使用深度数据和像素坐标运算

            #相机内参
            K          = np.array([[320, 0,   320],                 
                                   [0,   320, 240],
                                   [0,   0,   1  ]])
            
            #！！！！！！！！！！！！！！！！！！！！！！！！
            depth = XYZ[2][0] +0.10
            #！！！！！！！！！！！！！！！！！！！！！！！！


            point_pixel= np.array([[0.5*(lx+rx) * depth],
                                   [0.5*(ly+ry) * depth],
                                   [depth]               ])
            #像素坐标转相机坐标
            point_camera = np.dot(np.linalg.inv(K) , point_pixel)   

            #相机坐标转机体坐标
            point_body = np.dot(np.array([[0, 0, 1],                
                                          [1, 0, 0],
                                          [0, 1, 0 ]])
                        ,point_camera) + np.array([[0.26],[0],[0]])
            
            #机体坐标转世界坐标
            point_world = np.dot(self.R_b_w,point_body )+self.T_b_w 
            point_world_1x3 = point_world.reshape(1,3)

            #返回一个 1x3 的 数组
            return point_world_1x3           




########################################################################################################################################################################
    def left_circle_callback(self,msg):

        self.center_L = (msg.x,msg.y)
        self.got_circle_flag_L = True

    def right_circle_callback(self,msg):

        self.center_R = (msg.x,msg.y)
        self.got_circle_flag_R = True
########################################################################################################################################################################




    def circle_radius_callback(self, msg):
        self.circle_radius = msg.data




    def timer_callback(self,event):

        self.flagCircle = False
        circle_msg = CircleCenterPoint()

        Point_c_w = self.uvToXYZ(self.center_L[0],self.center_L[1],self.center_R[0],self.center_R[1])

        self.point_array = append_with_limit(self.point_array,Point_c_w)



        if  self.got_circle_flag_L  and self.got_circle_flag_R :             
            # variances = np.var(self.point_array, axis=0)
            # # 满足条件，认为数据可信
            # if (variances < np.array([10,10,10])).all():
                mean_values = np.mean(self.point_array, axis=0)
                self.flagCircle = True
                magic_param = [0.563, 0.310, 0.118]    # 圆心坐标修正量
                circle_msg.x             = mean_values[0] + magic_param[0]
                circle_msg.y             = mean_values[1] + magic_param[1]
                circle_msg.z             = mean_values[2] + magic_param[2]
                circle_msg.r             = self.circle_radius
                self.circle_pub.publish(circle_msg)     # 发布消息
                print('---Circle position published!---')
                print(mean_values)
            # else:
            #     print("---Variance is too large! Stablizing...---")

            	
# 主函数
if __name__ == "__main__":

    rospy.init_node("circle_detect_node")
    node = CircleDetectNode()
    rospy.spin()
