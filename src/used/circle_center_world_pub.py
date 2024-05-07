#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
import tf.transformations as transformations
import numpy as np
from nav_msgs.msg import Odometry

# 全局变量存储无人机的姿态和位置
drone_orientation = None
drone_position = None

# 全局变量存储平均深度值
average_depth = None

# 相机内参矩阵
camera_matrix = np.array([
    [320., 0., 320.],
    [0., 320., 240.],
    [0., 0., 1.]
])

# 相机到机体坐标的旋转矩阵
R_c_b = np.array([[0, 0, 1],                
                  [1, 0, 0],
                  [0, 1, 0]])

def imu_callback(imu_msg):
    global drone_orientation
    drone_orientation = imu_msg.orientation

def position_callback(position_msg):
    global drone_position
    drone_position = [position_msg.pose.pose.position.x, position_msg.pose.pose.position.y, position_msg.pose.pose.position.z]

def calculate_camera_extrinsics():
    if drone_orientation is not None and drone_position is not None:
        quaternion = [drone_orientation.x, drone_orientation.y, drone_orientation.z, drone_orientation.w]
        R_b_w = transformations.quaternion_matrix(quaternion)[:3, :3]
        T_b_w = np.array(drone_position)
        return R_b_w, T_b_w
    return None, None

def average_depth_callback(depth_msg):
    global average_depth
    average_depth = depth_msg.data

def circle_center_callback(data):
    global average_depth
    R_b_w, T_b_w = calculate_camera_extrinsics()
    if R_b_w is not None and T_b_w is not None and average_depth is not None:
        x_center = data.x
        y_center = data.y

        depth = average_depth

        # 像素坐标转相机坐标
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        x_camera = (x_center - cx) * depth / fx
        y_camera = (y_center - cy) * depth / fy
        z_camera = depth
        point_camera = np.array([x_camera, y_camera, z_camera])
        
        point_body = np.dot(np.array([[0, 0, 1],                
                                      [0, 1, 0],
                                      [1, 0, 0 ]])
                    ,point_camera) 

        # 相机坐标转机体坐标
        point_body = R_c_b.dot(point_camera)

        # 机体坐标转世界坐标
        point_world = R_b_w.dot(point_body) + T_b_w

        # 发布世界坐标
        world_point = Point()
        world_point.x, world_point.y, world_point.z = point_world
        world_coord_pub.publish(world_point)

if __name__ == '__main__':
    rospy.init_node('circle_center_world_coord')
    
    # IMU数据
    rospy.Subscriber('/airsim_node/drone_1/imu/imu', Imu, imu_callback)
    rospy.Subscriber('/vins_estimator/imu_propagate', Odometry, position_callback)
    
    # 订阅平均深度值
    rospy.Subscriber('/average_depth', Float32, average_depth_callback)

    rospy.Subscriber('/circle_center', Point, circle_center_callback)

    world_coord_pub = rospy.Publisher('/circle_center_world', Point, queue_size=10)

    rospy.spin()

