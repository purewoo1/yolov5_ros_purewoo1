import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

# 初始化ROS节点
rospy.init_node('depth_value_average_calculator')

# 创建CvBridge实例
bridge = CvBridge()

# 存储最近接收到的深度图
current_depth_image = None

def depth_image_callback(depth_image_msg):
    global current_depth_image
    try:
        # 将深度图从ROS消息转换为OpenCV图像
        current_depth_image = bridge.imgmsg_to_cv2(depth_image_msg, "16SC1")
        #rospy.loginfo("Depth image received.")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def process_depth_image():
    global current_depth_image
    if current_depth_image is None:
        rospy.logwarn("No depth image received yet.")
        return

    try:
        # 提取有效深度值（大于0的深度值）
        valid_depths = current_depth_image[current_depth_image > 0]

        # 判断是否有有效的深度数据
        if valid_depths.size > 0:
            #  计算均值和标准差
            mean_depth = np.mean(valid_depths)
            std_depth = np.std(valid_depths)

            #  确定筛选阈值
            lower_bound = mean_depth - 3 * std_depth
            upper_bound = mean_depth + 1 * std_depth

            #  筛选深度值
            # 只保留均值？个标准差内的值
            filtered_depth_values = valid_depths[(valid_depths >= lower_bound) & (valid_depths <= upper_bound)]

            # 步骤4: 计算筛选后深度值的平均值
            if filtered_depth_values.size > 0:
                average_depth = np.mean(filtered_depth_values)
                rospy.loginfo("Filtered average depth: {}".format(average_depth))
                # 发布筛选后的平均深度值
                average_depth_pub.publish(Float32(average_depth))
            else:
                rospy.logwarn("No valid depth values found within the specified range.")
        else:
            rospy.logwarn("No valid depth values found.")

    except Exception as e:
        rospy.logerr("Error processing the depth image: {}".format(e))

# 创建发布者，发布平均深度值
average_depth_pub = rospy.Publisher("/average_depth", Float32, queue_size=10)

# 订阅深度图话题
depth_image_sub = rospy.Subscriber("/depth_image", Image, depth_image_callback)

# 主循环
if __name__ == '__main__':
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        process_depth_image()
        rate.sleep()
