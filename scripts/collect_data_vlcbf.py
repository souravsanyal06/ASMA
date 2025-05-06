#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CompressedImage, Range
from geometry_msgs.msg import Twist, PoseStamped
from cv_bridge import CvBridge
import cv2
import os
import csv

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector', anonymous=True)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/bebop2/camera_base/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/bebop2/depth_camera/depth/image_raw', Image, self.depth_callback)
        self.pose_sub = rospy.Subscriber('/bebop2/ground_truth/pose', PoseStamped, self.pose_callback)
        self.vel_sub = rospy.Subscriber('/bebop/cmd_vel', Twist, self.vel_callback)

        # Image bridge
        self.bridge = CvBridge()
        
        # Storage
        self.image_folder = "images"
        self.depth_folder = "depth"
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.depth_folder, exist_ok=True)
        self.csv_file = open("flight_data.csv", "w", newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "image_path", "depth_path", "pose", "velocity"])

        # Last data
        self.last_pose = None
        self.last_velocity = None

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        timestamp = rospy.Time.now()
        img_filename = os.path.join(self.image_folder, f"{timestamp}.jpeg")
        cv2.imwrite(img_filename, cv_image)
        self.save_data(timestamp, img_filename, None)

    def depth_callback(self, data):
        cv_depth = self.bridge.imgmsg_to_cv2(data, "32FC1")
        timestamp = rospy.Time.now()
        depth_filename = os.path.join(self.depth_folder, f"{timestamp}.png")
        cv2.imwrite(depth_filename, cv_depth)
        self.save_data(timestamp, None, depth_filename)

    def pose_callback(self, data):
        self.last_pose = (data.pose.position.x, data.pose.position.y, data.pose.position.z,
                          data.pose.orientation.x, data.pose.orientation.y,
                          data.pose.orientation.z, data.pose.orientation.w)

    def vel_callback(self, data):
        self.last_velocity = (data.linear.x, data.linear.y, data.linear.z,
                              data.angular.x, data.angular.y, data.angular.z)

    def save_data(self, timestamp, image_path, depth_path):
        if self.last_pose and self.last_velocity:
            self.csv_writer.writerow([timestamp, image_path, depth_path, self.last_pose, self.last_velocity])

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    collector = DataCollector()
    collector.run()

