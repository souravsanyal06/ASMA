#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <mav_msgs/conversions.h>
#include <mav_msgs/default_topics.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <ros/ros.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/Vector3.h>



#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Joy.h>

#include <eigen_conversions/eigen_msg.h>

//#include "rotors_joy_interface/fake_driver.h"



// Use the structure definitions from the rotors_joy_interface 
// #include "rotors_joy_interface/joy.h"

#define DEG2RAD(x) ((x) / 180.0 * M_PI)

ros::Publisher trajectory_pub;
ros::Subscriber odom_sub, joy_sub, joy_enable_sub, takeoff_sub, land_sub, stop_sub, move_sub, traj_sub;
nav_msgs::Odometry odom_msg;
sensor_msgs::Joy joy_msg;

bool joy_msg_ready = false;
bool joy_enable = true;
bool init_pose_set = false;

bool start = false;
bool move = false;

bool takeoff = false;
bool emergency = false;
bool land = false;

float linear_x = 0.0;
float linear_y = 0.0;
float linear_z = 0.0;
float angular_x = 0.0;
float angular_y = 0.0;
float angular_z = 0.0;
float loc = 0.0;

int axis_roll  , axis_pitch, 
    axis_thrust, axis_yaw;
int axis_direction_roll, 
    axis_direction_pitch, 
    axis_direction_thrust, 
    axis_direction_yaw;
 
double max_vel,
       max_yawrate;

void traj_callback(
      const trajectory_msgs::MultiDOFJointTrajectoryConstPtr& trajectory_reference_msg);
void joy_callback(const sensor_msgs::JoyConstPtr& msg);
void odom_callback(const nav_msgs::OdometryConstPtr& msg);
void joy_enable_callback(const std_msgs::Bool& msg);
void TakeoffCallback(const std_msgs::Empty& msg);
void LandCallback(const std_msgs::Empty& msg);
void StopCallback(const std_msgs::Empty& msg);
void MoveCallback(const geometry_msgs::Twist& msg);
void StopMav();


int main(int argc, char** argv) {

  ros::init(argc, argv, "fake_driver");
  ros::NodeHandle nh("~");

  // Continuously publish waypoints.
  trajectory_pub = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>(
                    mav_msgs::default_topics::COMMAND_TRAJECTORY, 100);

  traj_sub = nh.subscribe("/bebop2/command/trajectory", 1, &traj_callback);

  // Subscribe to message for enabling/disabling joystick control
  joy_enable_sub = nh.subscribe("joy_enable", 10, &joy_enable_callback);

  // Subscribe to joystick messages
  joy_sub = nh.subscribe("joy", 10, &joy_callback);

  // Subscribe to gt odometry messages
  odom_sub = nh.subscribe("odom", 10, &odom_callback);


  takeoff_sub = nh.subscribe("/bebop/takeoff", 10, &TakeoffCallback);
  land_sub = nh.subscribe("/bebop/land", 10, &LandCallback);
  stop_sub = nh.subscribe("/bebop/reset", 10, &StopCallback);
  move_sub = nh.subscribe("/bebop/cmd_vel", 10, &MoveCallback);



  ROS_INFO("Started velocity_control_with_joyyyyy.");

  nh.param("axis_roll_"  , axis_roll, 3);
  nh.param("axis_pitch_" , axis_pitch, 2);
  nh.param("axis_yaw_"   , axis_yaw, 0);
  nh.param("axis_thrust_", axis_thrust, 1);

  nh.param("axis_direction_roll_"  , axis_direction_roll, -1);
  nh.param("axis_direction_pitch_" , axis_direction_pitch, 1);
  nh.param("axis_direction_yaw_"   , axis_direction_yaw, 1);
  nh.param("axis_direction_thrust_", axis_direction_thrust, -1);
 
  nh.param("max_vel", max_vel, 1.0);
  nh.param("max_yawrate", max_yawrate, DEG2RAD(45));
  
  ros::spin();
}

void traj_callback(
      const trajectory_msgs::MultiDOFJointTrajectoryConstPtr& msg){

      loc = msg->points[0].transforms[0].translation.z;
      //ROS_INFO("%f", loc);
}

void StopCallback(const std_msgs::Empty& msg){
  emergency = true;
  takeoff = false;
  start = false;
  ROS_INFO("STOP");
} 

void LandCallback(const std_msgs::Empty& msg){
  land = true;
  takeoff = false;
  start = false;
  ROS_INFO("LAND");

} 

void TakeoffCallback(const std_msgs::Empty& msg) 
{
  if (start){
    return;
  }
  takeoff = true;
  emergency = false;
  land = false;
  start = true;
  ROS_INFO("TAKEOFF");
}

void MoveCallback(const geometry_msgs::Twist& msg){
  if (takeoff || !start){
    return;
  }
  linear_x = msg.linear.x;
  linear_y = msg.linear.y;
  linear_z = msg.linear.z;
  angular_x = msg.angular.x;
  angular_y = msg.angular.y;
  angular_z = msg.angular.z;
} 

void joy_enable_callback(const std_msgs::Bool& msg)
{
  ROS_INFO("Changing joy_enable to %d", msg.data);
  joy_enable = msg.data;

  // If we are re-enabling the joystick the init_position might have changed, so make sure it's updated
  if (joy_enable == true) {
    init_pose_set = false;
  }
}

void joy_callback(const sensor_msgs::JoyConstPtr& msg){
  joy_msg = *msg;
  joy_msg_ready = true;
}

void odom_callback(const nav_msgs::OdometryConstPtr& msg){
  
 // if(joy_msg_ready == false || joy_enable == false)
 //   return;
 
  odom_msg = *msg;
  static ros::Time prev_time = ros::Time::now();
  static Eigen::Vector3d init_position;
  static double init_yaw;


  if(init_pose_set == false){
    Eigen::Affine3d eigen_affine;
    tf::poseMsgToEigen(odom_msg.pose.pose, eigen_affine);
    init_position = eigen_affine.matrix().topRightCorner<3, 1>();
    //init_yaw = eigen_affine.matrix().topLeftCorner<3, 3>().eulerAngles(0, 1, 2)(2);
    // Direct cosine matrix, a.k.a. rotation matrix
    Eigen::Matrix3d dcm = eigen_affine.matrix().topLeftCorner<3, 3>();
    double phi = asin(dcm(2, 1));
    double cosphi = cos(phi);
    double the = atan2(-dcm(2, 0) / cosphi, dcm(2, 2) / cosphi);
    double psi = atan2(-dcm(0, 1) / cosphi, dcm(1, 1) / cosphi);
    init_yaw = psi;
    init_pose_set = true;
  }  

  double roll = linear_y * axis_direction_roll;
  double pitch  = linear_x  * axis_direction_pitch;
  double thrust  = linear_z  * axis_direction_thrust;
  double yaw  = angular_z  * axis_direction_yaw;

  double dt = (ros::Time::now() - prev_time).toSec();
  prev_time = ros::Time::now();

  //double yaw    = joy_msg.axes[axis_yaw]    * axis_direction_yaw;
  //double roll   = joy_msg.axes[axis_roll]   * axis_direction_roll;
  //double pitch  = joy_msg.axes[axis_pitch]  * axis_direction_pitch;
  //double thrust = joy_msg.axes[axis_thrust] * axis_direction_thrust;


  if (emergency){
      static trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
      Eigen::Vector3d desired_dposition( cos(init_yaw) * pitch - sin(init_yaw) * roll, 
                                      sin(init_yaw) * pitch + cos(init_yaw) * roll, 
                                      5.0);
      init_position += desired_dposition * max_vel * dt;

      init_yaw = init_yaw + max_yawrate * yaw * dt;

      trajectory_msg.header.stamp = ros::Time::now();
      trajectory_msg.header.seq++;

      mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(init_position,
          init_yaw, &trajectory_msg);

      trajectory_msg.points[0].time_from_start = ros::Duration(0.01);
      trajectory_pub.publish(trajectory_msg);    
    if (loc < 0.0){
    takeoff = false;
    start = false;        
    emergency = false;
    }
  }


  if (land){
    static trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;

  
    // for loop avec z = 1??
    Eigen::Vector3d desired_dposition( cos(init_yaw) * pitch - sin(init_yaw) * roll, 
                                    sin(init_yaw) * pitch + cos(init_yaw) * roll, 
                                    -1.0);
    init_position += desired_dposition * max_vel * dt;

    init_yaw = init_yaw + max_yawrate * yaw * dt;

    trajectory_msg.header.stamp = ros::Time::now();
    trajectory_msg.header.seq++;

    mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(init_position,
        init_yaw, &trajectory_msg);

    trajectory_msg.points[0].time_from_start = ros::Duration(0.01);
    trajectory_pub.publish(trajectory_msg);
    if (loc < 0.0){
      start = false;
      land = false;
      takeoff = false;
    }
  }
  if (takeoff){
    static trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
    Eigen::Vector3d desired_dposition( cos(init_yaw) * pitch - sin(init_yaw) * roll, 
                                    sin(init_yaw) * pitch + cos(init_yaw) * roll, 
                                    5);
    init_position += desired_dposition * max_vel * dt;

    init_yaw = init_yaw + max_yawrate * yaw * dt;
    trajectory_msg.header.stamp = ros::Time::now();
    trajectory_msg.header.seq++;

    mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(init_position,
        init_yaw, &trajectory_msg);

    trajectory_msg.points[0].time_from_start = ros::Duration(0.01);
    trajectory_pub.publish(trajectory_msg);
    if (loc > 1.0){
      start = true;
      takeoff = false;
    }

  }

  if (start){
    static trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
    Eigen::Vector3d desired_dposition( cos(init_yaw) * pitch - sin(init_yaw) * roll, 
                                      sin(init_yaw) * pitch + cos(init_yaw) * roll, 
                                      thrust);
    init_position += desired_dposition * max_vel * dt;

    init_yaw = init_yaw + max_yawrate * yaw * dt;
    trajectory_msg.header.stamp = ros::Time::now();
    trajectory_msg.header.seq++;

    mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(init_position,
        init_yaw, &trajectory_msg);

    trajectory_msg.points[0].time_from_start = ros::Duration(1.0);
    trajectory_pub.publish(trajectory_msg);
  }
}

