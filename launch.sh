#!/usr/bin/env bash
source /opt/ros/noetic/setup.bash
source /home/sourav/ASMA/ros_ws/devel/setup.bash
source ~/killross.sh
roslaunch rotors_gazebo test_new.launch lockstep:=true &
rosrun keyboard keyboard &
#rosrun rviz rviz &

