# iROS_drone

Pour toute question ou problème lié au TP drone (hors séance de TP), merci d'ouvrir une issue sur le repo suivant : https://github.com/simonernst/TP_drone/issues

Pour tout problème de compilation et d'erreurs liées à la simulation, merci de poser d'ouvrir une issue sur ce repo dans l'onglet "Issues"

## Prérequis

- Ubuntu 20.04
- ROS Noetic 
## Installation


```bash
mkdir -p bebop_ws/src && cd bebop_ws/src

sudo apt install build-essential python3-rosdep python3-catkin-tools
sudo apt install libusb-dev python3-osrf-pycommon libspnav-dev libbluetooth-dev libcwiid-dev libgoogle-glog-dev
sudo apt install ros-noetic-mavros ros-noetic-octomap-ros 

git clone https://github.com/ethz-asl/mav_comm
git clone -b noetic https://github.com/simonernst/iROS_drone
git clone https://github.com/ros-drivers/joystick_drivers

cd ..
catkin build
```

##### Lancement mode simulateur
```bash
roslaunch rotors_gazebo mav_velocity_control_with_fake_driver.launch
```
