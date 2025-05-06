# iROS_drone

## Prérequis

- Ubuntu 18.04
- ROS Kinetic + Gazebo 9

## Prise en main et installation

1. Créez votre workspace (cf Tuto ROS si besoin)
2. Depuis votre workspace

```bash
sudo apt-get install build-essential python-rosdep python-catkin-tools
cd $Votre_Workspace/src
git clone -b https://github.com/simonernst/iROS_drone
git clone https://github.com/ros-drivers/joystick_drivers
git clone https://github.com/AutonomyLab/bebop_autonomy
cd ..
rosdep update
rosdep install --from-paths src -i
catkin build
```

Ne pas oublier de sourcer son workspace après chaque compilation (```source devel/setup.bash```). Je vous recommande de l'écrire dans votre .bashrc

## Utilisation


Lancement mode simulateur
```bash
roslaunch rotors_gazebo mav_velocity_control_with_fake_driver.launch
```
Lancement mode réel
```bash
roslaunch bebop_driver bebop_node.launch
```

## Debug
