import rospy
import numpy as np
import tf
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

class ASMA:
    def __init__(self, instruction_id):
        rospy.init_node('waypoint_follower', anonymous=True)
        self.cmd_pub = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=10)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.rate = rospy.Rate(10)  # 10 Hz control loop

        # Define waypoints for different instructions
        self.waypoints_sets = {
            1: [  # Instruction 1 
                (-27.9, -0.014, 1.22),
                (1.06, 0.76, 1.22),
                (20.82, 1.81, 1.22),
                (36.09, 0.605, 1.22),
                (51.36, -6.49, 0.44)
            ],
            2: [  # Instruction 2
                (-27.9, -0.014, 1.22),
                (0.10, -0.19, 1.22),
                (15.42, 2.29, 1.22),
                (20.0, 3.21, 1.22),
                (20.0, 8.5, 1.22),
                (20.0, 12.39, 1.22),
                (14.24, 14.9, 1.22),
                (7.92, 14.4, 1.22),
                (-2.66, 13.59, 1.22),
                (-7.29, 12.55, 1.22),
                (-12.55, 9.59, 1.22),
                (-12.91, -12.62, 1.22)
            ],
            3: [  # Instruction 3
                (-27.9, -0.014, 1.22),
                (-0.21, -0.13, 1.22),
                (31.59, -0.73, 1.22),
                (55.21, 0.95, 1.22),
                (78.11, 0.55, 1.22),
                (121.10, -0.21, 1.22),
                (134.05, -0.02, 1.22)
            ],
            4: [  # Instruction 4
                (-27.9, -0.014, 1.22),
                (6.24, 0.21, 1.22),
                (36.0, 0.14, 1.22),
                (35.0, 1.0, 1.22),
                (47.63, -7.67, 1.22),
                (46.99, -17.85, 1.22),
                (55.3, -22.44, 1.22),
                (69.20, -21.71, 1.22),
                (69.20, -21.71, 8.49),
                (77.56, -21.60, 8.49)
            ]
        }

        # Select waypoints based on instruction ID
        self.waypoints = self.waypoints_sets.get(instruction_id, self.waypoints_sets[1])

        self.close_proximity_threshold = 0.5  # Stop distance to waypoint
        self.safe_stopping_margin = 0.2
        self.dimension = 0.5  # Drone body radius
        self.safe_distance = 3.0  # Threshold distance for dynamic obstacle avoidance

        # Define all dynamic obstacles and the subset for the first crossing
        self.all_dynamic_obstacles = ['actor2', 'actor4', 'actor7', 'actor8', 'actor9']
        self.first_crossing_actors = ['actor1', 'actor3', 'actor5', 'actor6']

        self.trajectory = []

    def get_current_drone_state(self):
        """Retrieve the drone's current position (XY) and yaw from Gazebo."""
        try:
            drone_state = self.get_model_state('bebop2', '')
            position = np.array([drone_state.pose.position.x, drone_state.pose.position.y, drone_state.pose.position.z])
            quaternion = [
                drone_state.pose.orientation.x,
                drone_state.pose.orientation.y,
                drone_state.pose.orientation.z,
                drone_state.pose.orientation.w
            ]
            euler = tf.transformations.euler_from_quaternion(quaternion)
            yaw = euler[2]  # Extract the yaw angle
            return position, yaw
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)
            return None, None

    def angle_difference(self, current_yaw, target_yaw):
        """Compute the shortest angular difference between two angles."""
        angle_diff = target_yaw - current_yaw
        return (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

    def move_to_waypoint(self, target_pos, waypoint_index, instruction_id):
        """
        Navigate toward a waypoint while proactively avoiding dynamic obstacles.
        target_pos is a 3D tuple: (x, y, z).
        """
        # Select the obstacle list based on instruction_id and waypoint_index
        if instruction_id == 2:
            obstacle_list = self.first_crossing_actors
        else:
            if waypoint_index < 2:
                obstacle_list = self.first_crossing_actors
            else:
                obstacle_list = self.all_dynamic_obstacles

        # Threshold for deciding if we've matched altitude
        z_threshold = 0.2  # you can tune this

        while not rospy.is_shutdown():
            current_pos, current_yaw = self.get_current_drone_state()
            if current_pos is None:
                continue

            # Record current (x, y, z) in trajectory
            self.trajectory.append(current_pos)

            # ---- 2D horizontal distance ----
            distance_2d = np.linalg.norm(current_pos[:2] - np.array(target_pos[:2]))

            # ---- Vertical difference ----
            z_error = target_pos[2] - current_pos[2]

            # ---- Check if drone is close enough in both XY and Z ----
            arrived_2d = distance_2d < (self.close_proximity_threshold 
                                        + self.safe_stopping_margin 
                                        + self.dimension)
            arrived_z = abs(z_error) < z_threshold

            # If we are close in XY *and* we've matched altitude, we consider this waypoint reached
            if arrived_2d and arrived_z:
                rospy.loginfo(f"Reached waypoint: {target_pos}")
                self.cmd_pub.publish(Twist())  # publish zero velocities
                break

            # Compute nominal control command
            control_msg = Twist()

            # 1) Horizontal movement
            #    Move forward in XY plane toward the waypoint
            #    Speed is clamped based on distance_2d
            control_msg.linear.x = min(0.5, distance_2d)

            # 2) Yaw alignment (2D heading)
            angle_to_target = np.arctan2(target_pos[1] - current_pos[1],
                                        target_pos[0] - current_pos[0])
            angular_error = self.angle_difference(current_yaw, angle_to_target)
            control_msg.angular.z = 1.5 * angular_error

            # 3) Vertical movement
            #    Climb or descend to match target Z if there's a significant difference
            vertical_gain = 0.4          # Proportional gain
            vertical_speed = vertical_gain * z_error
            vertical_speed = np.clip(vertical_speed, -0.3, 0.3)  # clamp for safety
            control_msg.linear.z = vertical_speed

            # ---- Proactive dynamic obstacle avoidance (in 2D) ----
            for obs_name in obstacle_list:
                try:
                    obs_state = self.get_model_state(obs_name, '')
                    obs_pos = np.array([obs_state.pose.position.x, 
                                        obs_state.pose.position.y])
                    drone_xy = current_pos[:2]
                    obs_distance_2d = np.linalg.norm(drone_xy - obs_pos)

                    angle_to_obs = np.arctan2(obs_pos[1] - drone_xy[1],
                                            obs_pos[0] - drone_xy[0])
                    relative_angle = abs(self.angle_difference(current_yaw, angle_to_obs))

                    # If obstacle is too close and roughly in front
                    if obs_distance_2d < (self.safe_distance + 1.0) and relative_angle < np.radians(30):
                        rospy.loginfo(f"Actor {obs_name} predicted to intersect path. Initiating turn.")

                        # 2D cross product sign => turn direction
                        target_vector = np.array([target_pos[0] - drone_xy[0],
                                                target_pos[1] - drone_xy[1],
                                                0.0])
                        obs_vector = np.array([obs_pos[0] - drone_xy[0],
                                            obs_pos[1] - drone_xy[1],
                                            0.0])
                        cross_product = np.cross(target_vector, obs_vector)
                        turn_direction = int(np.sign(cross_product[2]))  # +1 or -1

                        # Adjust the command to avoid collision
                        control_msg.linear.x = 0.1
                        control_msg.angular.z = turn_direction
                        break  
                except rospy.ServiceException:
                    rospy.logwarn(f"Could not retrieve state for {obs_name}")
            
            self.cmd_pub.publish(control_msg)
            self.rate.sleep()

    def follow_trajectory(self, instruction_id):
        """Sequentially navigate through all waypoints and land at the final waypoint."""
        # import pdb; pdb.set_trace()   
        self.trajectory = []            
        waypoints_list = self.waypoints_sets[instruction_id]  # Retrieve the list of waypoints

        for idx, waypoint in enumerate(waypoints_list):            
            rospy.loginfo("Navigating to waypoint {}: {}".format(idx+1, waypoint))
            self.move_to_waypoint(waypoint, idx, instruction_id)


        # Land the drone after reaching the last waypoint
        self.land_drone()
        self.save_trajectory(instruction_id)

    def land_drone(self):
        """Land the drone at the final waypoint."""
        rospy.loginfo("Landing the drone at final waypoint.")
        landing_msg = Twist()
        landing_msg.linear.z = -0.2
        landing_duration = 5  

        for _ in range(landing_duration * 10):  
            self.cmd_pub.publish(landing_msg)
            self.rate.sleep()

        rospy.loginfo("Drone landed successfully.")
        self.cmd_pub.publish(Twist())


    def save_trajectory(self, instruction_id):
        """Save the collected 3D trajectory to an npy file and plot it."""
        if not self.trajectory:
            print(f"Trajectory for instruction {instruction_id} is empty. Skipping save.")
            return
        
        trajectory_np = np.array(self.trajectory)  # Convert list to NumPy array

        # Ensure it has 3D coordinates (X, Y, Z)
        if trajectory_np.shape[1] < 3:
            print(f"Trajectory data does not have enough dimensions: {trajectory_np.shape}. Skipping save.")
            return

        # Save the trajectory
        filename = f"/home/sourav/ASMA/dataset/trajectories/drone_trajectory_instr_{instruction_id}.npy"
        np.save(filename, trajectory_np)
        rospy.loginfo(f"Saved trajectory for instruction {instruction_id} as {filename}")

        # Plot the trajectory
        self.plot_trajectory(trajectory_np, instruction_id)

    def plot_trajectory(self, trajectory, instruction_id):
        """Visualize the saved 3D trajectory."""
        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            rospy.logwarn(f"Invalid trajectory data for instruction {instruction_id}. Skipping plot.")
            return

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Extract X, Y, Z coordinates
        x_positions = trajectory[:, 0]
        y_positions = trajectory[:, 1]
        z_positions = trajectory[:, 2]

        # Plot the trajectory in 3D without markers.
        # We use a simple line style and color; remove marker='o'.
        ax.plot(x_positions, y_positions, z_positions,
                linestyle='-', color='blue', label=f"Instruction {instruction_id}")

        # Set axis labels
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position (Altitude)")

        # Restrict Z range to 0–6
        ax.set_zlim(0, 6)

        ax.set_title(f"3D Drone Trajectory for Instruction {instruction_id}")
        ax.legend()
        ax.grid(True)

        plt.show()



if __name__ == "__main__":
    try:
        instruction_id = int(input("Enter instruction ID (1-4): "))  # User selects instruction
        if instruction_id not in [1, 2, 3, 4]:
            rospy.logwarn("Invalid instruction ID. Defaulting to Instruction 1.")
            instruction_id = 1

        asma = ASMA(instruction_id)
        asma.follow_trajectory(instruction_id)
    except rospy.ROSInterruptException:
        pass
