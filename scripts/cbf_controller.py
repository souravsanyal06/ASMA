import numpy as np
from geometry_msgs.msg import Twist

class CBF:
    def __init__(self, safe_distance, alpha=1):
        self.safe_distance = safe_distance
        self.alpha = alpha  # Class K function parameter

    def compute_control(self, current_position, target_position, obstacles):
        twist = Twist()
        min_distance = float('inf')
        target_vector = np.array(target_position[:2]) - np.array(current_position[:2])
        target_distance = np.linalg.norm(target_vector)

        if target_distance == 0:
            return twist  # Avoid division by zero, no movement if on target

        for obs_name, obs_pos in obstacles:
            obs_vector = np.array(obs_pos[:2]) - np.array(current_position[:2])
            obs_distance = np.linalg.norm(obs_vector)
            if obs_distance < min_distance:
                min_distance = obs_distance
                cos_angle = np.clip(np.dot(target_vector, obs_vector) / (target_distance * obs_distance + 1e-8), -1.0, 1.0)
                angle = np.arccos(cos_angle)

                # Adjust controls based on proximity to the obstacle
                if obs_distance < self.safe_distance and angle < np.radians(30):
                    # Compute cross product to determine direction of adjustment
                    cross_product = np.cross(np.append(target_vector, 0), np.append(obs_vector, 0))

                    # Apply discrete adjustment for angular.z and angular.y
                    twist.angular.z = int(np.sign(cross_product[2]))  # Ensure adjustment is exactly 1 or -1
                    twist.angular.y = int(np.sign(cross_product[2]))  # Apply same logic for roll

                    # Reduce linear velocity as the obstacle approaches
                    twist.linear.x = 0.1  # Minimal forward speed to maintain motion

        if twist.linear.x == 0:
            twist.linear.x = 0.5  # Default forward motion if no immediate obstacle is detected
        # import pdb; pdb.set_trace()
        return twist


    def compute_control_safety_only(self, current_position, target_position, obstacles):
        twist = Twist()
        min_distance = float('inf')
        target_vector = np.array(target_position[:2]) - np.array(current_position[:2])
        target_distance = np.linalg.norm(target_vector)

        if target_distance == 0:
            return twist  # Avoid division by zero, no movement if on target

        for obs_name, obs_pos in obstacles:
            obs_vector = np.array(obs_pos[:2]) - np.array(current_position[:2])
            obs_distance = np.linalg.norm(obs_vector)
            if obs_distance < min_distance:
                min_distance = obs_distance

                # Safety Constraint: Slow down if an obstacle is close
                if obs_distance < self.safe_distance:
                    twist.linear.x = 0.1  # Minimal forward speed to maintain motion

        if twist.linear.x == 0:
            twist.linear.x = 0.5  # Default forward motion if no immediate obstacle is detected
        return twist






