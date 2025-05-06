import numpy as np
np.float = np.float64
import torch
from torch import nn
import clip
import torch.nn.functional as nnf
import sys
from tqdm import tqdm, trange
import skimage.io as io
import os
import skimage
import IPython.display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospy
import ros_numpy
import threading
import cv2
import imutils
import sensor_msgs.msg as sms
import std_msgs.msg
from rospy.numpy_msg import numpy_msg
from collections import OrderedDict
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, Transform, Point, Quaternion
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from torchvision.transforms.functional import to_tensor
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from gazebo_msgs.srv import GetModelState
from tqdm.notebook import tqdm
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from beam import generate_beam, generate2
from yolov5 import YOLOv5 
from torchvision import transforms
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
import tf
import math
import spacy
from transformers import pipeline
from cbf_controller import CBF
from test_nlp import CommandParser 
import argparse

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')
nlp = spacy.load("en_core_web_sm")

def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')

def visualize_image(image, window_name="Input Image"):
    # Check if the image is single-channel (like a depth map)
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Normalize the image to 0-255 for displaying
        image_normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image_normalized = np.uint8(image_normalized)  # Convert to uint8
        cv2.imshow(window_name, image_normalized)
    else:
        # For BGR images, display as is
        cv2.imshow(window_name, image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_cropped_depth_map_opencv(depth_image, bbox):
    # Calculate the bounds ensuring they are within the image dimensions
    x1, y1, x2, y2 = map(int, bbox)
    cropped_depth_map = depth_image[y1:y2, x1:x2]

    # Normalize the depth map for better visualization
    cropped_depth_map_normalized = cv2.normalize(cropped_depth_map, None, 0, 255, cv2.NORM_MINMAX)
    cropped_depth_map_normalized = cropped_depth_map_normalized.astype('uint8')

    # Display the cropped depth map
    cv2.imshow('Cropped Depth Map', cropped_depth_map_normalized)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

############## The Three Modalities Class Definition ##########

class Detection:
    def __init__(self, model_name='custom', device=None):
        """
        Initialize the YOLO detector with a specified model.
        :param model_name: Model name, e.g., 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', custom
        :param device: The device to run the model on, 'cpu' or 'cuda'.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load the model using torch.hub
        self.model = torch.hub.load('ultralytics/yolov5', model_name, 
        path='/home/sourav/safe_ai/yolov5/runs/train/outdoor_more/weights/best.pt', force_reload=True)
        self.model.to(self.device)  # Ensure the model is on the correct device
        print("Initialized Detection Sub-process")


    def detect(self, img):
        """
        Perform object detection on an image.
        :param img: The image tensor or path to an image file.
        :return: Detections with bounding box coordinates, class names, and confidence scores.
        """      
        # Inference
        results = self.model(img)
        results_df = results.pandas().xyxy[0]
        if visualize:
            self.visualize_results(img, results_df)    #for debugging and visualization
        
        # Parse results
        # results = results.pandas().xyxy[0]  # Extract results to Pandas DataFrame
        return results_df


    def visualize_results(self, img, results_df):
        for index, row in results_df.iterrows():
            xmin, ymin, xmax, ymax, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence']
            cls = int(row['class'])
            label = self.model.names[cls]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(img, f'{label} {conf:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Detection', img)
        cv2.waitKey(1)  # Waits for a millisecond for a key press

class SceneCaptioning:
    """
    Class to handle scene captioning using CLIP and GPT-2.
    """
    def __init__(self,load_model=True, encoder="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        import pdb; pdb.set_trace()
        self.clip_model, self.preprocess = clip.load(encoder, device=self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = 10
        current_directory = os.getcwd()
        save_path = os.path.join(current_directory, "pretrained_models")
        os.makedirs(save_path, exist_ok=True)

        if load_model:
            model_path = os.path.join(save_path, 'outdoor1_prefix-009.pt')
            if os.path.exists(model_path):
                self.model = ClipCaptionModel(self.prefix_length)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
                self.model.to(self.device).eval()
            else:
                raise FileNotFoundError(f"No model file found at {model_path}")

    def scene_caption(self, image):
        image_input = to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image_input).float()
            prefix_embed = self.model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        
        if self.use_beam_search:
            generated_text_prefix = generate_beam(self.model, self.tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(self.model, self.tokenizer, embed=prefix_embed)

        print('\n', generated_text_prefix)
        return generated_text_prefix

    def verify_and_detect_landmarks(self, image, landmarks, detection_model, reverify=False):
        """
        Verify landmarks using CLIP and detect them using an object detection model.
        """
        image_input = self.preprocess(PILImage.fromarray(image)).unsqueeze(0).to(self.device)
        verified_landmarks = {}
        # Example of flattening a list of lists
        flat_landmarks = [item for sublist in landmarks for item in (sublist if isinstance(sublist, list) else [sublist])]
        landmarks = flat_landmarks

        for landmark in landmarks:
            text = clip.tokenize([f"a photo of {landmark}"]).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text)
            
            similarity = torch.cosine_similarity(image_features, text_features).cpu().numpy()
            if similarity > 0.2:  # Adjust threshold as needed
                if reverify:
                    return True
                else:
                    landmark_key = str(landmark) 
                    verified_landmarks[landmark_key] = None  # Initialize as not detected
        detections = detection_model.detect(image)
        # import pdb; pdb.set_trace
        for _, detection in detections.iterrows():
            label, xmin, ymin, xmax, ymax = detection['name'], detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
            box = (xmin, ymin, xmax, ymax)  # Create a tuple for the bounding box

            # Update verified landmarks with detected bounding boxes
            if label in verified_landmarks:
                verified_landmarks[label] = box

        return verified_landmarks

def calculate_collision_probability(drone_path, dynamic_obstacles, time_step=1, threshold=0.5, cone_angle=np.radians(30), is_cbf=False):
    """Calculate the probability of collision based on the distance threshold and the cone angle for vision."""
    collision_count = 0
    total_time = len(drone_path) * time_step

    # Iterate over each point in the drone path
    for i, drone_position in enumerate(drone_path):
        for obs_key, obs_path in dynamic_obstacles.items():
            if i < len(obs_path):  # Ensure there's a corresponding obstacle position
                obstacle_position = obs_path[i]
                obs_vector = np.array(obstacle_position) - np.array(drone_position)
                obs_distance = np.linalg.norm(obs_vector)

                if obs_distance < threshold:
                    drone_velocity = drone_path[i + 1] - np.array(drone_position) if i + 1 < len(drone_path) else np.zeros(3)
                    drone_speed = np.linalg.norm(drone_velocity)
                    drone_direction = drone_velocity / drone_speed if drone_speed != 0 else np.zeros_like(drone_velocity)

                    obs_direction = obs_vector / obs_distance if obs_distance != 0 else np.zeros_like(obs_vector)
                    dot_product = np.dot(drone_direction, obs_direction)
                    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

                    if angle < cone_angle:
                        print(f"Collision detected at time {i * time_step}: Drone at {drone_position}, Obstacle at {obstacle_position}")
                        collision_count += 1

    collision_probability = collision_count / total_time if total_time > 0 else 0
    return collision_probability


class ASMA:
    def __init__(self, args):
        rospy.init_node('ASMA_node')
        rospy.on_shutdown(self.on_shutdown)
        self.bridge = CvBridge()
        self.args = args
        self.parser = CommandParser()
        self.image_topic = '/bebop2/camera_base/image_raw'
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.scene_callback)
        self.cmd_pub = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=10) 
        self.command_publisher = rospy.Publisher('/bebop2/command/trajectory', MultiDOFJointTrajectory, queue_size=10)
        self.depth_sub = rospy.Subscriber("/bebop2/depth_camera/depth/image_raw", numpy_msg(sms.Image), self.depthLogger)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.close_proximity_threshold = 2.0 
        self.global_image = None
        self.depth_array = None
        self.depth_map = None
        self.image_lock = threading.Lock()
        self.image_available = threading.Event()
        self.obstacle_names = ['bird', 'actor']  

        self.drone_path = []
        self.total_distance = 0.0
        self.obstacle_path = {}

        # Initialize processing classes
        encoder = getattr(self.args, 'encoder', 'ViT-B/32')
        if encoder:
            print(f"SceneUnderstanding initialized with encoder: {encoder}")
            self.scene_captioner = SceneCaptioning(load_model=False, encoder=encoder)
        else:
            self.scene_captioner = None
        self.detector = Detection()

        # Threads for asynchronous processing
        self.threads = []   
        self.preplanned = False

        if self.preplanned:
            # self.waypoints = [(2, -1, 1), (-4, -1, 1), (-6, -2, 1)]
            self.waypoints = [(-1,1,1), (-2, 2, 1)]
        else:
            self.waypoints = []
        self.current_waypoint_index = 0
        self.arrival_threshold = 2.0  # Distance at which a waypoint is considered reached
        self.safe_distance = 5.0 # Safe Distance for obstacle avoidance

        self.image_width = 856
        self.image_height = 480
        self.horizontal_fov = 1.047198  # Horizontal field of view in radians
        self.focal_length = (self.image_width / 2) / np.tan(self.horizontal_fov / 2)

        # Initialize the CBF controller
        safe_distance = self.args['safe_distance'] if 'safe_distance' in self.args else 2.0  # Default to 2.0 if not provided
        self.cbf_controller = self.init_cbf_controller(safe_distance)


    def on_shutdown(self):
        print("Shutdown procedure triggered.")
        np.save(f'drone_path_{self.args.landmark}.npy', self.drone_path)
        print("Drone path recorded and saved.")
        # Saving the positions of dynamic obstacles
        # import pdb; pdb.set_trace()
        np.save('obstacle_data.npy', self.obstacle_path)
        print("Obstacle data recorded and saved.")

        # Ensure that there is more than one point in the drone path to calculate distances
        if self.drone_path and len(self.drone_path) > 1:
            # Trajectory Length (TL)
            distances = np.linalg.norm(np.diff(self.drone_path, axis=0), axis=1)
            total_path_length = np.sum(distances)
            print(f"Total path length (TL): {total_path_length:.2f} meters")

            # Navigation Error (NE)
            final_position = np.array(self.drone_path[-1][:len(self.target_position)]) 
            navigation_error = np.linalg.norm(final_position - self.target_position)
            print(f"Navigation Error (NE): {navigation_error:.2f} meters")

            max_distance_for_success = 3.0  # Define the maximum distance for perfect success
            max_penalty_distance = 10.0  # Define a distance beyond which success is 0

            # Calculate success percentage
            if navigation_error <= max_distance_for_success:
                success_rate = 1.0  # 100% success if within the perfect range
            else:
                # Linearly decrease the success rate beyond max_distance_for_success
                excess_distance = navigation_error - max_distance_for_success
                success_rate = max(0.0, 1.0 - (excess_distance / (max_penalty_distance - max_distance_for_success)))

            success_percentage = success_rate * 100
            print(f"Success Rate (SR): {success_percentage:.2f}%")

        else:
            total_path_length = 0
            navigation_error = float('inf')  # Impractical value if no path
            success_rate = 0.0
            print("Insufficient data to calculate path metrics.")

        # # Estimated Collision Probability
        # self.estimated_collision_probability = calculate_collision_probability(self.drone_path, self.obstacle_path, time_step=1, threshold=0.5)        
        # print(f"Estimated collision probability: {self.estimated_collision_probability:.2%}")

        print("Simulation completed successfully.")


    def update_positions(self):
        drone_position = self.get_current_drone_position()
        if drone_position:
            self.drone_path.append(drone_position)
        obstacle_positions = self.get_current_obstacle_position()
        for obs_key, position in obstacle_positions.items():
            if obs_key not in self.obstacle_path:
                self.obstacle_path[obs_key] = []  # Initialize if not already
            self.obstacle_path[obs_key].append(position)  # Append the position list directly


    def get_current_drone_position(self):
        try:
            response = self.get_model_state('bebop2', '')  
            return [response.pose.position.x, response.pose.position.y, response.pose.position.z]
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)
            return None

    def get_current_obstacle_position(self):
        positions = {}
        for obstacle_name in self.obstacle_names:
            try:
                response = self.get_model_state(obstacle_name, '')
                position = [response.pose.position.x, response.pose.position.y, response.pose.position.z]
                positions[obstacle_name] = position
            except rospy.ServiceException as e:
                rospy.logwarn(f"Failed to get position for {obstacle_name}: {e}")
        return positions

    def init_cbf_controller(self, safe_distance):
        return CBF(safe_distance)

    def scene_callback(self, data):
        with self.image_lock:
            self.global_image = ros_numpy.numpify(data)
            self.image_available.set() 

    def depthLogger(self, data):
        self.depth_array = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)

    def computeDepth(self):
        depth_array = self.depth_array
        nan_count = np.isnan(depth_array).sum()
        if nan_count > 0:
            depth_array = np.nan_to_num(depth_array, nan=10.0)  
        self.depth_map = depth_array
 
    def understand_scene_and_navigate(self):
        action = 'go'
        attributes = {'tree': 'right', 'house': 'left', 'mailbox': None}

        while not rospy.is_shutdown():
            self.image_available.wait()
            with self.image_lock:
                self.image = self.global_image

            if hasattr(self.args, 'user_command') and self.args.user_command:
                # Parse command if specified
                user_command = self.args.user_command
                action, landmark, attribute = self.parser.parse_command(user_command)
            else:
                landmark = self.args.landmark  
                attribute = attributes.get(landmark, '') 

            full_image = self.image
            image_height, image_width, _ = full_image.shape

            if attribute == 'right':
                image = full_image[:, image_width//2:, :]  
            elif attribute == 'left':
                image = full_image[:, :image_width//2, :]  
            else:
                image = full_image  #
            image_resized = cv2.resize(image, (224, 224))  

            # Only verify and detect landmarks if scene_captioner is not None
            if self.scene_captioner is not None:
                verified_landmarks = self.scene_captioner.verify_and_detect_landmarks(image_resized, [landmark], self.detector)
                landmark_verified = verified_landmarks.get(landmark)
            else:
                # Assume landmark is always verified if scene_captioner is None
                landmark_verified = True
                bbox = None  # Example placeholder, ensure navigation logic does not depend on bbox if None

            if landmark_verified:
                # Proceed to navigate even if bbox is None when scene_captioner is None
                if self.navigate_to_landmark(action, landmark, attribute):
                    rospy.signal_shutdown("Done")

            else:
                print(f"{landmark} not verified in the scene.")
                self.stop_drone()

    def stop_drone(self):
        """
        Stop the drone by setting all motion commands to zero.
        """
        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.cmd_pub.publish(twist)

    def depth_at_center(self, bbox):
        depth_image = self.depth_map
        # Calculate the center coordinates of the bounding box
        x_center = int((bbox[0] + bbox[2]) / 2)
        y_center = int((bbox[1] + bbox[3]) / 2)

        # Ensure the center coordinates are within the image bounds
        x_center = max(0, min(x_center, depth_image.shape[1] - 1))
        y_center = max(0, min(y_center, depth_image.shape[0] - 1))

        # Return the depth at the center point
        return depth_image[y_center, x_center]

    def navigate_to_landmark(self, action, landmark, attribute):
        safe_stopping_margin = 1.0
        object_dimensions = {
            'tree': 2.0,
            'house': 3.0,
            'mailbox': 3.0
        }

        model_names = self.get_model_positions(landmark)
        if not model_names:
            rospy.loginfo(f"No models found for the landmark '{landmark}'.")
            return
        
        selected_model = None
        selected_pos = None
        for model_name in model_names:
            model_state = self.get_model_state(model_name, '')
            model_pos = np.array([model_state.pose.position.x, model_state.pose.position.y])
            if selected_model is None or (attribute == "right" and model_pos[1] > selected_pos[1]) or (attribute == "left" and model_pos[1] < selected_pos[1]):
                selected_model = model_name
                selected_pos = model_pos

        if selected_model is None:
            rospy.loginfo(f"No suitable model found for attribute '{attribute}'.")
            return False

        dimension = object_dimensions.get(landmark)  # Default dimensions if not listed
        drone_state = self.get_model_state('bebop2', '')
        drone_pos = np.array([drone_state.pose.position.x, drone_state.pose.position.y])
        distance = np.linalg.norm(drone_pos - selected_pos)
        self.target_position = selected_pos

        # Calculate the drone's current yaw from its orientation quaternion
        quaternion = [
            drone_state.pose.orientation.x,
            drone_state.pose.orientation.y,
            drone_state.pose.orientation.z,
            drone_state.pose.orientation.w
        ]
        euler = tf.transformations.euler_from_quaternion(quaternion)
        current_yaw = euler[2]  # yaw is the third element
        angle_to_target = np.arctan2(selected_pos[1] - drone_pos[1], selected_pos[0] - drone_pos[0])
        angular_error = self.angle_difference(current_yaw, angle_to_target)

        if distance < self.close_proximity_threshold + safe_stopping_margin:
            rospy.loginfo("Close to the landmark, stopping navigation.")
            self.cmd_pub.publish(Twist())  # Send zero velocity to stop the drone
            return True

        # Prepare the normal control message
        control_msg = Twist()
        control_msg.linear.x = min(0.5, distance)  # Move at a calculated speed or slow down as approaching
        control_msg.angular.z = angular_error  # Adjust angular velocity to turn the drone towards the target
        
        
        # Check proximity to moving obstacles
        obstacles = ['bird', 'actor']  # Example names of moving obstacles
        for obs_name in obstacles:
            obs_state = self.get_model_state(obs_name, '')
            obs_pos = np.array([obs_state.pose.position.x, obs_state.pose.position.y])
            obs_distance = np.linalg.norm(drone_pos - obs_pos)
            if obs_distance < self.safe_distance:
                rospy.loginfo(f"Close to moving obstacle {obs_name}, applying CBF adjustments.")
                cbf_control = self.cbf_controller.compute_control(drone_pos, selected_pos, [(obs_name, obs_pos)])
                rospy.loginfo(f"CBF Control - Linear: {cbf_control.linear.x}, Angular: {cbf_control.angular.z}")
                control_msg.linear.x = max(control_msg.linear.x, cbf_control.linear.x)  # Ensure some minimum forward velocity
                control_msg.angular.z += cbf_control.angular.z
                break
        self.cmd_pub.publish(control_msg)
        rospy.loginfo("Published control command to move towards landmark.")
        self.update_positions()

    def angle_difference(self, current_angle, target_angle):
        """
        Calculate the shortest path between two angles
        """
        diff = target_angle - current_angle
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def get_current_drone_altitude(self):
        states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        while states is None:
            states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        current_state = states('bebop2', '')
        current_z = current_state.pose.position.z
        return current_z

    def publish_trajectory(self, target):
        x_target, y_target = target[:2]  # Only take x and y coordinates
        z_target = target[2] if len(target) > 2 else self.get_current_drone_altitude()  # Default to current drone altitude if z not provided

        # Calculate direction to target for orientation
        current_position = self.get_current_drone_position()
        if current_position:
            direction = np.array([x_target, y_target, z_target]) - np.array(current_position)
            yaw = np.arctan2(direction[1], direction[0])
        else:
            yaw = 0  # Default orientation facing north if current position is unknown

        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

        # Check if there are subscribers to the trajectory topic before publishing
        if self.command_publisher.get_num_connections() > 0:
            transforms = Transform(translation=Point(x_target, y_target, z_target), rotation=Quaternion(*quaternion))
            traj = MultiDOFJointTrajectory()
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'  # Assuming 'world' is the fixed frame in use
            traj.joint_names.append('base_link')  # Assuming the drone is controlled as 'base_link'
            traj.header = header
            velocities = Twist()
            accelerations = Twist()
            point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accelerations], rospy.Time(1))
            traj.points.append(point)
            self.command_publisher.publish(traj)
            rospy.loginfo("Published trajectory to /bebop2/command/trajectory")
            
            # Start monitoring the drone's position after trajectory command
            self.monitor_drone_movement(duration=5)  # Monitor for 5 seconds or until it reaches the target
            return True
        else:
            rospy.logwarn("No subscribers on /bebop2/command/trajectory, not publishing trajectory.")
            return False

    def monitor_drone_movement(self, duration):
        """ Monitor the drone's position for a specified duration after sending a trajectory command. """
        start_time = rospy.Time.now()
        while rospy.Time.now() - start_time < rospy.Duration(duration):
            self.update_positions()
            rospy.sleep(0.1)  # Update position every 100 ms

    def apply_cbf_control(self, current_position, target, obstacles):
        return self.cbf_controller.compute_control(current_position, target, obstacles)

    def get_model_positions(self, landmark):
        landmark_to_model_mapping = {
            'tree': ['pine_tree', 'pine_tree_0'],
            'house': ['House 1', 'House 2'],
            'mailbox': ['Mailbox'],
            'door': ['House 1', 'House 2'],  # Assuming doors are part of house models
            'porch': ['House 1', 'House 2'],  # Assuming porches are part of house models
            'window': ['House 1', 'House 2']  # Assuming windows are part of house models
        }

        model_names = landmark_to_model_mapping.get(landmark, [])
        positions = {}
        for model_name in model_names:
            try:
                get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                model_state = get_model_state(model_name, "")
                positions[model_name] = (model_state.pose.position.x, model_state.pose.position.y, model_state.pose.position.z)
            except rospy.ServiceException as e:
                rospy.logwarn("Service call failed: %s" % e)
        return positions


    def is_near_specific_obstacles(self, target, landmark, threshold=5.0):
        obstacle_positions = self.get_model_positions(landmark)
        target_x, target_y = target[:2]
        close_obstacles = []  # To store obstacles that are close to the target
        all_obstacles = []  # To store all obstacles regardless of proximity

        for model_name, position in obstacle_positions.items():
            obstacle_x, obstacle_y = position[:2]
            distance = ((target_x - obstacle_x) ** 2 + (target_y - obstacle_y) ** 2) ** 0.5
            if distance <= threshold:
                close_obstacles.append((model_name, (obstacle_x, obstacle_y), distance))
            all_obstacles.append((model_name, (obstacle_x, obstacle_y), distance))
        
        if close_obstacles:
            return True, close_obstacles, (target_x, target_y)
        return False, all_obstacles, (target_x, target_y)

    def start_processing(self):
        # Thread for captioning and navigation
        thread = threading.Thread(target=self.understand_scene_and_navigate)
        thread.start()
        self.threads.append(thread)

    def wait_for_completion(self):
        for thread in self.threads:
            thread.join() 


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Start ASMA system with configurable CBF type.')
    parser.add_argument('--safe_distance', type=float, default=2.0,
                        help='Safe distance for CBF calculations.')
    parser.add_argument('user_command', nargs='?', default="", help='Navigation command to execute.')


    args = parser.parse_args()

    asma = ASMA(args)
    visualize = False  ## Please keep this False during data collection
    try:
        asma.start_processing()
        rospy.spin()
        asma.wait_for_completion()        
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Shutting down")
    cv2.destroyAllWindows()





