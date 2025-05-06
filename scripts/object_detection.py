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
import matplotlib.pyplot as plt
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
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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


class run_YOLO:
    def __init__(self):
        rospy.init_node('yolo_node')
        self.bridge = CvBridge()
        self.image_topic = '/bebop2/camera_base/image_raw'
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.scene_callback)
        self.global_image = None
        self.image_lock = threading.Lock()
        self.image_available = threading.Event()
        self.detector = Detection()
        self.threads = []   
        self.image_width = 856
        self.image_height = 480

    def scene_callback(self, data):
        with self.image_lock:
            self.global_image = ros_numpy.numpify(data)
            self.image_available.set() 

    def detect_object(self):
        while not rospy.is_shutdown():
            self.image_available.wait()  # Wait for the latest image
            with self.image_lock:
                self.image = self.global_image
                self.detector.detect(self.image)

    def start_processing(self):
        # Thread for captioning and navigation
        thread = threading.Thread(target=self.detect_object)
        thread.start()
        self.threads.append(thread)

    def wait_for_completion(self):
        for thread in self.threads:
            thread.join() 


if __name__=="__main__":
    yolo = run_YOLO()
    visualize = True
    try:
        yolo.start_processing()
        rospy.spin()
        yolo.wait_for_completion()        
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Shutting down")
    cv2.destroyAllWindows()


