import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

class DepthEstimator:
    def __init__(self, model_size='small', device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        self.device = device
        # Depth scaling parameters to convert from model prediction to real-world cm
        self.depth_scale_factor = 3.1002
        self.depth_offset = -0.4657
        
        print(f"Using device: {self.device} for depth estimation")
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        model_name = model_map.get(model_size.lower(), model_map['small'])
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=0 if self.device == 'cuda' else -1)
            print(f"Loaded Depth Anything v2 {model_size} model on {self.device}")
        except Exception as e:
            print(f"Error loading model on {self.device}: {e}")
            print("Falling back to CPU for depth estimation")
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=-1)
            print("Loaded Depth Anything v2 model on CPU (fallback)")
    
    def estimate_depth(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image_rgb)
        
        try:
            depth_result = self.pipe(pil_image)
            depth_map = depth_result["depth"]
            
            if isinstance(depth_map, Image.Image):
                depth_map = np.array(depth_map)
            elif isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.cpu().numpy()
        except RuntimeError as e:
            if self.device == 'mps':
                print(f"MPS error during depth estimation: {e}")
                print("Temporarily falling back to CPU for this frame")
                cpu_pipe = pipeline(task="depth-estimation", model=self.pipe.model.config._name_or_path, device='cpu')
                depth_result = cpu_pipe(pil_image)
                depth_map = depth_result["depth"]
                
                if isinstance(depth_map, Image.Image):
                    depth_map = np.array(depth_map)
                elif isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
            else:
                raise
        
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        
        return depth_map
    
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
        return colored_depth
    
    def get_grayscale_inverted_depth(self, depth_map):
        """
        Invert the depth map values and convert to grayscale image.
        This makes far objects darker and near objects brighter.
        
        Args:
            depth_map (numpy.ndarray): Normalized depth map (0-1)
            
        Returns:
            numpy.ndarray: Grayscale inverted depth map (uint8)
        """
        # Invert the depth map (far becomes near, near becomes far)
        inverted_depth = 1.0 - depth_map
        
        # Convert to grayscale (0-255)
        grayscale_depth = (inverted_depth * 255).astype(np.uint8)
        
        return grayscale_depth
    
    def get_depth_at_point(self, depth_map, x, y):
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0
    
    def convert_to_real_depth_cm(self, depth_val):
        """
        Convert the model's depth prediction to real-world depth in centimeters
        using the calibrated scale factor and offset.
        
        Args:
            depth_val (float): Raw depth value from the depth map
            
        Returns:
            float: Depth in centimeters
        """
        return self.depth_scale_factor * depth_val * 100 + self.depth_offset
    
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)
        
        region = depth_map[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region)) 