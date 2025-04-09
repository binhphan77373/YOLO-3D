import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque

class ObjectDetector:
    def __init__(self, model_path='yolo11n.onnx', conf_thres=0.25, iou_thres=0.45, classes=None, device=None):

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        print(f"Using device: {self.device} for object detection")

        self.model = YOLO(model_path)
        print(f"Loaded YOLOv11 model from {model_path}")
        
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        
        if classes is not None:
            self.model.overrides['classes'] = classes
        
        self.tracking_trajectories = {}
    
    def detect(self, image, track=True):
        detections = []
        annotated_image = image.copy()
        
        try:
            if track:
                results = self.model.track(image, verbose=False, device=self.device, persist=True)
            else:
                results = self.model.predict(image, verbose=False, device=self.device)
        except RuntimeError as e:
            if self.device == 'mps' and "not currently implemented for the MPS device" in str(e):
                print(f"MPS error during detection: {e}")
                print("Falling back to CPU for this frame")
                if track:
                    results = self.model.track(image, verbose=False, device='cpu', persist=True)
                else:
                    results = self.model.predict(image, verbose=False, device='cpu')
            else:
                raise
        
        if track:
            for id_ in list(self.tracking_trajectories.keys()):
                if id_ not in [int(bbox.id) for predictions in results if predictions is not None 
                              for bbox in predictions.boxes if bbox.id is not None]:
                    del self.tracking_trajectories[id_]
            
            for predictions in results:
                if predictions is None:
                    continue
                
                if predictions.boxes is None:
                    continue
                
                for bbox in predictions.boxes:
                    scores = bbox.conf
                    classes = bbox.cls
                    bbox_coords = bbox.xyxy
                    
                    if hasattr(bbox, 'id') and bbox.id is not None:
                        ids = bbox.id
                    else:
                        ids = [None] * len(scores)
                    
                    for score, class_id, bbox_coord, id_ in zip(scores, classes, bbox_coords, ids):
                        xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()
                        
                        detections.append([
                            [xmin, ymin, xmax, ymax],  # bbox
                            float(score),              # confidence score
                            int(class_id),             # class id
                            int(id_) if id_ is not None else None  # object id
                        ])
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, 
                                     (int(xmin), int(ymin)), 
                                     (int(xmax), int(ymax)), 
                                     (0, 0, 225), 2)
                        
                        # Add label
                        label = f"ID: {int(id_) if id_ is not None else 'N/A'} {predictions.names[int(class_id)]} {float(score):.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(annotated_image, 
                                     (int(xmin), int(ymin)), 
                                     (int(xmin) + dim[0], int(ymin) - dim[1] - baseline), 
                                     (30, 30, 30), cv2.FILLED)
                        cv2.putText(annotated_image, label, 
                                   (int(xmin), int(ymin) - 7), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Update tracking trajectories
                        if id_ is not None:
                            centroid_x = (xmin + xmax) / 2
                            centroid_y = (ymin + ymax) / 2
                            
                            if int(id_) not in self.tracking_trajectories:
                                self.tracking_trajectories[int(id_)] = deque(maxlen=10)
                            
                            self.tracking_trajectories[int(id_)].append((centroid_x, centroid_y))
            
            # Draw trajectories
            for id_, trajectory in self.tracking_trajectories.items():
                for i in range(1, len(trajectory)):
                    thickness = int(2 * (i / len(trajectory)) + 1)
                    cv2.line(annotated_image, 
                            (int(trajectory[i-1][0]), int(trajectory[i-1][1])), 
                            (int(trajectory[i][0]), int(trajectory[i][1])), 
                            (255, 255, 255), thickness)
        
        else:
            # Process results for non-tracking mode
            for predictions in results:
                if predictions is None:
                    continue
                
                if predictions.boxes is None:
                    continue
                
                # Process boxes
                for bbox in predictions.boxes:
                    # Extract information
                    scores = bbox.conf
                    classes = bbox.cls
                    bbox_coords = bbox.xyxy
                    
                    # Process each detection
                    for score, class_id, bbox_coord in zip(scores, classes, bbox_coords):
                        xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()
                        
                        # Add to detections list
                        detections.append([
                            [xmin, ymin, xmax, ymax],  # bbox
                            float(score),              # confidence score
                            int(class_id),             # class id
                            None                       # object id (None for no tracking)
                        ])
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, 
                                     (int(xmin), int(ymin)), 
                                     (int(xmax), int(ymax)), 
                                     (0, 0, 225), 2)
                        
                        # Add label
                        label = f"{predictions.names[int(class_id)]} {float(score):.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(annotated_image, 
                                     (int(xmin), int(ymin)), 
                                     (int(xmin) + dim[0], int(ymin) - dim[1] - baseline), 
                                     (30, 30, 30), cv2.FILLED)
                        cv2.putText(annotated_image, label, 
                                   (int(xmin), int(ymin) - 7), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_image, detections
    
    def get_class_names(self):
        return self.model.names 