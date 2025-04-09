# #!/usr/bin/env python3
# import os
# import time
# import cv2
# import numpy as np
# import torch
# from detection_model import ObjectDetector
# from depth_model import DepthEstimator

# def main():
#     source = "./AriaEverydayActivities_1.0.0_loc5_script4_seq6_rec1_preview_rgb.mp4"
#     output_path = "./output_simple.mp4"
    
#     device = 'cuda'
#     depth_model_size = "small"
    
#     conf_threshold = 0.25
#     iou_threshold = 0.45
#     classes = None
#     enable_tracking = True
    
#     print(f"Using device: {device}")
    
#     try:
#         detector = ObjectDetector(
#             model_path='yolo11n.onnx',
#             conf_thres=conf_threshold,
#             iou_thres=iou_threshold,
#             classes=classes,
#             device=device
#         )
#     except Exception as e:
#         print(f"❌ Error initializing object detector: {e}")
#         detector = ObjectDetector(
#             model_path='yolo11n.onnx',
#             conf_thres=conf_threshold,
#             iou_thres=iou_threshold,
#             classes=classes,
#             device='cpu'
#         )
    
#     try:
#         depth_estimator = DepthEstimator(
#             model_size=depth_model_size,
#             device=device
#         )
#     except Exception as e:
#         print(f"❌ Error initializing depth estimator: {e}")
#         depth_estimator = DepthEstimator(
#             model_size=depth_model_size,
#             device='cpu'
#         )
    
#     cap = cv2.VideoCapture(source)
#     if not cap.isOpened():
#         print(f"Error: Could not open video source {source}")
#         return
    
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
#     frame_count = 0
#     start_time = time.time()
#     fps_display = "FPS: --"
    
#     print("🔄 Starting processing...")
    
#     while True:
#         key = cv2.waitKey(1)
#         if key == ord('q') or key == 27:
#             break
        
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         original_frame = frame.copy()
#         result_frame = frame.copy()
        
#         # Object Detection
#         try:
#             result_frame, detections = detector.detect(result_frame, track=enable_tracking)
#         except Exception as e:
#             print(f"⚠️ Detection Error: {e}")
#             detections = []
#             cv2.putText(result_frame, "Detection Error", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Depth Estimation
#         try:
#             depth_map = depth_estimator.estimate_depth(original_frame)
#             depth_colored = depth_estimator.colorize_depth(depth_map)
#         except Exception as e:
#             print(f"⚠️ Depth Error: {e}")
#             depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
#             cv2.putText(depth_colored, "Depth Error", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Add FPS info
#         frame_count += 1
#         if frame_count % 10 == 0:
#             elapsed = time.time() - start_time
#             fps_value = frame_count / elapsed
#             fps_display = f"FPS: {fps_value:.1f}"
        
#         cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Add depth map to corner
#         try:
#             d_h = height // 4
#             d_w = int(d_h * width / height)
#             depth_small = cv2.resize(depth_colored, (d_w, d_h))
#             result_frame[0:d_h, 0:d_w] = depth_small
#         except Exception as e:
#             print(f"⚠️ Failed to overlay depth map: {e}")
        
#         # Output
#         out.write(result_frame)
#         cv2.imshow("Object Detection + Depth", result_frame)
        
#         key = cv2.waitKey(1)
#         if key == ord('q') or key == 27:
#             break
    
#     print("✅ Done. Cleaning up...")
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"📁 Output saved to {output_path}")

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("👋 Program interrupted by user.")
#         cv2.destroyAllWindows()

#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
import torch
from detection_model import ObjectDetector
from depth_model import DepthEstimator

def main():
    source = "./AriaEverydayActivities_1.0.0_loc5_script4_seq6_rec1_preview_rgb.mp4"
    output_path = "./output_simple.mp4"
    
    device = 'cuda'
    depth_model_size = "small"
    
    conf_threshold = 0.7
    iou_threshold = 0.45
    classes = None
    enable_tracking = True
    
    print(f"Using device: {device}")
    
    try:
        detector = ObjectDetector(
            model_path='yolo11n.onnx',
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device
        )
    except Exception as e:
        print(f"❌ Error initializing object detector: {e}")
        detector = ObjectDetector(
            model_path='yolo11n.onnx',
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device='cpu'
        )
    
    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device
        )
    except Exception as e:
        print(f"❌ Error initializing depth estimator: {e}")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu'
        )
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    
    print("🔄 Starting processing...")
    
    while True:
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        result_frame = frame.copy()
        
        # Object Detection
        try:
            result_frame, detections = detector.detect(result_frame, track=enable_tracking)
        except Exception as e:
            print(f"⚠️ Detection Error: {e}")
            detections = []
            cv2.putText(result_frame, "Detection Error", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Depth Estimation
        try:
            depth_map = depth_estimator.estimate_depth(original_frame)
            depth_colored = depth_estimator.colorize_depth(depth_map)
        except Exception as e:
            print(f"⚠️ Depth Error: {e}")
            depth_map = np.zeros((height, width), dtype=np.float32)
            depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(depth_colored, "Depth Error", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw depth value for each detected object
        for det in detections:
            try:
                bbox, score, class_id, obj_id = det
                x1, y1, x2, y2 = map(int, bbox)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Clip to image size
                cx = np.clip(cx, 0, width - 1)
                cy = np.clip(cy, 0, height - 1)

                depth_val = depth_map[cy, cx]
                depth_text = f"{depth_val:.2f}m"

                label = f"{detector.get_class_names()[class_id]} {depth_text}"
                cv2.putText(result_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception as e:
                print(f"⚠️ Error adding depth to detection: {e}")
        
        # Add FPS info
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps_value = frame_count / elapsed
            fps_display = f"FPS: {fps_value:.1f}"
        
        cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add depth map to corner
        try:
            d_h = height // 4
            d_w = int(d_h * width / height)
            depth_small = cv2.resize(depth_colored, (d_w, d_h))
            result_frame[0:d_h, 0:d_w] = depth_small
        except Exception as e:
            print(f"⚠️ Failed to overlay depth map: {e}")
        
        # Output
        out.write(result_frame)
        cv2.imshow("Object Detection + Depth", result_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
    
    print("✅ Done. Cleaning up...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"📁 Output saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("👋 Program interrupted by user.")
        cv2.destroyAllWindows()
