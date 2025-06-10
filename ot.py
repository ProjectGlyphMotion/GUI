import cv2
from ultralytics import YOLO
import argparse
from tqdm import tqdm 
import torch
import os
import sys
import subprocess
import threading
import time
import queue

# Attempt to import psutil for CPU & Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ [WARNING] psutil library not found. RAM usage limiting and CPU/Memory utilization display will not be available.")

# Attempt to import GPUtil for GPU monitoring
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except (ImportError, Exception):
    GPUTIL_AVAILABLE = False

# --- FFMPEG Check ---
FFMPEG_AVAILABLE = False

def check_ffmpeg():
    global FFMPEG_AVAILABLE
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("[INFO] ffmpeg found and seems to be working.")
        FFMPEG_AVAILABLE = True
    except Exception:
        print("⚠️ [WARNING] ffmpeg command not found or not working. Audio processing will be skipped.")
        FFMPEG_AVAILABLE = False
    return FFMPEG_AVAILABLE

# --- Configuration & Setup ---
DEFAULT_MODEL_PATH = "yolov8m.pt"
DEFAULT_ALLOWED_CLASSES = [
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "airplane", "bird", "cat", "dog",
    "train", "boat", "bench", "backpack", "umbrella", "handbag", "suitcase", "sports ball",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "chair", "couch", "potted plant", "bed", "dining table",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "refrigerator", "book", "clock", "vase", "scissors"
]
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
MAX_RAM_USAGE_PERCENT = 80.0

# User Configuration 
USE_GPU_IN_SCRIPT = True 

def get_system_utilization(device_to_use):
    util_stats = {}
    if PSUTIL_AVAILABLE:
        util_stats['cpu'] = psutil.cpu_percent()
        mem_info = psutil.virtual_memory()
        util_stats['mem_used_gb'] = mem_info.used / (1024**3)
        util_stats['mem_total_gb'] = mem_info.total / (1024**3)
        util_stats['mem'] = mem_info.percent
    if GPUTIL_AVAILABLE and device_to_use == "cuda":
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                util_stats['gpu_load'] = gpu.load * 100
                util_stats['gpu_mem'] = gpu.memoryUtil * 100
        except Exception: pass
    return util_stats

def format_utilization_string(stats):
    parts = []
    if 'cpu' in stats: parts.append(f"CPU:{stats['cpu']:.1f}%")
    if 'mem' in stats:
        parts.append(f"Mem:{stats['mem']:.1f}% ({stats.get('mem_used_gb',0):.1f}/{stats.get('mem_total_gb',0):.1f}GB)")
    if 'gpu_load' in stats: parts.append(f"GPU-L:{stats['gpu_load']:.1f}%")
    if 'gpu_mem' in stats: parts.append(f"GPU-M:{stats['gpu_mem']:.1f}%")
    return " | ".join(parts) if parts else "Stats N/A"


def process_audio_ffmpeg(video_source_path, temp_silent_video_abs_path, final_output_video_path):
    # This function is now used only for final audio merge, outside the real-time processing loop
    if not FFMPEG_AVAILABLE:
        print("⚠️ [WARNING] [AUDIO_THREAD] ffmpeg not available. Skipping audio merge.")
        if os.path.exists(temp_silent_video_abs_path) and not os.path.exists(final_output_video_path):
            try: os.rename(temp_silent_video_abs_path, final_output_video_path); print(f"✅ [SUCCESS] Silent video saved as '{final_output_video_path}'.")
            except OSError as e: print(f"❌ [ERROR] Could not rename temp file: {e}.")
        return False

    if not os.path.exists(temp_silent_video_abs_path):
        print(f"❌ [ERROR] [AUDIO_THREAD] Temporary silent video '{temp_silent_video_abs_path}' not found. Cannot merge audio.")
        return False

    print(f"\n[AUDIO_THREAD] Adding audio using ffmpeg (direct video stream copy mode).")
    if os.path.exists(final_output_video_path): print(f"⚠️ [WARNING] Output file {final_output_video_path} exists. Overwriting.")

    ffmpeg_command_base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
                           "-i", temp_silent_video_abs_path, "-i", video_source_path]

    video_codec_params = ["-c:v", "copy"]

    ffmpeg_command = ffmpeg_command_base + video_codec_params + ["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", final_output_video_path]

    try:
        print(f"[AUDIO_THREAD] Executing: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True)
        print(f"\n✅ [SUCCESS] [AUDIO_THREAD] ffmpeg successfully processed. Output: '{final_output_video_path}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ [ERROR] [AUDIO_THREAD] ffmpeg failed (code {e.returncode}).")
    except Exception as e_ffmpeg: print(f"❌ [ERROR] [AUDIO_THREAD] ffmpeg error: {e_ffmpeg}")

    return False


class YOLOv8Tracker:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, allowed_classes=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, box_color=(0, 255, 0)):
        print("--- FFMPEG Check ---"); check_ffmpeg(); print("--------------------")
        if PSUTIL_AVAILABLE: print("[DEBUG] Priming psutil.cpu_percent()..."); psutil.cpu_percent()
        if GPUTIL_AVAILABLE:
            print("[DEBUG] Attempting to prime GPUtil.getGPUs()...")
            try: GPUtil.getGPUs(); print("[DEBUG] GPUtil.getGPUs() primed.")
            except Exception as e: print(f"[DEBUG] Error GPUtil priming: {e}")

        self.allowed_classes = allowed_classes if allowed_classes is not None else DEFAULT_ALLOWED_CLASSES
        self.confidence_threshold = confidence_threshold
        self.box_color = tuple(box_color)

        self.device = "cpu"
        if USE_GPU_IN_SCRIPT and torch.cuda.is_available():
            self.device = "cuda"
            print("✅ [SUCCESS] CUDA GPU available. Using GPU for tracking.")
        elif USE_GPU_IN_SCRIPT:
            print("⚠️ [WARNING] CUDA GPU not found. Falling back to CPU for tracking.")
        else:
            print("ℹ️ [INFO] Using CPU for tracking.")

        print(f"[INFO] Loading model: {model_path} on {self.device}")
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"✅ [SUCCESS] Model loaded.")
        except Exception as e:
            print(f"❌ [ERROR] Failed to load model: {e}")
            self.model = None 

        # Suggest number of threads for PyTorch CPU operations.
        try:
            cpu_cores = os.cpu_count()
            if cpu_cores:
                num_threads_to_set = max(1, cpu_cores // 2)
                torch.set_num_threads(num_threads_to_set)
                print(f"[INFO] Suggested {num_threads_to_set} threads for PyTorch CPU operations.")
            else:
                print("[INFO] Could not determine CPU core count for PyTorch thread setting.")
        except Exception as e:
            print(f"⚠️ [WARNING] Could not set PyTorch CPU threads: {e}")

        print(f"[INFO] Tracking classes: {self.allowed_classes}")
        self.last_track_id = 0 

    def process_frame(self, frame):
        """
        Processes a single frame for object tracking.
        Args:
            frame (numpy.ndarray): The input frame (BGR format from OpenCV).
        Returns:
            tuple: (original_frame, annotated_frame) or (None, None) if model not loaded.
        """
        if self.model is None:
            return frame, frame 

        # Make a copy of the frame for annotation to avoid modifying the original
        annotated_frame = frame.copy()

        # Run tracking on the frame
        # persist=True allows the tracker to maintain object identities across frames
        results = self.model.track(frame, persist=True, verbose=False, conf=self.confidence_threshold)

        if results and results[0].boxes:
            for box in results[0].boxes:
                # Ensure box.id exists for tracking results
                if box.id is None:
                    continue

                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                track_id = int(box.id[0])

                # Check if the detected class is in the allowed classes
                if class_name in self.allowed_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]

                    # Convert box_color (RGB) to BGR for OpenCV drawing
                    bgr_color = self.box_color[::-1]

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bgr_color, 2)

                    # Prepare label
                    label = f"ID:{track_id} {class_name} {conf:.2f}"

                    # Calculate text size and draw background rectangle for label
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1 - 5), bgr_color, -1)

                    # Draw label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame, annotated_frame # Return both original and annotated

# This block will only run if ot.py is executed directly, not when imported
if __name__ == "__main__":
    print("This script is now primarily designed to be imported as a module for GUI integration.")
    print("If you run it directly, it will only initialize the tracker. It will not process videos.")
    tracker = YOLOv8Tracker(model_path=DEFAULT_MODEL_PATH, box_color=(0, 255, 0))
    print("Tracker initialized. Exiting.")
