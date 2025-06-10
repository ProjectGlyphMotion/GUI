import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
from tkinter.colorchooser import askcolor
import subprocess
import threading
import queue
import os
import sys
import re
import cv2
import time

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Path For Resources
logo_path = resource_path("LOGO.png")
icon_path = resource_path("Icon.ico")

# Path For Model
model_path = resource_path("yolov8m.pt") 

try:
    from ot import YOLOv8Tracker, check_ffmpeg, process_audio_ffmpeg
    FFMPEG_AVAILABLE_GLOBAL = False
    if check_ffmpeg():
        FFMPEG_AVAILABLE_GLOBAL = True
except ImportError as e:
    messagebox.showerror("Error", f"Could not import YOLOv8Tracker from ot.py. Make sure ot.py is in the same directory and contains the YOLOv8Tracker class. Error: {e}")
    sys.exit(1)


if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Project GlyphMotion") 
        self.root.geometry("1000x850")
        self.root.resizable(True, True)

        try:
            self.root.iconbitmap(os.path.join(BUNDLE_DIR, "Icon.ico"))
        except Exception as e:
            print(f"[WARNING] Icon.ico not found or could not be loaded: {e}. Skipping icon display.")

        # self.theme = "dark"
        self.video_path = tk.StringVar()
        self.r_val = tk.StringVar(value="0")
        self.g_val = tk.StringVar(value="255")
        self.b_val = tk.StringVar(value="0")

        self.tracker_running = False
        self.output_queue = queue.Queue()
        self.raw_frame_queue = queue.Queue(maxsize=2)
        self.processed_frame_queue = queue.Queue(maxsize=2)

        self.cap = None
        self.tracker_instance = None

        self.video_playback_thread = None
        self.stop_event = threading.Event()
        self.video_frame_delay_ms = 10

        logo_path = os.path.join(BUNDLE_DIR, "LOGO.png")
        try:
            self.logo_img = Image.open(logo_path).resize((32, 32))
            self.logo_photo = ImageTk.PhotoImage(self.logo_img)
        except FileNotFoundError:
            self.output_queue.put(f"[WARNING] LOGO.png not found at {logo_path}. Skipping logo display.")
            self.logo_photo = None

        self.main_container = tk.Frame(self.root, bg="#1e1e1e")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.main_container.grid_rowconfigure(0, weight=0)
        self.main_container.grid_rowconfigure(1, weight=0)
        self.main_container.grid_rowconfigure(2, weight=0)
        self.main_container.grid_rowconfigure(3, weight=3)
        self.main_container.grid_rowconfigure(4, weight=1)
        self.main_container.grid_rowconfigure(5, weight=0)
        self.main_container.grid_rowconfigure(6, weight=0)

        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(2, weight=1)

        self.setup_styles()
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)
        self.create_widgets()
        self.poll_output_queue()
        self.poll_video_frames()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('default')

        # Directly set dark theme colors as there's no toggling
        self.label_fg = "white"
        self.label_bg = "#1e1e1e"
        self.entry_bg = "#2e2e2e"
        self.entry_fg = "white"
        self.console_bg = "#111"
        self.console_fg = "#0f0"
        self.button_bg = "#0078D7"
        self.button_fg = "white"

        # Configure default ttk styles
        self.style.configure("TProgressbar", troughcolor="#2e2e2e", background="#4caf50", bordercolor="#2e2e2e", lightcolor="#4caf50", darkcolor="#4caf50")
        self.style.configure("TLabelFrame", background=self.label_bg, foreground=self.label_fg)
        self.style.configure("TLabelframe.Label", background=self.label_bg, foreground=self.label_fg, font=("Segoe UI", 10, "bold"))

        # Apply initial colors to main_container
        self.main_container.configure(bg=self.label_bg)

    def _update_widget_colors(self, widget):
        """Helper function to recursively update colors for widgets."""
        try:
            if isinstance(widget, (tk.Frame, tk.Label)):
                widget.config(bg=self.label_bg, fg=self.label_fg)

            elif isinstance(widget, tk.Button):
                widget.config(bg=self.button_bg, fg=self.button_fg,
                              activebackground=self.button_bg, activeforeground=self.button_fg)

            elif isinstance(widget, tk.Entry):
                widget.config(bg=self.entry_bg, fg=self.entry_fg, insertbackground=self.entry_fg)

            elif isinstance(widget, scrolledtext.ScrolledText):
                widget.config(bg=self.console_bg, fg=self.console_fg, insertbackground=self.entry_fg)
        except tk.TclError:
            pass

        if hasattr(widget, 'winfo_children'):
            for child in widget.winfo_children():
                self._update_widget_colors(child)

    def choose_color(self):
        color = askcolor(title="Select Box Color", initialcolor=(int(self.r_val.get()), int(self.g_val.get()), int(self.b_val.get())))
        if color[0]:
            r, g, b = map(int, color[0])
            self.r_val.set(str(r))
            self.g_val.set(str(g))
            self.b_val.set(str(b))

    def create_widgets(self):
        label_style = {"fg": self.label_fg, "bg": self.label_bg, "font": ("Segoe UI", 10)}
        entry_style = {"bg": self.entry_bg, "fg": self.entry_fg, "insertbackground": self.entry_fg, "relief": tk.FLAT}
        button_style = {"bg": self.button_bg, "fg": self.button_fg, "activebackground": "#005a9e", "activeforeground": self.button_fg, "relief": tk.FLAT}
        labelframe_style = {"fg": self.label_fg, "bg": self.label_bg, "font": ("Segoe UI", 10, "bold"), "bd": 2, "relief": "groove"}


        # Header Frame for Logo and Title
        header_frame = tk.Frame(self.main_container, bg=self.label_bg)
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        # Centering frame within header_frame
        center_header_content_frame = tk.Frame(header_frame, bg=self.label_bg)
        center_header_content_frame.pack(expand=True, anchor="center")

        if self.logo_photo:
            tk.Label(center_header_content_frame, image=self.logo_photo, bg=self.label_bg).pack(side="left", padx=(10, 5))
        tk.Label(center_header_content_frame, text="Project GlyphMotion", fg=self.label_fg, bg=self.label_bg, font=("Segoe UI", 12, "bold")).pack(side="left") # Changed from "YOLOv8 Object Tracker"

        # --- Input Settings LabelFrame ---
        input_settings_frame = tk.LabelFrame(self.main_container, text=" Input Settings ", **labelframe_style)
        input_settings_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 5))

        # Input Video row within input_settings_frame
        input_video_inner_frame = tk.Frame(input_settings_frame, bg=self.label_bg)
        input_video_inner_frame.pack(fill="x", padx=5, pady=5)
        tk.Label(input_video_inner_frame, text="📹 Input Video:", **label_style).pack(side="left", padx=5)
        tk.Entry(input_video_inner_frame, textvariable=self.video_path, width=50, **entry_style).pack(side="left", expand=True, fill="x", padx=5)
        tk.Button(input_video_inner_frame, text="📂 Browse", command=self.browse_video, **button_style).pack(side="left", padx=5)

        # Color selection row within input_settings_frame
        color_inner_frame = tk.Frame(input_settings_frame, bg=self.label_bg)
        color_inner_frame.pack(fill="x", padx=5, pady=5)
        tk.Label(color_inner_frame, text="🎨 Box Color (RGB):", **label_style).pack(side="left", padx=5)
        tk.Entry(color_inner_frame, width=4, textvariable=self.r_val, **entry_style).pack(side="left", padx=(0, 5))
        tk.Entry(color_inner_frame, width=4, textvariable=self.g_val, **entry_style).pack(side="left", padx=(0, 5))
        tk.Entry(color_inner_frame, width=4, textvariable=self.b_val, **entry_style).pack(side="left", padx=(0, 5))
        tk.Button(color_inner_frame, text="🎨 Pick Color", command=self.choose_color, **button_style).pack(side="left", padx=5)

        # Run/Stop Buttons
        button_frame = tk.Frame(self.main_container, bg=self.label_bg)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        self.run_button = tk.Button(button_frame, text="▶ Run Tracker", command=self.run_tracker, font=("Segoe UI", 10, "bold"), **button_style)
        self.run_button.pack(side="left", padx=5)
        self.stop_button = tk.Button(button_frame, text="⏹ Stop Tracker", command=self.stop_tracker, font=("Segoe UI", 10, "bold"), **button_style, state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=5)


        # --- Video Preview LabelFrame ---
        video_preview_frame = tk.LabelFrame(self.main_container, text=" Live Preview ", **labelframe_style)
        video_preview_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=10, pady=(5, 5))

        video_preview_frame.grid_rowconfigure(0, weight=1)
        video_preview_frame.grid_columnconfigure(0, weight=1)
        video_preview_frame.grid_columnconfigure(1, weight=1)

        tk.Label(video_preview_frame, text="Input", fg=self.label_fg, bg=self.label_bg, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="n", pady=(0, 5))
        self.video_input_label = tk.Label(video_preview_frame, bg="black")
        self.video_input_label.config(width=480, height=270)
        self.video_input_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        tk.Label(video_preview_frame, text="Output", fg=self.label_fg, bg=self.label_bg, font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="n", pady=(0, 5))
        self.video_output_label = tk.Label(video_preview_frame, bg="black")
        self.video_output_label.config(width=480, height=270)
        self.video_output_label.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)


        # --- Tracking Output Console LabelFrame ---
        output_frame = tk.LabelFrame(self.main_container, text=" Tracking Output Console ", **labelframe_style)
        output_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=10, pady=(5, 5))

        output_frame.grid_rowconfigure(0, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)

        self.output_text = scrolledtext.ScrolledText(output_frame, width=80, height=12, bg=self.console_bg, fg=self.console_fg,
                                                     insertbackground=self.entry_fg, font=("Consolas", 9), relief=tk.FLAT)
        self.output_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.output_text.configure(state="disabled")

        self.progress = ttk.Progressbar(output_frame, orient="horizontal", length=680, mode="determinate")
        self.progress.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))

        # Footer Text
        tk.Label(self.main_container, text="Made with Love by Sayan and Shitij", fg=self.label_fg, bg=self.label_bg, font=("Segoe UI", 9)).grid(row=5, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        tk.Label(self.main_container, text="This project is based on the Ultralytics YOLOv8, an acclaimed real-time object detection and image segmentation model.",
                 fg=self.label_fg, bg=self.label_bg, font=("Segoe UI", 8)).grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 10))


    def browse_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_path.set(file_path)

    def on_drop(self, event):
        file_path = event.data.strip('{}')
        if os.path.isfile(file_path) and file_path.lower().endswith((".mp4", ".avi", ".mov")):
            self.video_path.set(file_path)
            messagebox.showinfo("File Dropped", f"Loaded video:\n{file_path}")
        else:
            messagebox.showerror("Invalid File", "Only video files (.mp4, .avi, .mov) are supported.")

    def append_output(self, text):
        cleaned = text.replace('\r', '').strip()
        if cleaned:
            self.output_text.configure(state="normal")
            self.output_text.insert(tk.END, cleaned + "\n")
            lines = self.output_text.get("1.0", tk.END).split("\n")
            if len(lines) > 1000:
                self.output_text.delete("1.0", f"{len(lines)-1000}.0")
            self.output_text.see(tk.END)
            self.output_text.configure(state="disabled")

    def poll_output_queue(self):
        while not self.output_queue.empty():
            line = self.output_queue.get()
            self.append_output(line)
        self.root.after(100, self.poll_output_queue)

    def poll_video_frames(self):
        """Polls queues for new video frames and updates the labels."""
        try:
            if not self.raw_frame_queue.empty():
                raw_frame = self.raw_frame_queue.get_nowait()
                self.update_video_label(self.video_input_label, raw_frame)

            if not self.processed_frame_queue.empty():
                processed_frame = self.processed_frame_queue.get_nowait()
                self.update_video_label(self.video_output_label, processed_frame)
        except queue.Empty:
            pass
        except Exception as e:
            self.output_queue.put(f"[ERROR] Video frame polling error: {e}")

        self.root.after(self.video_frame_delay_ms, self.poll_video_frames)

    def update_video_label(self, label_widget, frame):
        """Converts an OpenCV frame to PhotoImage and updates a Tkinter Label."""
        if frame is None:
            return

        label_width = label_widget.winfo_width()
        label_height = label_widget.winfo_height()

        if label_width < 100 or label_height < 100:
            label_width = 480
            label_height = 270

        h_frame, w_frame, _ = frame.shape

        aspect_ratio_frame = w_frame / h_frame
        aspect_ratio_label = label_width / label_height

        new_width, new_height = label_width, label_height

        if aspect_ratio_frame > aspect_ratio_label:
            new_height = int(label_width / aspect_ratio_frame)
            new_width = label_width
        else:
            new_width = int(label_height * aspect_ratio_frame)
            new_height = label_height

        new_width = max(1, new_width)
        new_height = max(1, new_height)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2image = cv2.resize(cv2image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label_widget.imgtk = imgtk
        label_widget.config(image=imgtk)


    def run_tracker(self):
        if self.tracker_running:
            messagebox.showwarning("Busy", "Tracker is already running.")
            return
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file.")
            return
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Selected video file does not exist.")
            return

        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.tracker_running = True
        self.stop_event.clear()

        while not self.raw_frame_queue.empty(): self.raw_frame_queue.get()
        while not self.processed_frame_queue.empty(): self.processed_frame_queue.get()

        box_color = (int(self.r_val.get()), int(self.g_val.get()), int(self.b_val.get()))
        self.tracker_instance = YOLOv8Tracker(box_color=box_color)

        if self.tracker_instance.model is None:
            messagebox.showerror("Error", "YOLOv8 model could not be loaded. Check console for details.")
            self.root.after(100, self.stop_tracker)
            return

        self.video_playback_thread = threading.Thread(target=self._video_processing_loop, daemon=True)
        self.video_playback_thread.start()


    def stop_tracker(self):
        if not self.tracker_running:
            return

        self.output_queue.put("[INFO] Stopping tracker...")
        self.stop_event.set()

        if self.video_playback_thread and self.video_playback_thread.is_alive() and \
           self.video_playback_thread is not threading.current_thread():
            self.video_playback_thread.join(timeout=5)
            if self.video_playback_thread.is_alive():
                self.output_queue.put("[WARNING] Video processing thread did not terminate gracefully.")
            else:
                self.output_queue.put("[INFO] Video processing thread successfully joined.")

        if self.cap:
            self.cap.release()
            self.cap = None

        self.tracker_running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
        self.output_queue.put("[INFO] Tracker stopped.")


    def _video_processing_loop(self):
        """Reads video, processes frames, and puts them into queues."""
        video_path = self.video_path.get()
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            self.output_queue.put(f"❌ [ERROR] Could not open video file: {video_path}")
            self.root.after(100, self.stop_tracker)
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_frame_delay_ms = max(1, int(1000 / fps))

        self.output_queue.put(f"[INFO] Started processing video: {os.path.basename(video_path)}")
        self.output_queue.put(f"[INFO] FPS: {fps:.2f}, Total frames: {total_frames}")

        frame_count = 0
        self.progress.config(maximum=total_frames)
        self.progress.config(value=0)

        parent_dir = os.path.dirname(os.path.abspath(__file__))
        output_base_dir = os.path.join(parent_dir, "output")
        output_dir = os.path.join(output_base_dir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_silent_video_path = os.path.join(output_dir, f"{base_name}_tracked_silent.mp4")
        final_output_video_path = os.path.join(output_dir, f"{base_name}_tracked.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_silent_video_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            self.output_queue.put(f"❌ [ERROR] Could not open video writer for {temp_silent_video_path}")
            self.root.after(100, self.stop_tracker)
            return

        start_time = time.time()
        while not self.stop_event.is_set() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            raw_frame, annotated_frame = self.tracker_instance.process_frame(frame)

            try:
                self.raw_frame_queue.put_nowait(raw_frame)
                self.processed_frame_queue.put_nowait(annotated_frame)
            except queue.Full:
                pass

            out.write(annotated_frame)

            frame_count += 1
            self.root.after(0, lambda fc=frame_count: self.progress.config(value=fc))

        self.cap.release()
        out.release()
        self.output_queue.put("[INFO] Video processing loop finished.")

        if FFMPEG_AVAILABLE_GLOBAL:
            self.output_queue.put("[INFO] Attempting to merge audio...")
            if process_audio_ffmpeg(video_path, temp_silent_video_path, final_output_video_path):
                self.output_queue.put(f"✅ [SUCCESS] Final output video at: {final_output_video_path}")
                folder_path_to_open = os.path.abspath(os.path.dirname(final_output_video_path))
                if messagebox.askyesno("Open Folder", f"Video processing complete. Do you want to open the output folder:\n{folder_path_to_open}?"):
                    try:
                        os.startfile(folder_path_to_open)
                    except Exception as e:
                        self.output_queue.put(f"[WARNING] Could not open output folder: {e}")
                try:
                    if os.path.exists(temp_silent_video_path):
                        os.remove(temp_silent_video_path)
                        self.output_queue.put(f"[INFO] Deleted temporary silent video: {os.path.basename(temp_silent_video_path)}")
                except Exception as e:
                    self.output_queue.put(f"[WARNING] Could not delete temporary silent video: {e}")
            else:
                self.output_queue.put(f"❌ [ERROR] Audio merge failed. Silent video saved at: {temp_silent_video_path}")
                folder_path_to_open = os.path.abspath(os.path.dirname(temp_silent_video_path))
                if messagebox.askyesno("Open Folder", f"Audio merge failed. Do you want to open the folder containing the silent video:\n{folder_path_to_open}?"):
                    try:
                        os.startfile(folder_path_to_open)
                    except Exception as e:
                        self.output_queue.put(f"[WARNING] Could not open silent video folder: {e}")
        else:
            self.output_queue.put(f"ℹ️ [INFO] ffmpeg not available. Silent video saved at: {temp_silent_video_path}")
            folder_path_to_open = os.path.abspath(os.path.dirname(temp_silent_video_path))
            if messagebox.askyesno("Open Folder", f"ffmpeg not available. Do you want to open the folder containing the silent video:\n{folder_path_to_open}?"):
                try:
                    os.startfile(folder_path_to_open)
                except Exception as e:
                    self.output_queue.put(f"[WARNING] Could not open silent video folder: {e}")

        end_time = time.time()
        total_time = end_time - start_time
        self.output_queue.put(f"[INFO] Total processing time: {total_time:.2f} seconds.")

        self.root.after(100, self.stop_tracker)


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = App(root)
    root.mainloop()