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

# Specify the resources directory in the system path for imports
sys.path.append(resource_path("resources"))

# Path For Resources
logo_path = resource_path("resources/LOGO.png")
icon_path = resource_path("resources/Icon.ico")

# Path For Model
model_path = resource_path("resources/yolov8m.pt") 

try:
    from ot import YOLOv8Tracker, check_ffmpeg, process_audio_ffmpeg
    FFMPEG_AVAILABLE_GLOBAL = False
    if check_ffmpeg():
        FFMPEG_AVAILABLE_GLOBAL = True
except ImportError as e:
    messagebox.showerror("Error", f"Could not import YOLOv8Tracker from ot.py... Error: {e}")
    sys.exit(1)


if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))


# Design Tokens (This is kindaa isnpired by our website design)
COL_BG           = "#000000"   
COL_SURFACE      = "#0d0d0d"   
COL_CARD         = "#111111"   
COL_INPUT        = "#1a1a1a"   
COL_BORDER       = "#2a1a00"   
COL_BORDER_ACC   = "#f9731633" 
COL_ACCENT       = "#f97316"   
COL_ACCENT_DARK  = "#ea580c"   
COL_ACCENT_DIM   = "#f9731640" 
COL_DANGER       = "#dc2626"  
COL_DANGER_DARK  = "#b91c1c"   
COL_TEXT         = "#e5e5e5"   
COL_TEXT_SEC     = "#9ca3af"   
COL_TEXT_DIM     = "#6b7280"   
COL_CONSOLE_BG   = "#080808"   
COL_CONSOLE_FG   = "#f97316"   
COL_PREVIEW_BG   = "#0a0a0a"   
FONT_BRAND       = ("Segoe UI", 18, "bold")
FONT_SUBTITLE    = ("Segoe UI", 10)
FONT_HEADING     = ("Segoe UI", 11, "bold")
FONT_LABEL       = ("Segoe UI", 10)
FONT_BUTTON      = ("Segoe UI", 10, "bold")
FONT_CONSOLE     = ("Consolas", 9)
FONT_FOOTER      = ("Segoe UI", 8)
FONT_SMALL       = ("Segoe UI", 9)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Project GlyphMotion") 
        self.root.geometry("1050x900")
        self.root.resizable(True, True)
        self.root.configure(bg=COL_BG)

        try:
            self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"[WARNING] Icon.ico not found or could not be loaded: {e}. Skipping icon display.")
            

        self.video_path = tk.StringVar()
        self.r_val = tk.StringVar(value="0")
        self.g_val = tk.StringVar(value="255")
        self.b_val = tk.StringVar(value="0")
        self.watermark_enabled = tk.BooleanVar(value=True)
        self.roi_bbox = None  # (x, y, w, h) or None

        self.tracker_running = False
        self.output_queue = queue.Queue()
        self.raw_frame_queue = queue.Queue(maxsize=2)
        self.processed_frame_queue = queue.Queue(maxsize=2)

        self.cap = None
        self.tracker_instance = None

        self.video_playback_thread = None
        self.stop_event = threading.Event()
        self.video_frame_delay_ms = 10


        try:
            self.logo_img = Image.open(logo_path).resize((40, 40), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(self.logo_img)
        except FileNotFoundError:
            self.output_queue.put(f"[WARNING] LOGO.png not found at {logo_path}. Skipping logo display.")
            self.logo_photo = None

        # Main container
        self.main_container = tk.Frame(self.root, bg=COL_BG)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        self.main_container.grid_rowconfigure(0, weight=0)  # Header
        self.main_container.grid_rowconfigure(1, weight=0)  # Input Settings
        self.main_container.grid_rowconfigure(2, weight=0)  # Buttons
        self.main_container.grid_rowconfigure(3, weight=3)  # Video Preview
        self.main_container.grid_rowconfigure(4, weight=1)  # Console
        self.main_container.grid_rowconfigure(5, weight=0)  # Footer

        self.main_container.grid_columnconfigure(0, weight=1)

        self.setup_styles()
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)
        self.create_widgets()
        self.poll_output_queue()
        self.poll_video_frames()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('default')

        # Progress bar
        self.style.configure("Custom.Horizontal.TProgressbar",
            troughcolor=COL_INPUT,
            background=COL_ACCENT,
            bordercolor=COL_INPUT,
            lightcolor=COL_ACCENT,
            darkcolor=COL_ACCENT_DARK,
            borderwidth=0,
        )

    # Helpers

    def _make_card(self, parent, **grid_kwargs):
        """Create a card-like frame with subtle orange border."""
        outer = tk.Frame(parent, bg=COL_ACCENT_DARK, bd=0)
        outer.grid(**grid_kwargs)
        inner = tk.Frame(outer, bg=COL_CARD, bd=0)
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        return inner

    def _make_button(self, parent, text, command, accent=True, danger=False, **kwargs):
        """Create a styled button."""
        if danger:
            bg, active_bg, fg = COL_DANGER, COL_DANGER_DARK, "#ffffff"
        elif accent:
            bg, active_bg, fg = COL_ACCENT, COL_ACCENT_DARK, "#000000"
        else:
            bg, active_bg, fg = COL_INPUT, "#252525", COL_TEXT

        btn = tk.Button(parent, text=text, command=command,
            font=FONT_BUTTON, fg=fg, bg=bg,
            activeforeground=fg, activebackground=active_bg,
            relief=tk.FLAT, bd=0, cursor="hand2",
            padx=16, pady=6, **kwargs)
        return btn

    def _make_entry(self, parent, textvariable=None, width=50, **kwargs):
        """Create a styled entry field."""
        entry = tk.Entry(parent, textvariable=textvariable, width=width,
            bg=COL_INPUT, fg=COL_TEXT, insertbackground=COL_ACCENT,
            relief=tk.FLAT, bd=0, font=FONT_LABEL,
            highlightthickness=1, highlightbackground="#333333",
            highlightcolor=COL_ACCENT, **kwargs)
        return entry

    def choose_color(self):
        color = askcolor(title="Select Box Color", initialcolor=(int(self.r_val.get()), int(self.g_val.get()), int(self.b_val.get())))
        if color[0]:
            r, g, b = map(int, color[0])
            self.r_val.set(str(r))
            self.g_val.set(str(g))
            self.b_val.set(str(b))

    # Widget Creation

    def create_widgets(self):

        # HEADER
       
        header_frame = tk.Frame(self.main_container, bg=COL_BG)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(8, 4))

        header_center = tk.Frame(header_frame, bg=COL_BG)
        header_center.pack(anchor="center")

        if self.logo_photo:
            tk.Label(header_center, image=self.logo_photo, bg=COL_BG).pack(side="left", padx=(0, 10))

        title_block = tk.Frame(header_center, bg=COL_BG)
        title_block.pack(side="left")

        tk.Label(title_block, text="Project GlyphMotion",
            fg=COL_ACCENT, bg=COL_BG, font=FONT_BRAND).pack(anchor="w")
        tk.Label(title_block, text="Real-time object tracking powered by YOLOv8",
            fg=COL_TEXT_SEC, bg=COL_BG, font=FONT_SUBTITLE).pack(anchor="w")

        # INPUT SETTINGS CARD
        input_card = self._make_card(self.main_container,
            row=1, column=0, sticky="ew", padx=4, pady=(12, 4))

        # Section title
        tk.Label(input_card, text="⚙  Input Settings",
            fg=COL_TEXT, bg=COL_CARD, font=FONT_HEADING).pack(anchor="w", padx=14, pady=(12, 4))

        # Separator
        tk.Frame(input_card, bg="#1f1f1f", height=1).pack(fill="x", padx=14, pady=(0, 8))

        # Video input row
        video_row = tk.Frame(input_card, bg=COL_CARD)
        video_row.pack(fill="x", padx=14, pady=(0, 8))

        tk.Label(video_row, text="📹  Input Video",
            fg=COL_TEXT_SEC, bg=COL_CARD, font=FONT_LABEL).pack(side="left", padx=(0, 8))

        self._make_entry(video_row, textvariable=self.video_path, width=55).pack(
            side="left", expand=True, fill="x", padx=(0, 8), ipady=4)

        self._make_button(video_row, "📂  Browse", self.browse_video, accent=False).pack(side="left")

        # Color row
        color_row = tk.Frame(input_card, bg=COL_CARD)
        color_row.pack(fill="x", padx=14, pady=(0, 8))

        tk.Label(color_row, text="🎨  Box Color (RGB)",
            fg=COL_TEXT_SEC, bg=COL_CARD, font=FONT_LABEL).pack(side="left", padx=(0, 8))

        for var in (self.r_val, self.g_val, self.b_val):
            self._make_entry(color_row, textvariable=var, width=4).pack(
                side="left", padx=(0, 6), ipady=4)

        self._make_button(color_row, "🎨  Pick Color", self.choose_color, accent=False).pack(side="left", padx=(4, 0))

        # ACTION BUTTONS
        button_frame = tk.Frame(self.main_container, bg=COL_BG)
        button_frame.grid(row=2, column=0, pady=(8, 8))

        self.run_button = self._make_button(button_frame, "▶  Start Tracking", self.run_tracker, accent=True)
        self.run_button.pack(side="left", padx=(0, 10), ipady=2)

        self.stop_button = self._make_button(button_frame, "⏹  Stop Tracker", self.stop_tracker, danger=True, state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=(0, 10), ipady=2)

        self.roi_button = self._make_button(button_frame, "🔲  Select ROI", self.select_roi, accent=False)
        self.roi_button.pack(side="left", padx=(0, 10), ipady=2)

        self.roi_clear_button = self._make_button(button_frame, "✖ Clear ROI", self.clear_roi, accent=False)
        self.roi_clear_button.pack(side="left", ipady=2)
        self.roi_clear_button.config(state=tk.DISABLED)

        self.roi_status_label = tk.Label(button_frame, text="",
            fg=COL_TEXT_SEC, bg=COL_BG, font=FONT_SMALL)
        self.roi_status_label.pack(side="left", padx=(10, 0))

        # LIVE PREVIEW CARD
        preview_card = self._make_card(self.main_container,
            row=3, column=0, sticky="nsew", padx=4, pady=(4, 4))

        tk.Label(preview_card, text="👁  Live Preview",
            fg=COL_TEXT, bg=COL_CARD, font=FONT_HEADING).pack(anchor="w", padx=14, pady=(12, 4))

        tk.Frame(preview_card, bg="#1f1f1f", height=1).pack(fill="x", padx=14, pady=(0, 8))

        preview_inner = tk.Frame(preview_card, bg=COL_CARD)
        preview_inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 12))

        preview_inner.grid_columnconfigure(0, weight=1)
        preview_inner.grid_columnconfigure(1, weight=1)
        preview_inner.grid_rowconfigure(1, weight=1)

        # Input panel
        tk.Label(preview_inner, text="Input",
            fg=COL_TEXT_SEC, bg=COL_CARD, font=FONT_SMALL).grid(row=0, column=0, sticky="n", pady=(0, 4))
        self.video_input_label = tk.Label(preview_inner, bg=COL_PREVIEW_BG,
            highlightthickness=1, highlightbackground="#1f1f1f")
        self.video_input_label.config(width=480, height=270)
        self.video_input_label.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 4))

        # Output panel
        tk.Label(preview_inner, text="Output",
            fg=COL_TEXT_SEC, bg=COL_CARD, font=FONT_SMALL).grid(row=0, column=1, sticky="n", pady=(0, 4))
        self.video_output_label = tk.Label(preview_inner, bg=COL_PREVIEW_BG,
            highlightthickness=1, highlightbackground="#1f1f1f")
        self.video_output_label.config(width=480, height=270)
        self.video_output_label.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 4))

        # CONSOLE CARD
        console_card = self._make_card(self.main_container,
            row=4, column=0, sticky="nsew", padx=4, pady=(4, 4))

        tk.Label(console_card, text="📋  Tracking Output Console",
            fg=COL_TEXT, bg=COL_CARD, font=FONT_HEADING).pack(anchor="w", padx=14, pady=(12, 4))

        tk.Frame(console_card, bg="#1f1f1f", height=1).pack(fill="x", padx=14, pady=(0, 8))

        console_inner = tk.Frame(console_card, bg=COL_CARD)
        console_inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 12))
        console_inner.grid_rowconfigure(0, weight=1)
        console_inner.grid_columnconfigure(0, weight=1)

        self.output_text = scrolledtext.ScrolledText(console_inner,
            width=80, height=10,
            bg=COL_CONSOLE_BG, fg=COL_CONSOLE_FG,
            insertbackground=COL_ACCENT, font=FONT_CONSOLE,
            relief=tk.FLAT, bd=0,
            highlightthickness=1, highlightbackground="#1f1f1f")
        self.output_text.grid(row=0, column=0, sticky="nsew")
        self.output_text.configure(state="disabled")

        self.progress = ttk.Progressbar(console_inner,
            orient="horizontal", mode="determinate",
            style="Custom.Horizontal.TProgressbar")
        self.progress.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        # FOOTER
        footer_frame = tk.Frame(self.main_container, bg=COL_BG)
        footer_frame.grid(row=5, column=0, sticky="ew", pady=(4, 8))

        tk.Label(footer_frame, text="Made with ❤ by Sayan and Shitij",
            fg=COL_TEXT_DIM, bg=COL_BG, font=FONT_FOOTER).pack()
        tk.Label(footer_frame, text="Powered by Ultralytics YOLOv8 — Real-time object detection and image segmentation",
            fg=COL_TEXT_DIM, bg=COL_BG, font=FONT_FOOTER).pack()


    # Functionality
    def browse_video(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_path.set(file_path)

    def select_roi(self):
        """Open a Tkinter-based ROI selector on the first frame of the selected video."""
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file first.")
            return
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Selected video file does not exist.")
            return

        cap = cv2.VideoCapture(self.video_path.get())
        # Disable OpenCV auto-rotation so we can handle it manually
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        except Exception:
            pass
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return

        ret, first_frame = cap.read()

        # Handle rotation metadata for portrait videos
        try:
            orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
            if orientation == 90:
                first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 270:
                first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception:
            pass

        cap.release()

        if not ret:
            messagebox.showerror("Error", "Could not read first frame from video.")
            return

        # Dynamic resizing for selection window
        MAX_WINDOW_H = 700
        MAX_WINDOW_W = 1200
        h, w = first_frame.shape[:2]
        scale_factor = 1.0

        if h > MAX_WINDOW_H or w > MAX_WINDOW_W:
            scale_factor = min(MAX_WINDOW_H / h, MAX_WINDOW_W / w)

        disp_w = int(w * scale_factor)
        disp_h = int(h * scale_factor)
        display_frame = cv2.resize(first_frame, (disp_w, disp_h)) if scale_factor != 1.0 else first_frame

        # Convert to PIL for Tkinter display
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        tk_img = ImageTk.PhotoImage(pil_img)

        # Build Toplevel for ROI selector
        roi_win = tk.Toplevel(self.root)
        roi_win.title("Select ROI — Draw a rectangle, then click Confirm")
        roi_win.configure(bg=COL_BG)
        roi_win.resizable(False, False)
        roi_win.grab_set()

        # Instructions
        tk.Label(roi_win, text="Draw a rectangle on the frame to select your Region of Interest",
                 fg=COL_TEXT_SEC, bg=COL_BG, font=FONT_LABEL).pack(pady=(8, 4))

        canvas = tk.Canvas(roi_win, width=disp_w, height=disp_h,
                           bg=COL_BG, highlightthickness=0, cursor="crosshair")
        canvas.pack(padx=8)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas._tk_img = tk_img 

        # Drawing state
        roi_state = {"start_x": 0, "start_y": 0, "rect_id": None, "result": None}

        def on_press(event):
            roi_state["start_x"] = event.x
            roi_state["start_y"] = event.y
            if roi_state["rect_id"]:
                canvas.delete(roi_state["rect_id"])
            roi_state["rect_id"] = canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline=COL_ACCENT, width=2, dash=(4, 4))

        def on_drag(event):
            if roi_state["rect_id"]:
                canvas.coords(roi_state["rect_id"],
                              roi_state["start_x"], roi_state["start_y"],
                              event.x, event.y)

        def on_release(event):
            if roi_state["rect_id"]:
                canvas.coords(roi_state["rect_id"],
                              roi_state["start_x"], roi_state["start_y"],
                              event.x, event.y)

        def confirm():
            if roi_state["rect_id"]:
                x1, y1, x2, y2 = canvas.coords(roi_state["rect_id"])
                # Normalize so x1<x2 and y1<y2
                sx, sy = min(x1, x2), min(y1, y2)
                ex, ey = max(x1, x2), max(y1, y2)
                rw, rh = ex - sx, ey - sy
                if rw > 5 and rh > 5:
                    # Map back to full resolution
                    real_x = int(sx / scale_factor)
                    real_y = int(sy / scale_factor)
                    real_w = int(rw / scale_factor)
                    real_h = int(rh / scale_factor)
                    roi_state["result"] = (real_x, real_y, real_w, real_h)
            roi_win.destroy()

        def cancel():
            roi_win.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        # Buttons
        btn_frame = tk.Frame(roi_win, bg=COL_BG)
        btn_frame.pack(pady=(8, 10))
        self._make_button(btn_frame, "✔  Confirm ROI", confirm, accent=True).pack(side="left", padx=(0, 10), ipady=2)
        self._make_button(btn_frame, "✖  Cancel", cancel, accent=False).pack(side="left", ipady=2)

        self.output_queue.put("[INFO] ROI selector opened. Draw a rectangle and click Confirm.")

        # Wait for the window to close
        self.root.wait_window(roi_win)

        if roi_state["result"]:
            real_x, real_y, real_w, real_h = roi_state["result"]
            self.roi_bbox = roi_state["result"]
            self.output_queue.put(f"✅ [SUCCESS] ROI selected: x={real_x}, y={real_y}, w={real_w}, h={real_h}")
            self.roi_status_label.config(text=f"ROI: {real_w}×{real_h} @ ({real_x},{real_y})", fg=COL_ACCENT)
            self.roi_clear_button.config(state=tk.NORMAL)
        else:
            self.output_queue.put("⚠️ [WARNING] ROI selection cancelled. Processing full frame.")

    def clear_roi(self):
        """Clear the current ROI selection."""
        self.roi_bbox = None
        self.roi_status_label.config(text="", fg=COL_TEXT_SEC)
        self.roi_clear_button.config(state=tk.DISABLED)
        self.output_queue.put("[INFO] ROI cleared. Will process full frame.")

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
        self.tracker_instance = YOLOv8Tracker(
            box_color=box_color,
            roi_bbox=self.roi_bbox,
            enable_watermark=self.watermark_enabled.get()
        )

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
        # Disable OpenCV auto-rotation so we handle it manually via CAP_PROP_ORIENTATION_META
        try:
            self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        except Exception:
            pass

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

        # Detect portrait videos using rotation metadata from the video file
        # Phone cameras store portrait video as landscape pixels + a rotation tag (90° or 270°)
        # OpenCV reads raw pixels without applying rotation, so we must handle it manually
        # This is kind of a drity hack but hey if it works it works!!!!
        rotate_for_portrait = False
        rotation_angle = None
        try:
            orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))
            if orientation in (90, 270):
                rotate_for_portrait = True
                rotation_angle = orientation
                self.output_queue.put(f"[INFO] Video has rotation metadata: {orientation}°. Frames will be rotated to correct portrait orientation.")
            else:
                self.output_queue.put(f"[INFO] Video rotation metadata: {orientation}°. No rotation needed.")
        except Exception:
            self.output_queue.put(f"[INFO] Could not read rotation metadata. Assuming no rotation needed.")

        output_width = frame_width
        output_height = frame_height
        if rotate_for_portrait:
            output_width = frame_height
            output_height = frame_width
            self.output_queue.put(f"[INFO] Output video dimensions will be {output_width}x{output_height} (after rotation).")

        out = cv2.VideoWriter(temp_silent_video_path, fourcc, fps, (output_width, output_height))

        if not out.isOpened():
            self.output_queue.put(f"❌ [ERROR] Could not open video writer for {temp_silent_video_path}")
            self.root.after(100, self.stop_tracker)
            return

        start_time = time.time()
        while not self.stop_event.is_set() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Rotate portrait videos based on rotation metadata
            if rotate_for_portrait:
                if rotation_angle == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_angle == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

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