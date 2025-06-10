# Project GlyphMotion 🎯🌀

This project is a fully-featured GUI application for real-time object detection and tracking using [YOLOv8](https://github.com/ultralytics/ultralytics). It lets users load videos, preview detections live, customize bounding box colors, and export annotated videos—complete with original audio.

---

## 🚀 Features

- ✅ **YOLOv8 Object Detection and Tracking**
- 🖼️ **Tkinter GUI with Live Video Preview**
- 📂 **Drag-and-Drop or File Picker for Input Videos**
- 🎨 **Customizable Bounding Box Colors (RGB)**
- 🔊 **Audio Preserved Using FFmpeg (if installed)**
- 📊 **Progress Bar and Console Logging**
- ⚙️ **Threaded Video Processing**
- 🧠 **Automatic CUDA Detection and Usage (if available)**
- 📁 **Auto-Saved Output in Timestamped Folders**

---

## 🛠️ Setup

### 📋 Clone the Repository

```bash
git clone https://github.com/ProjectGlyphMotion/GUI && cd GUI
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

> Ensure **FFmpeg** is installed and accessible from the command line.

> If you want GPU acceleration, install NVIDIA [CUDA](https://developer.nvidia.com/cuda-downloads) and the appropriate drivers.

> Cuda can be a pain in the 🍑HOLE if you have a 30 or 40 series card, here is a [FIX](https://www.reddit.com/r/StableDiffusion/comments/13n16r7/cuda_not_available_fix_for_anybody_that_is/)

---

### 🚀Launch the GUI:

```bash
python3 GUI.py
```

### 🖥️ Inside the GUI:

1. Browse or drag a `.mp4`, `.avi`, or `.mov` file.
2. Select the desired bounding box color (RGB or via color picker).
3. Click **▶ Run Tracker**.
4. Processed video will be saved in the `output/YYYYMMDD-HHMMSS/` directory.

---

## 📂 Output

- Format: `<original_filename>_tracked.mp4`
- Saved under: `output/YYYYMMDD-HHMMSS/`
- If FFmpeg is available, original audio is preserved.

---

## ✅ [Example](https://drive.google.com/file/d/1kV9-v5E5T7AiDEnNQWlmznmK0GhN4JMc/view) Result

---

## 📃 License

MIT License © 2025 Sayan Sarkar & Shitij Halder

---

## ❤️ Credits

Made with love by **Sayan** and **Shitij**

This project is based on the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), an acclaimed real-time object detection and image segmentation model.
</immersive>
