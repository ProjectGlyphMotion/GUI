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

## ✨ What's New
- 👀 **Updated UI:** Totally inspired by our website aesthetics.
- 🔲 **Interactive Region of Interest (ROI):** Draw a custom bounding box on the first frame of your video to restrict YOLOv8 tracking to a specific area. This dramatically reduces processing overhead and focuses your results.
- 📱 **Portrait Video Correction:** Automatically detects video rotation metadata and rotates mobile portrait videos to their correct orientation before processing.

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

On Linux, you may also need:
```bash
sudo apt install python3-pil.imagetk
```

> Ensure **FFmpeg** is installed and accessible from the command line.

### 🎮 GPU Acceleration (Optional but Recommended)

By default, `pip install torch` installs the **CPU-only** version. The app will still work on CPU, but tracking will be significantly slower.

To enable GPU acceleration, install **PyTorch with CUDA** from the [official PyTorch site](https://pytorch.org/get-started/locally/) based on your setup:

| CUDA Version | Install Command |
|---|---|
| CUDA 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| CUDA 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| CPU Only | `pip install torch torchvision` (default) |

> **Not sure which CUDA version you have?** Run `nvcc --version` or `nvidia-smi` in your terminal.

> ⚠️ CUDA can be a pain in the 🍑 if you have a 30 or 40 series card — here is a [FIX](https://www.reddit.com/r/StableDiffusion/comments/13n16r7/cuda_not_available_fix_for_anybody_that_is/)

**Verify GPU is detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

### 🚀Launch the GUI:

```bash
python3 GUI.py
```

### 🖥️ Inside the GUI:

1. Browse or drag a `.mp4`, `.avi`, or `.mov` file.
2. Select the desired bounding box color (RGB or via color picker).
3. Optionally you can also use the Region of Interest (ROI)
4. Click **▶ Run Tracker**.
---

## 👀 Preview 

![GUI Preview](assets/preview.gif)

- Note: This looks kinda distorted because of the conversion.
---

## 📂 Output

- Format: `<original_filename>_tracked.mp4`
- Saved under: `output/YYYYMMDD-HHMMSS/`
- If FFmpeg is available, original audio is preserved.

---

## ✅ [Example](https://drive.google.com/file/d/1LvyAFXvXy03LXBlhp-yG3kaQcWL1nHmM/view) Result

---

## 📃 License

MIT License © 2025 Sayan Sarkar & Shitij Halder

---

## ❤️ Credits

Made with love by **Sayan** and **Shitij**

This project is based on the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), an acclaimed real-time object detection and image segmentation model.
</immersive>
