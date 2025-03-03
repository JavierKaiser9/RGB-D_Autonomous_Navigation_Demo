# 🚀 Autonomous Sidewalk Navigation Demo  
**RGB-D Fusion Steering with EfficientNetV2**  

## 🌟 Overview  
This repository presents a **deep learning-based Automatic Steering Module Demo** designed for **Autonomous Mobile Robots (AMRs)** navigating pedestrian sidewalks. The model leverages **RGB-D fusion** to classify diverse sidewalk scenarios and generate steering commands in real time.  

## 🛠️ Key Technologies  
✅ **Deep Learning Framework**: PyTorch for model development and training  
✅ **Model Architecture**: **Dual-Input, Single-Output EfficientNetV2**  
   - Input: **RGB + Depth** (from Intel RealSense D415)  
   - Output: **Steering Command** (turn left, right, or go straight)  
✅ **Hardware Acceleration**: **Intel OpenVINO** for real-time inference  
✅ **Depth Sensing**: **Intel RealSense D415** for **RGB-D fusion**  
✅ **Performance Optimization**: Model converted to **OpenVINO IR format** for embedded deployment  
✅ **Real-Time Execution**: Achieves **51.85 FPS** on an embedded system  
✅ **Development Environment**:  
   - **TensorFlow**: 2.10  
   - **OpenCV (CV2)**: 4.8.0  
   - **Python**: 3.10  
✅ **Training Hardware**: **NVIDIA GeForce RTX 3050**  

---

## 🚀 How to Use  

This repository provides two main functionalities:  
1️⃣ **Directly use the pre-trained OpenVINO model** for real-time inference.  
2️⃣ **Train your own model** using the provided architecture.  

### 🔹 1. Running the Pre-Trained OpenVINO Model  
If you want to use the **pre-trained OpenVINO model**, follow these steps:  

#### ✅ Requirements  
- **Intel RealSense D415** camera connected to your computer  
- Python environment with **OpenVINO** installed

## 🎯 Performance Highlights  
🔥 **99% accuracy** in sidewalk scenario classification  
⚡ Optimized for **low-latency execution** on edge devices  
🎯 Designed for **real-world autonomous navigation**  
