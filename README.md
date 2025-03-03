# 🚀 Sidewalk Navigation Demo  
**RGB-D Fusion with EfficientNetV2**  

## 🌟 Overview  

This repository demonstrates the **architecture and implementation of a dual-input, single-output deep learning model** for real-time **sidewalk navigation**. A key focus is to:  
💡 Showcase the working principles of a **2-input, 1-output EfficientNetV2 model** for RGB-D fusion.  
💡 Present the **performance of OpenVINO-optimized models**.  

---

## 🛠️ Key Technologies  
**Deep Learning Framework**: TensorFlow for model development and training  
**Model Architecture**: **Dual-Input, Single-Output EfficientNetV2**  
   - Input: **RGB + Depth** (from Intel RealSense D415)  
   - Output: **Steering Command** (turn left, right, or go straight)  
✅ **Hardware Acceleration**: **Intel OpenVINO** for real-time inference  
✅ **Depth Sensing**: **Intel RealSense D415** for **RGB-D fusion**  
✅ **Performance Optimization**: Model converted to **OpenVINO IR format** for embedded deployment  
✅ **Real-Time Execution**: Achieves a mean of **50 FPS** on an embedded system  
✅ **Development Environment**:  
   - **TensorFlow**: 2.10  
   - **OpenCV (CV2)**: 4.8.0  
   - **Python**: 3.10  
✅ **Training Hardware**: **NVIDIA GeForce RTX 3050**  

---

#### 💻 Requirements  
- **Intel RealSense D415** camera connected to your computer  
- Python environment with **OpenVINO** installed

## 🚀 How to Use  

This repository provides two main functionalities:  
1️⃣ **Directly use the pre-trained OpenVINO model** for real-time inference.  
2️⃣ **Train your own model** using the provided architecture.  

### 🔹 1. Running the Pre-Trained OpenVINO Model  
If you want to use the **pre-trained OpenVINO model**, clone the repository and run the test_openvino_models.py file.

### 🔹 2. Train your own OpenVINO Model  
If you want to train your own model, change the paths in the train_two_input_one_output_model.py file to the locations where you want to store the training and test data. Then, you can transform the TensorFlow model into an OpenVINO model using the create_openvino_model.py file.

## 🎯 Performance Highlights  
🔥 **High accuracy** in sidewalk scenario classification  
⚡ Optimized for **low-latency execution** on edge devices  
