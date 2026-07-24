# Real-Time Sign Language to Speech Translation

A high-performance Computer Vision and Deep Learning project that translates Sign Language alphabet gestures into text and spoken audio in real-time. Built with a custom Convolutional Neural Network (CNN) in PyTorch and optimized for live webcam inference using OpenCV.

## ✨ Features
* **Real-Time Recognition:** Captures video feed via webcam and predicts hand gestures instantly without lag.
* **Custom CNN Architecture:** Engineered from scratch with 4 convolutional blocks, Batch Normalization, Adaptive Pooling, and Dropout layers for high generalization.
* **Text-to-Speech (TTS):** Integrated `pyttsx3` with multi-threading to read the generated words aloud without freezing the video frame.
* **Smart Inference Logic:** Includes logic to differentiate between static signs, dynamic spacing ("space"), backspacing ("del"), and word-completion ("nothing").
* **GPU Accelerated:** Fully optimized for CUDA, utilizing `pin_memory` and smart data loaders to train and infer on NVIDIA GPUs efficiently.

## 📊 Model Performance
The model was trained with dynamic Data Augmentation (Random Rotation, Color Jitter, Affine transformations) and an AdamW optimizer with a ReduceLROnPlateau learning rate scheduler.
* **Test Accuracy:** ~99.9% on the validation set.
* **Generalization:** Highly robust to lighting changes and minor hand rotations.
