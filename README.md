# Real-Time Sign Language Translator

Developed for the **Artificial Intelligence course at Sharif University of Technology**, this project bridges the communication gap by translating sign language alphabet gestures into text and spoken audio in real-time. We built a custom PyTorch CNN and integrated it with OpenCV for seamless, zero-lag live video inference.

## Key Features
- **Zero-Lag Recognition:** Instantly predicts hand gestures directly from your webcam feed.
- **Custom CNN:** Built from scratch with 4 conv blocks, Batch Normalization, and Dropout for robust real-world generalization.
- **Speaks for You:** Uses `pyttsx3` with background multi-threading to read signed words out loud without freezing the video stream.
- **Smart Typing Logic:** Understands dynamic spacing, backspaces ("del"), and word completions to form natural sentences.

## Performance
- **~99.9% Validation Accuracy:** Trained with heavy dynamic data augmentation (color jittering, affine transforms, random rotations) and an AdamW optimizer.
- **Real-World Ready:** Highly resilient to lighting shifts, background clutter, and natural variations in hand angles.
