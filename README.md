# Sanitary-Pad-Waste-Segregation-System
AI-powered real-time detection and automated sorting of absorbent hygiene products using computer vision and robotics.

## Overview

This system automatically detects and segregates sanitary pads from waste streams using:
- **Deep Learning**: CNN-based texture and feature extraction
- **Real-time Detection**: Live camera feed processing
- **Robotic Control**: Arduino-controlled servo mechanism for physical sorting

**Detection Class**: Absorbent Hygiene Product

## Hardware Requirements

- USB webcam (minimum 720p)
- Arduino Uno/Nano
- 2× SG90 servo motors
- Computer: Intel i3+, 4GB RAM minimum
- OS: Windows 10/11 or Ubuntu 20.04+

## Software Requirements

**Python Dependencies:**
```bash
pip install torch torchvision opencv-python numpy pillow pyserial
```

**Arduino IDE:** Version 1.8.19 or higher

## Files
```
├── detection_system_final__3_.py   # Main detection system
├── combined_servo__2_.ino          # Arduino servo control
├── project-1-at-2026-02-17.zip     # Labeled dataset (190 images)
├── ahp_model.pth                   # Trained model (auto-generated)
└── ahp_embeddings.npy              # Feature embeddings (auto-generated)
```

## Quick Start

### 1. Arduino Setup
1. Open `combined_servo__2_.ino` in Arduino IDE
2. Upload to Arduino
3. Wire servos:
   - Servo 1 signal → Pin 9
   - Servo 2 signal → Pin 10
   - 5V and GND to both servos

### 2. Dataset Setup
1. Extract `project-1-at-2026-02-17.zip`
2. Ensure it contains `images/` and `labels/` folders

### 3. Run System
```bash
python detection_system_final__3_.py
```

**First run**: Trains model (~20 mins)  
**Later runs**: Loads model instantly

## Configuration

### Python (detection_system_final__3_.py)
```python
ARDUINO_PORT = "COM3"    # Your Arduino port
CAMERA_ID    = 0         # Camera index
```

### Arduino (combined_servo__2_.ino)
```cpp
#define FLAP1_PIN     9     # Servo 1 pin
#define FLAP2_PIN     10    # Servo 2 pin
#define ANGLE_OPEN    45    # Open position
#define HOLD_TIME     3000  # Hold duration (ms)
```

## How It Works

**Detection Pipeline:**
1. Camera captures frame
2. CNN extracts features
3. Validates through:
   - Confidence check (>70%)
   - Texture similarity (>65%)
   - Multi-frame stability
4. Sends signal to Arduino if validated

**Servo Action:**
1. Receives signal from Python
2. Rotates LEFT to 45° (sanitary pad bin)
3. Holds for 3 seconds
4. Returns to REST (90°)

## Dataset

- **Format**: YOLO annotation
- **Images**: 190 labeled samples (224×224)
- **Class**: Absorbent Hygiene Product
- **Split**: 70% train / 20% val / 10% test

**Annotation format** (labels/*.txt):
```
class_id center_x center_y width height
0 0.5 0.348 0.901 0.604
```

## Model

- **Base**: ResNet18 (ImageNet pretrained)
- **Output**: 512-dimensional feature embeddings
- **Training**: 10 epochs, Adam optimizer
- **Accuracy**: ~80% validation

## Troubleshooting

**Camera not found:**
```python
CAMERA_ID = 1  # Try 1 or 2
```

**Arduino port error:**
- Windows: Check Device Manager → Ports
- Linux: `ls /dev/ttyUSB*`

**Servo not moving:**
- Verify wiring
- Check Serial Monitor output
- Ensure 5V supply is stable

## Performance

- **Detection Interval**: 5 seconds
- **Accuracy**: ~80%
- **False Positive Rate**: <15%
- **Processing Time**: ~100ms/frame

## Future Improvements

- Add depth camera for 3D positioning
- Increase dataset to 1000+ images
- Multi-class detection (wrapped/unwrapped/soiled)
- Web dashboard
- Conveyor belt integration

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

If you use this project or make improvements:
- **Fork** this repository
- **Open a Pull Request** with your changes
- **Share your results** - we'd love to see your modifications!

Feel free to open an issue if you find bugs or have suggestions.

## 📬 Contact

If you have questions, want to collaborate, or share updates:
- Open an issue in this repository
- Contributions and improvements are always appreciated!

## License

Open-source for educational and research purposes.

## Acknowledgments

- PyTorch (deep learning)
- OpenCV (computer vision)
- Arduino (hardware control)
- Roboflow (dataset management)


## ⭐ Support

If you found this project helpful:
- Give it a ⭐ star on GitHub
- Share it with others working on waste management solutions
- Consider contributing improvements back to the project

---

**Last Updated**: February 2026
