# Traffic Light Classifier - Implementation Plan (Fast & Simple)

## **1. Data Preparation & Preprocessing (2.1)**
- **Image Loading**: Load images from Red/ and Green/ folders using `ImageDataGenerator`
- **Data Exploration**: Display 2-3 sample images per class
- **Train-Test Split**: Simple 80-20 split
- **Data Augmentation** (minimal for speed):
  - Rotation (±10 degrees)
  - Horizontal flip only
- **Normalization**: Rescale to [0,1] using `1./255`
- **Image Resizing**: Small size **64×64** for faster training

## **2. CNN Architecture Design (2.2)**
- **Simple Model Structure** (3-4 layers total):
  - 2 Conv2D blocks (32, 64 filters) with ReLU + MaxPooling2D
  - Flatten layer
  - 1 Dense layer (64 units) + Dropout (0.5)
  - Output layer (1 unit, sigmoid activation)
- **Optimizer**: Adam (default learning rate)
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall

## **3. Training & Evaluation (2.3)**
- **Training Configuration**: 15-20 epochs (fast convergence)
- **Batch Size**: 32 (standard)
- **Validation Split**: 20% from training data
- **Visualizations**:
  - Training vs Validation Accuracy plot
  - Training vs Validation Loss plot
  - Simple accuracy table (train/test)
- **Performance Metrics**: Test accuracy percentage

## **4. External Testing (2.4)**
- Use 5 traffic light images from Google Images
- Resize to 64×64 and normalize
- Predict with model and show results
- Display image + predicted label side-by-side

## **5. Model Improvement Strategies (2.5)**
**Two simple evidence-based approaches**:
1. **Add more augmentation**: Add brightness/zoom → retrain → compare accuracy improvement
2. **Increase model depth**: Add one more Conv2D layer → retrain → show accuracy gain

**Alternative**: Adjust learning rate or increase epochs

## **Documentation Focus (Aligned with 60% grading)**
- Brief analysis of why accuracy improved/decreased
- Comment on overfitting if validation loss diverges
- Table comparing baseline vs improved model
- Simple interpretation of any errors

---

**Note**: This plan addresses all assignment requirements while maximizing the 60% marks for thorough analysis and 20% for visual presentation.
