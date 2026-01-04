# Implementation Plan for Traffic Light Classification

## 1. Setup and Imports
- **Libraries:** TensorFlow/Keras (for deep learning), NumPy (matrix ops), Matplotlib/Seaborn (visualization), Scikit-learn (metrics/splitting), OpenCV or PIL (image handling).
- **Configuration:** Set random seeds for reproducibility. Check for GPU availability for faster execution.

## 2. Data Preparation
- **Data Loading:** 
  - Load images from `traffic/Green` and `traffic/Red` directories.
  - Convert images to arrays.
- **Preprocessing:**
  - **Resizing:** Resize all images to a compact resolution (e.g., 64x64 or 128x128) to speed up convolution operations.
  - **Normalization:** Scale pixel values to range [0, 1].
  - **Labeling:** Encode categories (Green=0, Red=1).
- **Splitting:** 
  - Use `train_test_split` to create Training (70%), Validation (15%), and Testing (15%) sets.
- **Data Augmentation:** 
  - Use `ImageDataGenerator` or `tf.data` to apply on-the-fly transformations (Rotation, Zoom, Horizontal Flip) to prevent overfitting and improve generalization without bloating memory.

## 3. CNN Model Design
- **Architecture:** Design a lightweight Sequential 2D CNN optimized for speed.
  - **Layer 1:** Conv2D (32 filters, 3x3 kernel) + ReLU activation + MaxPooling2D (2x2).
  - **Layer 2:** Conv2D (64 filters, 3x3 kernel) + ReLU activation + MaxPooling2D (2x2).
  - **Layer 3:** Conv2D (128 filters, 3x3 kernel) + ReLU activation + MaxPooling2D (2x2).
  - **Classifier:** Flatten -> Dense (128 units, ReLU) -> Dropout (0.5) -> Output Dense (1 unit, Sigmoid).
- **Compilation:**
  - **Optimizer:** Adam (Adaptive Moment Estimation) for fast convergence.
  - **Loss Function:** Binary Crossentropy.
  - **Metrics:** Accuracy.

## 4. Training and Evaluation
- **Training:** 
  - Train for a moderate number of epochs (e.g., 15-20).
  - **Callback:** Implement `EarlyStopping` (monitor='val_loss', patience=3) to stop training immediately when the model stops improving, saving time.
- **Visualization:**
  - Generate line plots for **Training vs. Validation Loss**.
  - Generate line plots for **Training vs. Validation Accuracy**.
- **Metrics:** 
  - Calculate final accuracy on the unseen Test set.
  - Generate a Confusion Matrix to visualize False Positives/Negatives.

## 5. External Image Testing
- **Input:** Load 5 arbitrary traffic light images downloaded from the internet (ensure they are not in the training set).
- **Inference:** 
  - Preprocess (resize/normalize) exactly as per training data.
  - Predict class probabilities.
- **Output:** Display images alongside their predicted labels (Red/Green) and confidence scores.

## 6. Accuracy Improvement Strategies
- **Strategy 1: Transfer Learning (MobileNetV2)**
  - **Concept:** Utilize a pre-trained model (MobileNetV2 is chosen for its speed/efficiency) as a feature extractor.
  - **Implementation:** Freeze base layers, add custom classification head, and fine-tune. This typically provides a significant accuracy boost with minimal training.
- **Strategy 2: Learning Rate Scheduling**
  - **Concept:** Implement `ReduceLROnPlateau`.
  - **Implementation:** Dynamically lower the learning rate when validation accuracy plateaus, allowing the model to settle into a deeper global minimum.
