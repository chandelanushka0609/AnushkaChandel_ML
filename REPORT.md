# REPORT — Fashion MNIST (Wildcard Round)

**Author:** Your Name  
**Date:** 2025-08-27

## 1) Problem
Train a model to classify Fashion MNIST images into 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

## 2) Dataset
- Source: `keras.datasets.fashion_mnist` (built-in)
- Train/Validation/Test: 60k train / 10k test (10% of train used as validation)
- Preprocessing: Normalize to [0, 1], reshape to (28, 28, 1)

## 3) Approach
- **Architecture:** Small CNN with augmentation (RandomFlip/Rotation/Zoom), Dropout, Adam optimizer.
- **Loss:** SparseCategoricalCrossentropy
- **Callbacks:** EarlyStopping (patience=3), ModelCheckpoint (best val accuracy)
- **Epochs:** 10–15 (adjusted based on time)

## 4) Metrics & Results
Fill these after running:
- **Validation Accuracy:** X.XX
- **Test Accuracy:** X.XX
- **Test Loss:** X.XX

Include the generated **confusion_matrix.png** and sample predictions screenshots if possible.

## 5) Key Decisions / Originality
- Added **on-the-fly data augmentation** to improve generalization
- Simple, readable architecture for rapid training
- Saved best model by validation accuracy

## 6) How to Reproduce
- See `README.md` for exact steps
- Run the notebook end-to-end in Colab or `python train.py` locally

## 7) Limitations & Future Work
- Try learning rate schedules or label smoothing
- Evaluate with k-fold CV
- Attempt a deeper CNN or transfer learning (resizing to 96×96 and using MobileNetV2)
