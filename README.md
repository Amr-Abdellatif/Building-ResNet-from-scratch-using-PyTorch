# This is implmentation of Resnet Architecture searched from original paper and pyTorch official documentation for resnet 

# ResNet-18 Model Walkthrough

## 1. Initial Convolution Layer (Conv1):
- **Input Shape:** (1, 3, 214, 214)
- **Conv1 Parameters:** 64 filters, kernel size 7x7, stride 2, padding 3
- **Output Shape:** (1, 64, 107, 107)

## 2. Batch Normalization and ReLU:
- **Input Shape:** (1, 64, 107, 107)
- **Batch Normalization:** Adjusts the scale and offset of the features.
- **ReLU Activation:** Introduces non-linearity by setting negative values to zero.
- **Output Shape:** (1, 64, 107, 107)

## 3. MaxPooling Layer:
- **Input Shape:** (1, 64, 107, 107)
- **MaxPooling Parameters:** kernel size 3x3, stride 2, padding 1
- **Output Shape:** (1, 64, 54, 54)

## 4. Residual Blocks (Layer 1):
- **Input Shape:** (1, 64, 54, 54)
- **Bottleneck Block 1:**
  - **Input Shape:** (1, 64, 54, 54)
  - **Output Shape:** (1, 256, 54, 54)
- **Bottleneck Block 2:**
  - **Input Shape:** (1, 256, 54, 54)
  - **Output Shape:** (1, 256, 54, 54)
- **Output Shape (Layer 1):** (1, 256, 54, 54)

## 5. Residual Blocks (Layer 2):
- **Input Shape:** (1, 256, 54, 54)
- **Bottleneck Block 1:**
  - **Input Shape:** (1, 256, 54, 54)
  - **Output Shape:** (1, 512, 27, 27)
- **Bottleneck Block 2:**
  - **Input Shape:** (1, 512, 27, 27)
  - **Output Shape:** (1, 512, 27, 27)
- **Output Shape (Layer 2):** (1, 512, 27, 27)

## 6. Residual Blocks (Layer 3):
- **Input Shape:** (1, 512, 27, 27)
- **Bottleneck Block 1:**
  - **Input Shape:** (1, 512, 27, 27)
  - **Output Shape:** (1, 1024, 14, 14)
- **Bottleneck Block 2:**
  - **Input Shape:** (1, 1024, 14, 14)
  - **Output Shape:** (1, 1024, 14, 14)
- **Output Shape (Layer 3):** (1, 1024, 14, 14)

## 7. Residual Blocks (Layer 4):
- **Input Shape:** (1, 1024, 14, 14)
- **Bottleneck Block 1:**
  - **Input Shape:** (1, 1024, 14, 14)
  - **Output Shape:** (1, 2048, 7, 7)
- **Bottleneck Block 2:**
  - **Input Shape:** (1, 2048, 7, 7)
  - **Output Shape:** (1, 2048, 7, 7)
- **Output Shape (Layer 4):** (1, 2048, 7, 7)

## 8. Global Average Pooling:
- **Input Shape:** (1, 2048, 7, 7)
- **Global Average Pooling:** Computes the average value of each channel across the spatial dimensions.
- **Output Shape:** (1, 2048, 1, 1)

## 9. Fully Connected (FC) Layer:
- **Input Shape:** (1, 2048)
- **FC Layer Parameters:** Output size 1000 (assuming the original ResNet-18 classification task)
- **Output Shape:** (1, 1000)

The final output represents the model's predictions across the 1000 classes. Please note that if your task involves a different number of classes, the FC layer's output size would be adjusted accordingly.
