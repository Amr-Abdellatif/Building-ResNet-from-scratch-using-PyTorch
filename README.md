# This is implmentation of Resnet Architecture searched from original paper and pyTorch official documentation for resnet 

## The main difference between my implementation and pyTorch official implementation is that in pytorch they included a basic block in which i build using regular convs.2d instead of building it using class,i thought it doesn't really need that as its so simple to just write it Sequentially 

## The expansion factor  
The expansion factor in ResNets refers to the increase in the number of channels (also known as the width) in a convolutional block compared to the number of channels in its input. This expansion is controlled by a hyperparameter, often denoted as "expansion." It is usually associated with the Bottleneck blocks in ResNet architectures.

Let's take a closer look at the Bottleneck block as an example. The original ResNet introduced the Bottleneck architecture to reduce computational complexity while maintaining high expressiveness. A Bottleneck block consists of three convolutional layers:

1. 1x1 convolution to reduce the number of channels (dimensionality reduction)
2. 3x3 convolution that captures spatial information
3. another 1x1 convolution to increase the number of channels back to the original (or expanded) dimension.

The expansion factor comes into play with the second 1x1 convolution in the Bottleneck block. The purpose of expanding and then compressing the channels is to allow the network to learn more complex features without significantly increasing computational cost. By increasing the channels temporarily, the model gains additional capacity to represent complex relationships between features.

Mathematically, if the input to the Bottleneck block has in_channels channels, the second 1x1 convolution (before the 3x3 convolution) increases the channels to expansion * in_channels. The output of the Bottleneck block will then have expansion * in_channels channels.

## Shortcuts with Expansion

For example, consider a basic residual block with an expansion factor of 4. If the input has, say, 64 channels, the convolutional path may increase the number of channels to 256 before merging it with the identity path. This expansion can help the network capture more complex features and representations, allowing it to learn richer and more expressive representations of the data.



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
