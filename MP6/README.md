# Generative Adversarial Network (GAN)

## Part 1
Trained a classifier on the CIFAR10 dataset without the generator. Implemented a 8 layer Convolution Network for training an image classifier without the generator. Achieved a Test Accuracy of 87%

## Part 2
Trained a Generative Adversarial Network (GAN) on the CIFAR10 dataset. Implemented a 8 layer Generative Adversarial Network for generating
images. Used Layer Normalization to normalize the input tensors and Leaky ReLU for activation for the discriminator network and Batch Normalization and ReLU for the Generator network. Extracted 196 features and trained the model for over 200 epochs. Achieved a Test Accuracy of 83%

## Images Generated 

The following are the images generated during training

![Capture1](https://user-images.githubusercontent.com/38511470/56534541-8d44d200-651f-11e9-9c08-d4de67b957f0.PNG)
![Capture2](https://user-images.githubusercontent.com/38511470/56534570-99309400-651f-11e9-8501-6703d63d7946.PNG)
![Capture3](https://user-images.githubusercontent.com/38511470/56534590-a057a200-651f-11e9-90c1-5a3a5580bcee.PNG)
![Capture4](https://user-images.githubusercontent.com/38511470/56534608-a8afdd00-651f-11e9-895d-311b8f776d53.PNG)
![Capture5](https://user-images.githubusercontent.com/38511470/56534626-b1081800-651f-11e9-8cb7-182d7a79580d.PNG)

## Perturbing Real Images

Similar to the feature visualization technique, real images correctly classified by a network can be imperceptibly altered to produce a highly confident incorrect output. Backpropagate the error for an alternative label to real image, use the sign of the gradient (-1 or 1), multiply this by 0.0078, and add it to the original image. A value of 0.0078 is 1⁄255 which is how colors are discretized for storing a digital image. Changing the value of the input image by less than this is imperceptible to humans but convolution networks can be highly sensitive to these changes. The new images are then evaluated and high accuracy is reported on the real images and essentially random guessing is done for the altered images

![Capture6](https://user-images.githubusercontent.com/38511470/56535349-24f6f000-6521-11e9-81d2-ae12ebf62af0.PNG)
![Capture7](https://user-images.githubusercontent.com/38511470/56535370-30e2b200-6521-11e9-8bc4-b10e4f513168.PNG)
![Capture8](https://user-images.githubusercontent.com/38511470/56535379-36d89300-6521-11e9-8f56-875368c10495.PNG)


## Synthetic Images Maximizing Classification Output

It is possible to visualize the features and see how they become more interesting and complex as the network deepens. One method is to input a noisy initialized image and calculate the gradient of a particular feature within the network with respect to the input. This gradient indicates how changes in the input image affect the magnitude of the feature. This process can be repeated with the image being adjusted slightly each time with the gradient. The final image is then a synthesized image causing a very high activation of that particular feature. The loss function for each class will be used to repeatedly modify an image such that it maximizes

![Capture9](https://user-images.githubusercontent.com/38511470/56535468-5ff92380-6521-11e9-8581-10c79f1e9ab3.PNG)

## Synthetic Features Maximizing Features at Various Layers

Similar to the section above, this will create synthetic images maximizing a particular feature instead of a particular class. You should notice the size of the feature gets larger for features later in the network
