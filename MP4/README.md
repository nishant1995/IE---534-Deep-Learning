# ResNet

## Part 1
Built a Residual Network specified in the given figure to detect the classes in the CIFAR100 Dataset

![Alt text](https://user-images.githubusercontent.com/38511470/56448853-93953d00-62d8-11e9-9ec3-9722fc3e73c3.png)

![image](https://user-images.githubusercontent.com/38511470/56448935-7dd44780-62d9-11e9-9f9f-55ea7f7f8648.png)

Implemented a 26 layer Residual Network and trained the network with a dropout probability of 20%. Used RandomCrop() and RandomHorizontalFlip() as the data augmentation techniques and trained the model over 100 epochs using ADAM Optimizer. Used a batch size of 100 and a MaxPool layer with kernel size 4 and
stride 2.

### Test Accuracy
Achieved a test accuracy of about 62%

## Part 2
Fine-tuned a pre-trained ResNet-18 model to detect the classes in the CIFAR100 Dataset. Used RandomCrop() and RandomHorizontalFlip() as the data augmentation techniques and trained the model over 25 epochs using ADAM Optimizer with a learning rate of 0.001

### Test Accuracy
Achieved a test accuracy of about 74%




