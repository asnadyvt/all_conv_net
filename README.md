# ROLLING IN THE DEEP
## Implementation of "The All Convolutional Net"

### TODO
* Weight decay, lambda = 0.001
* SGD with momentum = 0.9
* Learning rate gamma = [0.25, 0.1, 0.05, 0.01], 0.1 for now
* Adapt learning rate at [200, 250, 300] by multiplying by 0.1
* 350 epochs
* Train all NNs with CIFAR-10
* Visualize first 3 layers
* If there is time, Visualize layers 6 and 9 and look into guided
  backprop


### CIFAR-10 Classification Error without data augmentation

| Model           | Error (%) |  Time (mins) |
| ---------------| -----------| ----- |
| Model A         |     15.16% | |
| Strided-CNN-A   |     18.23% | |
| ConvPool-CNN-A  |     11.44% | |
| ALL-CNN-A       |     12.65% |	642.76 |
| Model B         |     13.52% | |
| Strided-CNN-B   |     15.56% | |
| ConvPool-CNN-B  |     11.84% | 879.25 |
| ALL-CNN-B       |     11.84% | |
| Model C         |     error% | |
| Strided-CNN-C   |     error% | |
| ConvPool-CNN-C  |     error% | |
| ALL-CNN-C       |     error% | |

