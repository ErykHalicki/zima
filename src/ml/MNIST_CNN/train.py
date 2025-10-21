import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

MNIST_PATH = '/Users/erykhalicki/Documents/projects/current/zima/src/ml/MNIST_CNN/'
#TRAINING PARAMETERS
batch_size = 16

main_device = torch.device("cpu")
if torch.backends.mps.is_available():
    main_device = torch.device("mps")

# transform function turns PIL image into a tensor
PIL_to_Tensor = transforms.ToTensor()

mnist_train_dataset = torchvision.datasets.MNIST(MNIST_PATH, train=True, transform = PIL_to_Tensor)
mnist_train_dataloader = torch.utils.data.DataLoader(mnist_train_dataset, 
                                                     batch_size=batch_size, 
                                                     shuffle=True, num_workers=0)

mnist_test_dataset = torchvision.datasets.MNIST(MNIST_PATH, train=False, transform = PIL_to_Tensor)
mnist_test_dataloader = torch.utils.data.DataLoader(mnist_test_dataset, 
                                                     batch_size=batch_size, 
                                                     shuffle=False, num_workers=0)

training_iterator = iter(mnist_train_dataloader)

images, labels = next(training_iterator)

def imshow(img):
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.show()

print(labels)
imshow(torchvision.utils.make_grid(images))

class SimpleConvNet(nn.Module):
    def __init__(self, 
                 input_image_size=28, 
                 convolution_kernel_width=3, 
                 convolution_kernel_output_dim=3):
        super.__init__()
        self.input_size = input_image_size
        self.convolution_layer = nn.Conv2d(1, 
                                           convolution_kernel_output_dim, 
                                           convolution_kernel_width) # 1 input (grayscale),n output layers, nxn kernel
        convolution_layer_output_size = (self.input_size - ) + 1 
        #calculate output size, of pool and conv, then plug into linear input
        self.max_pool_1 = nn.MaxPool2d(2,2) #2x2 maxpool
        
        self.linear_layer_1 = nn.Linear(100, 10) #classifier
