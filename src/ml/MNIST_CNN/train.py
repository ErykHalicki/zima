import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# roughly following https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

MNIST_PATH = '/home/eryk/Documents/projects/zima/src/ml/MNIST_CNN'
#TRAINING PARAMETERS
batch_size = 64

main_device = torch.device("cpu")
if torch.backends.mps.is_available():
    main_device = torch.device("mps")
if torch.cuda.is_available():
    main_device = torch.device("cuda")

# transform function turns PIL image into a tensor
PIL_to_Tensor = transforms.ToTensor()

mnist_train_dataset = torchvision.datasets.MNIST(MNIST_PATH, train=True, transform = PIL_to_Tensor)
mnist_train_dataloader = torch.utils.data.DataLoader(mnist_train_dataset, 
                                                     batch_size=batch_size, 
                                                     shuffle=True, num_workers=6)

mnist_test_dataset = torchvision.datasets.MNIST(MNIST_PATH, train=False, transform = PIL_to_Tensor)
mnist_test_dataloader = torch.utils.data.DataLoader(mnist_test_dataset, 
                                                     batch_size=batch_size, 
                                                     shuffle=False, num_workers=6)

training_iterator = iter(mnist_train_dataloader)

images, labels = next(training_iterator)

def imshow(img):
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.show()

print(labels)
imshow(torchvision.utils.make_grid(images))

class SmallConvNet(nn.Module):
    def __init__(self, 
                 input_image_size=28, 
                 convolution_kernel_width=5, 
                 convolution_kernel_output_dim=10,
                 max_pool_width=2):
        super().__init__()
        self.image_size = input_image_size
        self.convolution_layer = nn.Conv2d(1, 
                                           convolution_kernel_output_dim, 
                                           convolution_kernel_width) # 1 input (grayscale),n output layers, nxn kernel

        convolution_layer_output_size = (input_image_size - convolution_kernel_width) + 1 
        self.max_pool = nn.MaxPool2d(max_pool_width, max_pool_width) #2x2 maxpool, for dim reduction (and nonlinearity?)
        
        self.linear_layer = nn.Linear(int(convolution_layer_output_size / max_pool_width)**2 * convolution_kernel_output_dim, 100) #classifier
        self.classifier = nn.Linear(100, 10)

    def forward(self, batch):
        if batch.size()[2] != self.image_size or batch.size()[2] != self.image_size:
            raise Exception("Size of passed tensor does not match intended image size!")
        result = self.max_pool(F.relu(self.convolution_layer(batch)))
        result = torch.flatten(result, start_dim=1)
        result = F.relu(self.linear_layer(result))
        result = self.classifier(result)
        return result

small_conv_net = SmallConvNet().to(main_device)
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(small_conv_net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    running_loss = 0.0
    for train_batch_index, train_batch_tuple in enumerate(mnist_train_dataloader):
        images, labels = train_batch_tuple
        images = images.to(main_device)
        labels = labels.to(main_device)
        optimizer.zero_grad()

        logits = small_conv_net(images)
        loss = cross_entropy_loss(logits, labels)
        # probability calculated via softmax = exp(correct class prediction) / sum(exp(all class predictions))
        # -ln(p_correct_class) is cross entropy loss
        # -ln(1) = 0 (correct) -> -ln(0) approaches inf
        loss.backward() # calculate gradient of each layer w.r.t loss
        optimizer.step() # do W = W - grad*lr
        running_loss += loss.item() #extract numeric value of the loss

        if train_batch_index % 100 == 99:    # print every 100 mini-batches
            print(f'[epoch: {epoch + 1:3d}, batch: {train_batch_index + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    total_correct_predictions = total_predictions = 0    
    with torch.no_grad(): #dont calculate gradients to speed up inference
        for test_batch_index, test_batch_tuple in enumerate(mnist_test_dataloader):
            images, labels = test_batch_tuple
            images = images.to(main_device)
            labels = labels.to(main_device)

            logits = small_conv_net(images)
            max_logits, predictions = torch.max(logits, dim=1)
            # getting max along dim 1, since dim 0 would return the sample that had highest confidence in the batch (for each class), not the highest confidence class in each sample
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    total_correct_predictions += 1
                total_predictions += 1

        test_set_accuracy = total_correct_predictions / total_predictions * 100.0
        print(f"Accuracy after epoch {epoch+1}: {test_set_accuracy}%")

MODEL_SAVE_PATH = './simple_mnist_conv_net.pt'
torch.save(small_conv_net.state_dict(), MODEL_SAVE_PATH)

small_conv_net = SmallConvNet().to(main_device)
small_conv_net.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
# re load the model for practice



    
