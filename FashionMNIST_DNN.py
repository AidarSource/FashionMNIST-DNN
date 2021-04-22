import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Introduce MNIST dataset, and train, test set respectively
# transform: preprocess the data(transform data to Tensor)
trainset = datasets.FashionMNIST("./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
testset = datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

# Slice the dataset into batch(batch_size)
# shuffle: whether shuffling the data first
# pin_memory: speed up the time to transfer data from CPU to GPU
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, pin_memory=True)

"""
# Show the image of dataset
X, y = iter(trainset).next() # pick one batch
print(X.shape) # output -> torch.Size([100, 1, 28, 28]); [batch size, channel, image size]
X = torchvision.utils.make_grid(X) # make image to grid
plt.figure(figsize=(10, 10))
plt.imshow(np.transpose(X, (1, 2, 0))) # to show the image, we need to transpose tensor to numpy array
"""

# Build the Neural Network(fully connected layer)
class Net(nn.Module):
    def __init__(self): # define the structure of Neural Network
        super().__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28*28, 64) # image size: 28*28; define output as 64
        self.fc2 = nn.Linear(64, 64) # the output of previous layer is 64, so input is 64
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) # the output of output layer has to be 10, since MNIST starts from 0~9(10 numbers)

    def forward(self, x): # define how the data x propagate from layers to layers, and return
        x = F.relu(self.fc1(x)) # fully connected layer -> relu activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # log after passing softmax
        # dim=1: do softmax for every elements in each row(output each row as 10 class possibilities of a training data)
        # E.g, if batch size = 100, then x size = 100*10(100 rows, 10 columns)
        return F.log_softmax(x, dim=1)
net = Net() # initial network

# Build Optimizer; train the network
# epochs: decide how many times it has to run the training set
# optimizer.zero_grad(): clear the gradient from aggregating everytime after we update parameters
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train() # set network to training mode
epochs = 10
for epoch in range(epochs):
    for data in trainloader:
        X, y = data # X: training data in batch; y: label
        optimizer.zero_grad()
        predicted = net(X.view(-1, 28*28)) # put the mini-batch training data to Neural Network, and get the predicted labels
        loss = F.nll_loss(predicted, y) # compare the predicted labels with ground-truth labels
        loss.backward() # compute the gradient
        optimizer.step() # optimize the network
    print(f'epoch:{epoch}, loss:{loss}')

# Evaluate Training set, Testing set by trained Neural Network
net.eval() # switch to evaluate mode
# Evaluate the training data
correct = 0
total = 0
with torch.no_grad():
    for data in trainloader:
        X, y = data
        output = net(X.view(-1, 28*28))
        correct += (torch.argmax(output, dim=1) == y).sum().item() # calculate the accuracy of this batch; item(): transform Tensor into Python to compute
        total += y.size(0) # total plus batch
print(f'Training data Accuracy: {correct}/{total} = {round(correct/total, 3)}')

# Evaluate the testing data
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        X, y = data
        output = net(X.view(-1, 28*28))
        correct += (torch.argmax(output, dim=1) == y).sum().item()
        total += y.size(0)
print(f'Testing data Accuracy: {correct}/{total} = {round(correct/total, 3)}')