import torch
from torchvision import datasets, transforms
import Train
import ClothesClassifier
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

print("Setting up training set and test set from Fashion MNIST datset")
#Transformation used to normalized dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

#Load up testset and training set
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data", download= True, train= False, transform= transform)
testloader = torch.utils.data.DataLoader(testset, batch_size= 64, shuffle= True)

print("Finished setting up subsets. Checking if pretrained model will be used and if we will need to train the model")

#determines if ptretrained model will be used
if len(sys.argv) == 3:
    do_training = True if sys.argv[1] == "True" else False
    model = ClothesClassifier.ClothesClassifier(sys.argv[2])
elif len(sys.argv) <3:
    do_training = True if sys.argv[1] == "True" else False
    model = ClothesClassifier.ClothesClassifier()

#if cuda is available, use cuda to run model
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#trains model and measures time needed to do so
if do_training:
    print("Beginning to train and validate model")
    start_time = time.time()
    Train.train(model, trainloader, testloader, device)
    print("Time taken to train and validate model was: {:.3f} seconds".format(time.time()-start_time))

#get the whole dataset and see random inputs and the model's output for each of these.
dataset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size= 1, shuffle= True)

clothes= ['T-shirt/top',
            'Trouser',
            'Pullover',
            'Dress',
            'Coat',
            'Sandal',
            'Shirt',
            'Sneaker',
            'Bag',
            'Ankle Boot']

model = model.to(device)
model.eval()
continue_input = 'y'

while continue_input == 'y':
    image, label = next(iter(dataloader))
    image = image.to(device)

    output = torch.exp(model(image))
    
    output_cpu = output.to("cpu")
    image = image.to("cpu")
    
    print("Truth is {}".format(clothes[label]), "while model predicted {}".format(clothes[np.argmax(output_cpu.data.numpy())]))
    plt.imshow(image.resize_(1,28,28).numpy().squeeze())
    plt.show()
    
    continue_input = input("continue? type y for yes    ")

#TODO add option to save trained model for future use
