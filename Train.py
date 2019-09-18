import torch
from torch import optim, nn
import ClothesClassifier

def train(net, trainloader, testloader, device = "cpu", epochs= 5, batch_size= 64, learning_rate= 0.001):
    train_loss = 0
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    net.to(device)

    for i in range(epochs):
        net.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            test_output = net(images)
            loss = criterion(test_output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss
        else:
            net.eval()

            with torch.no_grad():
                test_loss, accuracy = validation(net, testloader, device, criterion)
            
            print("Epoch: {}/{} ".format(i+1, epochs),
                      "Training Loss: {:.3f} ".format(train_loss/len(trainloader)),
                      "Test Loss: {:.3f} ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy))
        

def validation(net, testloader, device, criterion):
    test_loss = 0
    accuracy = 0
    
    net.to(device)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        test_output = net(images)
        test_loss += criterion(test_output, labels).item()

        probs = torch.exp(test_output)
        equality = (labels.data == probs.max(1)[1])
        accuracy = equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy
