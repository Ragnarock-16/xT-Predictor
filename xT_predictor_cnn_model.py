import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import numpy as np
import math
import json
import matplotlib.pyplot as plt

#increas batch
BATCH_SIZE = 32
NUM_EPOCHS = 250
#decrease 
LEARNING_RATE = 0.0001


class RLM(Dataset):
    
    def __init__(self):
        with open('/Users/nour/Documents/M1/TER/Cnn_model/Game2/label.json', 'r') as file:
            self.image_label = torch.tensor(json.load(file))
        with open('/Users/nour/Documents/M1/TER/Cnn_model/Game2/home_1.json', 'r') as file:
            self.home_1 = torch.tensor(json.load(file))
        with open('/Users/nour/Documents/M1/TER/Cnn_model/Game2/away_1.json', 'r') as file:
            self.away_1 = torch.tensor(json.load(file))
        with open('/Users/nour/Documents/M1/TER/Cnn_model/Game2/home_2.json', 'r') as file:
            self.home_2 = torch.tensor(json.load(file))
        with open('/Users/nour/Documents/M1/TER/Cnn_model/Game2/away_2.json', 'r') as file:
            self.away_2 = torch.tensor(json.load(file))

    def __getitem__(self,index):
        return self.away_1[index], self.away_2[index], self.home_1[index], self.home_2[index], self.image_label[index]

    def __len__(self):
        return len(self.image_label)


 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
                    #out_channels too big add conv between, circular padding mode
                    nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, stride=3, padding=5),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=3, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2),
                    )
        conv_output_size = self._get_conv_output_size()
        print(conv_output_size)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128,2)

    def _get_conv_output_size(self):
        # Dummy forward pass to get the output size of the last convolutional layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 15, 120)
            conv_output = self.layer(dummy_input)
            conv_output_size = conv_output.view(conv_output.size(0), -1).size(1)

        return conv_output_size
    
    def forward(self, home_1, home_2, away_1, away_2):
        # Stack input tensors along the channel dimension
        x = torch.stack([home_1, home_2, away_1, away_2], dim=1)
        x = self.layer(x)

        # Flatten the output for fully connected layers
        x = x.reshape(x.size(0), -1)

        # Apply fully connected layers with activation functions, apply it to each layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def run(train_loader, val_loader, test_loader, device):
    model = CNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    #ADAMW, SGD, 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0  
        for i, (away_1, away_2, home_1, home_2, labels) in enumerate(train_loader):
            away_1, away_2, home_1, home_2, labels = away_1.to(device), away_2.to(device), home_1.to(device), home_2.to(device), labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            outputs = model(home_1, home_2, away_1, away_2)

            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step (parameter update)
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
                
        # Validation after each epoch
        model.eval()  
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (away_1, away_2, home_1, home_2, labels) in enumerate(val_loader):
                away_1, away_2, home_1, home_2, labels = away_1.to(device), away_2.to(device), home_1.to(device), home_2.to(device), labels.to(device)
                outputs = model(home_1, home_2, away_1, away_2)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                predicted = torch.argmax(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)  

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print epoch-wise statistics
        print(f'EPOCH [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {100 * correct / total:4f}%')

    print('Training finished.')

    # Plot the loss graph after training is complete
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Test
    model.eval() 
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, (away_1, away_2, home_1, home_2, labels) in enumerate(test_loader):
            away_1, away_2, home_1, home_2, labels = away_1.to(device), away_2.to(device), home_1.to(device), home_2.to(device), labels.to(device)
            outputs = model(home_1, home_2, away_1, away_2)
            # Calculate accuracy
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test set: {100 * correct // total} %')



def main():
    dataset = RLM()

    train_size = int(0.8 * len(dataset))
    val_size = int(0.10 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Use random_split to split the dataset
    torch.manual_seed(42)  
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], )
    train_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size = BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size = 1, shuffle=False)
    run(train_loader,val_loader,test_loader,device)
    
if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    main()


    #Save model every 50 epoch and test