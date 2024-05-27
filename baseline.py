import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import numpy as np
import math
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#increas batch
BATCH_SIZE = 32
#G1 = 10 ,G2 = 16
NUM_EPOCHS = 21
#decrease 
LEARNING_RATE = 0.0001

class RLM(Dataset):
    
    def __init__(self, game_type):
        if game_type == 'game1':
            prefix = './Game1/'
        elif game_type == 'game2':
            prefix = './Game2/'
        elif game_type == 'full':
            prefix = './Full/'
        else:
            raise ValueError("Invalid game_type. Must be one of: 'game1', 'game2', or 'full'")
        
        with open(prefix + 'label.json', 'r') as file:
            self.image_label = torch.tensor(json.load(file))
        with open(prefix + 'baseline.json', 'r') as file:
            data = torch.tensor(json.load(file),dtype=torch.float32)
            self.base = data.permute(0, 1, 4, 2, 3)

    def __getitem__(self,index):
        return self.base[index], self.image_label[index]

    def __len__(self):
        return len(self.image_label)


 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
                    #out_channels too big add conv between, circular padding mode
                    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=3, padding=5),
                    #nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=3, padding=3),
                    #nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2),
                    #nn.BatchNorm2d(64),
                    )
        conv_output_size = self._get_conv_output_size()
        print(conv_output_size)
        self.fc1 = nn.Linear(conv_output_size, 1024)  # Adjust the input size
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)



    def _get_conv_output_size(self):
        # Dummy forward pass to get the output size of the last convolutional layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 450, 355)
            conv_output = self.layer(dummy_input)
            conv_output_size = (conv_output.view(conv_output.size(0), -1).size(1)) *2

        return conv_output_size
    
    def forward(self, base):

        first_frame = self.layer(base[:,0])
        last_frame = self.layer(base[:,1])
        print(base.shape)
        x = torch.cat((first_frame, last_frame), dim=1)

        # Flatten the output for fully connected layers
        x = x.reshape(x.size(0), -1)

        # Apply fully connected layers with activation functions, apply it to each layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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
        for i, (base,labels) in enumerate(train_loader):
            base, labels = base.to(device), labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            outputs = model(base)

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
            for i, (base,labels) in enumerate(val_loader):
                base, labels = base.to(device), labels.to(device)
                outputs = model(base)
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

    predicted_labels = []
    true_labels = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, (base, labels) in enumerate(test_loader):
            base, labels = base.to(device), labels.to(device)
            outputs = model(base)
            # Calculate accuracy
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print(f'Accuracy of the network on the test set: {100 * correct // total} %')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)
    plot_conf_matrix(conf_matrix)

    accuracy, precision, recall, f1 = compute_metrics(true_labels, predicted_labels)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    #torch.save(model.state_dict(), 'V_model_game2.pt')

def test_on_Game(game_type,model_name):
    dataset = RLM(game_type)
    test_loader = DataLoader(dataset,batch_size=1,shuffle=False)
    model = CNN()
    model.load_state_dict(torch.load(model_name))
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for i, (base, labels) in enumerate(test_loader):
            base, labels = base.to(device), labels.to(device)
            outputs = model(base)
            # Calculate accuracy
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print(f'Accuracy of the network on the test Game: {100 * correct // total} %')
    
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)
    plot_conf_matrix(conf_matrix)

    accuracy, precision, recall, f1 = compute_metrics(true_labels, predicted_labels)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

def compute_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    return accuracy, precision, recall, f1

def plot_conf_matrix(conf_matrix):
    
    plt.figure(figsize=(4, 3))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()

def main():
    dataset = RLM(game_type="full")

    train_size = int(0.8 * len(dataset))
    val_size = int(0.10 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Use random_split to split the dataset
    torch.manual_seed(42)  
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], )
    train_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size = BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size = 1, shuffle=False)
    print(len(dataset))
    run(train_loader,val_loader,test_loader,device)
    #test_on_Game(game_type="game1",model_name='V_model_game2.pt')


    
    
if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    main()


