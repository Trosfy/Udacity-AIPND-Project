import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type = str, default = 'flowers/', help = 'path to train data', required=True)
    parser.add_argument('--save_dir', type = str, default = './', help = 'path to save checkpoint')
    parser.add_argument('--arch', type = str, default = 'densenet161', help = 'model of the NN architecture, valid values = densenet161, vgg13')
    parser.add_argument('--learning_rate', type = float, default = 0.003, help = 'learning rate of the NN')
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'size of the hidden units used')
    parser.add_argument('--dropout', type = float, default = 0.5, help = 'learning rate of the NN')
    parser.add_argument('--epochs', type = int, default = 5, help = 'number of loops to train the NN')
    parser.add_argument('--gpu', action='store_true', help = 'enable gpu')
    
    
    return parser.parse_args()

def train(data_dir, save_dir, arch, learning_rate, hidden_units, dropout, epochs, gpu_enabled):
    # set device
    if gpu_enabled:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
        
    
    
    # differentiate train, valid, test directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # load data
    train_transforms = transforms.Compose([transforms.RandomRotation(40), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                           
    # load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    

    # define data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    print("Data loaded! Building NN...")

    # get pretrained model and set classifier
    input_units = 0
    output_units = len(train_dataset.class_to_idx)
    if arch == "densenet161":
        model = models.densenet161(pretrained=True)
        input_units = 2208
        classifier = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, output_units),
            nn.LogSoftmax(dim=1)
        )
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        input_units = 25088
        classifier = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, output_units),
            nn.LogSoftmax(dim=1)
        )
        print(model)
    else:
        print("Model is not in the option!")
        return
    # freeze features
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier                                    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
                                           
    # set params for printing
    iterator = 0
    running_loss = 0
    print_on = 5
    print("Starting to train...")
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            iterator += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            log_probs = model.forward(inputs)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if iterator % print_on == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log_probs = model.forward(inputs)
                        valid_batch_loss = criterion(log_probs, labels)

                        valid_loss += valid_batch_loss.item()

                        probs = torch.exp(log_probs)
                        top_p, top_class = probs.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print("Epoch {}/{}... Train loss: {}... Validation loss: {}... Validation accuracy: {}...".format(epoch+1, epochs, running_loss/print_on, valid_loss/len(valid_loader), accuracy/len(valid_loader)))
                
                running_loss = 0
                model.train()
                                           
    # return model to cpu if it was in gpu so its state would change to cpu 
    model.to('cpu')
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {
        'class_to_idx' : model.class_to_idx,
        'arch': arch,
        'input' : input_units,
        'output' : output_units,
        'hidden' : hidden_units,
        'dropout' : dropout,
        'classifier' : model.classifier.state_dict()
    }
    torch.save(checkpoint, save_dir)

def main():
    input_arg = get_input_args()
    print(input_arg)
    
    # set parameters
    data_dir = input_arg.data_dir
    save_dir = input_arg.save_dir
    arch = input_arg.arch
    learning_rate = input_arg.learning_rate
    hidden_units = input_arg.hidden_units
    dropout = input_arg.dropout
    epochs = input_arg.epochs
    gpu_enabled = input_arg.gpu
    
    train(data_dir, save_dir, arch, learning_rate, hidden_units, dropout, epochs, gpu_enabled)

if __name__ == "__main__":
    main()
    