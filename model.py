import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 클래스별로 수량 체크하고, 배치 클래스 수량만큼하고(최대한)
# 학습에 넣을때 클래스 하나씩 들어가게 하기
# 모델도 바꿔보기도하고, 데이터 먼저 확인하기

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 1. Data Preparation Module
def prepare_datasets(batch_size):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = datasets.CIFAR10(root='./', train=True, download=True, transform=to_tensor)
    testset = datasets.CIFAR10(root='./', train=False, download=True, transform=to_tensor)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader

# 2. Model Preparation Module
def prepare_model(model):
    if model == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet101(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model

# 3. Training Module
def train_model(model, trainloader, valloader, criterion, optimizer, scheduler, num_epochs, output_path, device=None):
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    with open(output_path, 'w') as f:
        for epoch in range(num_epochs):
            train_loss = 0.0
            train_corrects = 0
            total = 0

            model.train()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                print(labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_corrects += (predicted == labels.data).sum().item()
                total += labels.size(0)
            epoch_train_loss = train_loss / total
            epoch_train_acc = train_corrects / total
            metrics['train_loss'].append(epoch_train_loss)
            metrics['train_accuracy'].append(epoch_train_acc)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_corrects += (predicted == labels.data).sum().item()
                total += labels.size(0)
        epoch_val_loss = val_loss / total
        epoch_val_acc = val_corrects / total
        metrics['val_loss'].append(epoch_val_loss)
        metrics['val_accuracy'].append(epoch_val_acc)
        print(f'Epoch {epoch+1}/{num_epochs} - Train loss: {epoch_train_loss:.4f}, Train accuracy: {epoch_train_acc:.4f}, Val loss: {epoch_val_loss:.4f}, Val accuracy: {epoch_val_acc:.4f}')
        f.write(f'Epoch {epoch+1} - Train loss: {metrics["train_loss"][epoch]:.4f}, Train accuracy: {metrics["train_accuracy"][epoch]:.4f}, Val loss: {metrics["val_loss"][epoch]:.4f}, Val accuracy: {metrics["val_accuracy"][epoch]:.4f}\n')

        scheduler.step()

    return model
