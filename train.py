import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from model import prepare_datasets, prepare_model, train_model

# Function to load parameters from JSON
def load_params(json_path):
    with open(json_path, encoding='UTF-8') as f:
        print(f)
        params = json.load(f)
    return params

model_params = ['model.json', 'model_2.json', 'model_3.json']
for m in model_params:
    parser = argparse.ArgumentParser(description='Deep Learning Training Parameters')
    parser.add_argument('--config', type=str, default='C:/Users/kimhyesu/Documents/GitHub/resnet_cifar10/'+m, help='Path to the config file')
    args = parser.parse_args()

    params = load_params(args.config)

    # Main execution
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = prepare_datasets(batch_size=params['batch_size'])
    model = prepare_model(model=params['model']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    model = train_model(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs=params['num_epochs'], output_path=params["output_path"], device=device)
    print(f'train finish {m}')
