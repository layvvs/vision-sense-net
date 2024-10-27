import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
from tempfile import TemporaryDirectory
from tqdm import tqdm
from datetime import datetime 


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = './data_analyze/processed_data'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == 'cuda':
    torch.cuda.set_device(0)

print('Device:', device)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
}


data_dir = DATA_DIR


data_sets = {
    data_type: datasets.ImageFolder(os.path.join(data_dir, data_type), data_transforms[data_type])
    for data_type in ['train', 'test']
}


data_set_loaders = {
    data_type: torch.utils.data.DataLoader(
        data_sets[data_type],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )
    for data_type in ['train', 'test']
}


data_set_sizes = {data_type: len(data_sets[data_type]) for data_type in ['train', 'test']}


data_set_classes = data_sets['train'].classes


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in tqdm(range(num_epochs)):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(data_set_loaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / data_set_sizes[phase]
                epoch_acc = running_corrects.double() / data_set_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                with open(f'metrics-{phase}.txt', 'a') as _f:
                    _f.write(f'epoch #{epoch + 1}/{num_epochs} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best test Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model

if __name__ == '__main__':
    resnet50 = models.resnet50(pretrained=True)

    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()

    criterion.to(device)
    resnet50.to(device)
    optimizer_ft = optim.RMSprop(resnet50.parameters(), lr=LEARNING_RATE)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    resnet50 = train_model(
        resnet50,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=NUM_EPOCHS)
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    torch.save(resnet50, f'fine_tuned_resnet_{dt_string}.pt')
