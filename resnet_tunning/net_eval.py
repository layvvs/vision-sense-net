import torch
from torchvision import transforms
from PIL import Image

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

prep = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet50 = torch.load('../models/fine_tuned_resnet.pt', map_location=torch.device(device))

resnet50 = resnet50.to(device)

resnet50.eval()

image = Image.open('angry_face.jpg').convert('RGB')
image = prep(image)
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = resnet50(image)
    _, predicted = torch.max(outputs, 1)

print(predicted.item())
