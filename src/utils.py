import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class LogoEncoder(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(LogoEncoder, self).__init__()
        self.model = torchvision.models.resnet50(weights=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, p=2, dim=1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(file):
    image = Image.open(file).convert('RGB')
    return image

def preprocess_image(image: Image.Image):
    device = torch.device('cpu')
    return transform(image).unsqueeze(0).to(device)