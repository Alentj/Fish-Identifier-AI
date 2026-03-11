import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image

# load class names from dataset
dataset = datasets.ImageFolder("dataset/train")
classes = dataset.classes

# load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))

model.load_state_dict(torch.load("fish_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

img = Image.open("test_fish.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)

_, predicted = torch.max(output,1)

print("Prediction:", classes[predicted.item()])