import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from flask import Flask, render_template, request
from PIL import Image
import os
from fish_info import fish_data

app = Flask(__name__)

# make upload folder
os.makedirs("static/uploads", exist_ok=True)

# load class names from dataset
classes = [
    "anchovy",
    "barracuda",
    "catfish",
    "croaker",
    "grouper",
    "karimeen",
    "mackerel",
    "milkfish",
    "pomfret",
    "ribbon_fish",
    "sardine",
    "seer_fish",
    "shark",
    "silver_belly",
    "sole_fish",
    "stingray",
    "threadfin_bream",
    "tilapia",
    "trevally",
    "tuna"
]

# load trained model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))

model.load_state_dict(torch.load("fish_model.pth", map_location="cpu"))
model.eval()

# image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# prediction function
def predict_image(path):

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    probs = F.softmax(output, dim=1)

    confidence, predicted = torch.max(probs,1)

    fish_name = classes[predicted.item()]
    conf = round(confidence.item()*100,2)

    return fish_name, conf


@app.route("/", methods=["GET","POST"])
def index():

    prediction=None
    confidence=None
    img_path=None
    info=None

    if request.method=="POST":

        file=request.files["image"]

        img_path=os.path.join("static/uploads",file.filename)

        file.save(img_path)

        prediction,confidence=predict_image(img_path)

        if prediction in fish_data:
            info=fish_data[prediction]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path,
        info=info
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)