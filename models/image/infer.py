import os
import argparse
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

DEFAULT_MODEL = "best_resnet18.pth"


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes", None)
    # создаём модель с нужным количеством выходов
    model = models.resnet18(pretrained=False)
    num_classes = len(classes) if classes is not None else None
    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, classes


def predict(model, classes, image_paths, device):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    results = []
    with torch.no_grad():
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            out = model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1)
            topk = torch.topk(probs, k=min(5, probs.size(1)))
            top_indices = topk.indices.squeeze(0).cpu().tolist()
            top_vals = topk.values.squeeze(0).cpu().tolist()
            top = [
                (classes[i], float(top_vals[idx])) for idx, i in enumerate(top_indices)
            ]
            results.append({"image": p, "predictions": top})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--images", nargs="+", required=True, help="paths to images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.model, device)
    res = predict(model, classes, args.images, device)

    for r in res:
        print(f"\nImage: {r['image']}")
        for label, prob in r["predictions"]:
            print(f"  {label}: {prob:.3f}")
