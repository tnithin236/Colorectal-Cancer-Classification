from PIL import Image
from torchvision import transforms
from model_loader import MEAN, STD


def get_transform(img_size=224):
    # identical to eval_transform in the training notebook
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_image(path, img_size=224):
    return get_transform(img_size)(Image.open(path).convert("RGB"))
