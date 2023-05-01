from torchvision import transforms
from src.data.components.custom_transforms import RandomText, RandomCircle


def train(font_path):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            RandomText(font_path),
            RandomCircle(),
            transforms.RandomRotation((-25, 25)),
            transforms.RandomPerspective(0.2, 0.5),
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms

def test():
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return test_transforms