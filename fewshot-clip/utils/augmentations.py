import random
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFilter
import torch

def augment_image(image):
    """
    Apply random augmentations to the image, including color adjustments, rotation, flipping, 
    blurring, adding Gaussian noise, and masking a small patch.
    """
    # Define a list of possible augmentations to apply randomly
    augmentation_transforms = [
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Color jitter
        transforms.RandomRotation(30),  # Random rotation
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip with a probability of 0.5
        transforms.RandomVerticalFlip(p=0.5),  # Vertical flip with a probability of 0.5
        transforms.GaussianBlur(kernel_size=(5, 9)),  # Gaussian blurring
    ]

    # Randomly select and apply one augmentation from the list
    transform = random.choice(augmentation_transforms)
    image = transform(image)
    
    # Add Gaussian noise by converting to tensor, applying noise, and converting back to PIL image
    image = transforms.ToTensor()(image)  # Convert image to tensor
    image = AddGaussianNoise(0, 0.1)(image)  # Add Gaussian noise
    image = transforms.ToPILImage()(image)  # Convert back to PIL image

    # Apply a random patch mask to simulate an anomaly
    image = mask_random_patch(image)
    
    return image

class AddGaussianNoise:
    """
    Class to add Gaussian noise to a tensor.
    """
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Generate Gaussian noise with the same shape as the input tensor
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise  # Add noise to the tensor
        return noisy_tensor.clamp(0, 1)  # Clamp values to keep them in the range [0, 1]

def mask_random_patch(image, mask_size_ratio=0.2):
    """
    Apply a random white square patch to the image to create an anomaly effect.
    :param image: Input PIL image
    :param mask_size_ratio: Ratio of the patch size relative to the image size
    :return: PIL image with the masked patch applied
    """
    # Calculate dimensions for the mask patch
    width, height = image.size
    mask_size = int(min(width, height) * mask_size_ratio)
    x_start = random.randint(0, width - mask_size)
    y_start = random.randint(0, height - mask_size)
    
    # Apply a white square patch on the image
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle([x_start, y_start, x_start + mask_size, y_start + mask_size], fill="white")
    
    return image
