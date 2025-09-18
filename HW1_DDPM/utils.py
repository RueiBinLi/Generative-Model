import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from model import *
from PIL import Image


'''
def load_image(image_path, img_size=64):
    """Load and preprocess an image for diffusion testing"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, img_size, img_size]
    
    return image_tensor

def visualize_noising_process(image_path="HW1_DDPM/data/class_A/test.JPEG"):
    diffusion = Diffusion(steps=1000, img_size=64, device='cpu')

    # Load the real image instead of using random noise
    image = load_image(image_path, img_size=64)
    print(f"Loaded image shape: {image.shape}")

    timesteps = torch.tensor([0, 100, 300, 500, 700, 900])

    fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 3))

    for i, t in enumerate(timesteps):
        if t == 0:
            img_to_show = image[0]
        else:
            noisy_img, _ = diffusion.noise_images(image, torch.tensor([t]))
            img_to_show = noisy_img[0]

        # Normalize to [0,1] for display
        img_display = (img_to_show + 1) / 2
        img_display = torch.clamp(img_display, 0, 1)
        
        axes[i].imshow(img_display.permute(1, 2, 0))
        axes[i].set_title(f't={t}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('diffusion_test_image.png', dpi=150, bbox_inches='tight')
    print("Image saved as 'diffusion_test_image.png'")
'''

def plot_loss(loss_all):
    plt.figure(figsize=(6,6))
    epochs_all = np.arange(1, len(loss_all)+1, 1)
    plt.plot(epochs_all, loss_all, marker="", label='loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig('../loss.png', dpi=300, bbox_inches='tight')