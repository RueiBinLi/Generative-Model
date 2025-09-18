import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torch import optim
from model import *
from utils import *

def data_preprocessing():
    IMG_SIZE = 64
    BATCH_SIZE = 8

    transforms_pipeline = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), # normalize to [-1,1]
    ])

    data_folder_path = './data'
    data = torchvision.datasets.ImageFolder(root=data_folder_path, transform=transforms_pipeline)
    print(f'Found {len(data)} images in the dataset')

    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    return dataloader

def train(epochs=20, model_path=None, dataloader=None, model=None, diffusion=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse = nn.MSELoss()

    l = len(dataloader)
    loss_all = []

    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0.0
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad() # reset the gradients
            loss.backward() # calculate the gradients
            optimizer.step() # update all the models parameter
            epoch_loss += loss.item() # calculate the epoch loss
            pbar.set_postfix(MSE=loss.item()) # display the loss of each batch

    
        avg_loss = epoch_loss / l
        loss_all.append(avg_loss)

    return loss_all

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(c_in=3, c_out=3, img_size=64, time_dim=256, device=device)
diffusion = Diffusion(steps=500, beta_start=1e-4, beta_end=2e-2, img_size=64, device=device)
dataloader = data_preprocessing()
loss_all = train(epochs=20, dataloader=dataloader, model=model, diffusion=diffusion)

plot_loss(loss_all)