import matplotlib
matplotlib.use('Agg') # <-- Add this line

import torch
import torchvision
import matplotlib.pyplot as plt
from model import *
from utils import *

if __name__ == "__main__":
    # IMG_SIZE = 64
    # BATCH_SIZE = 128
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)

    # data_folder_path = '/home/robby/master/1st_year/Generative_Model/HW1_DDPM/data'
    # num_samples = 1

    # data = torchvision.datasets.ImageFolder(root=data_folder_path)
    # image, label = data[0]
    
    # plt.figure(figsize=(8, 8))
    
    # plt.imshow(image)
    # plt.savefig('output.png')
    # print("Plot saved to output.png")

    visualize_noising_process()