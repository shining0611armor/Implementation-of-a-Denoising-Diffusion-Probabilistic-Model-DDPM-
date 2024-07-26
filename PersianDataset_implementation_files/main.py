# main.py

#--------------dataset-------------------------------
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Dataset import get_data_loaders
import utils
from utils import get_images_by_class, save_plot_images



batch_size = 128

train_loader, test_loader = get_data_loaders(batch_size)

train_images = get_images_by_class(train_loader)
test_images = get_images_by_class(test_loader)

save_plot_images(train_images, "Train Dataset Images by Class", "train_images.png")
save_plot_images(test_images, "Test Dataset Images by Class", "test_images.png")


print(next(iter(train_loader))[0].shape)



#-------------------------------------------------

