import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import einops
import imageio
from tqdm.auto import tqdm
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))



class DiffusionModel(nn.Module):
    def __init__(self, backward_process_model, beta_start, beta_end, timesteps=1000, device="cuda"):
        super(DiffusionModel, self).__init__()

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.device = device
        self.image_chw = (1, 28, 28)
        self.backward_process_model = backward_process_model

        self.betas = self.get_linear_beta_schedule(beta_start, beta_end, timesteps).to(device)
        # we define alphas variable based on q(xt|x0) formula
        self.alphas = 1.0 - self.betas.to(device)

        # we define alpha_bars variable based on q(xt|x0) formula
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)




    def get_linear_beta_schedule(self, beta_start = 0.0001, beta_end = 0.02, timesteps = 1000):
        return torch.linspace(beta_start, beta_end, timesteps)

    def add_noise(self, x_0, timestep, visualization = False):#As we said time step is including 128 random intiger number between 0 and 1000
        # Make input image more noisy (we can directly skip to the desired step)
        # this function get and specific time step and then it computes the corresponding noise and
        n, c, h, w = x_0.shape
        a_bar = self.alpha_bars[timestep]
        eta = torch.randn(n, c, h, w).to(self.device)
        if visualization ==False:
          noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x_0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        else:
          noisy = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * eta
        return noisy, eta

    def backward(self, x, t):
        return self.backward_process_model(x, t)


    def forward(self, x): # size of x0 is almost torch.Size([128, 1, 28, 28]) exept the last epoch  which is torch.Size([96, 1, 28, 28])
        n = len(x) # n is alomost equal to batch size exept the last (128)
        steps = torch.randint(0, self.timesteps, (n,)).to(self.device) # creating random integer number between 0 and the last time state. creating almost 128 number
        noisy_imgs, noise = self.add_noise(x, steps) # going forward to make noisy
        eta_theta = self.backward_process_model(noisy_imgs, steps.reshape(n, -1))  # we reshape steps to torch.Size([128, 1]) so steps.reshape(n, -1) size is torch.Size([128, 1])
        return noisy_imgs, noise, eta_theta


def sample(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=64, w=64):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, ddpm.timesteps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.timesteps))[::-1]):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)

            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)
    return x



from torch.optim.lr_scheduler import StepLR

def get_loss(noise, noise_pred):

    return torch.mean((noise - noise_pred) ** 2)


def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    #mse = nn.MSELoss()

    best_loss = float("inf")
    n_steps = ddpm.timesteps
    scheduler = StepLR(optim, step_size=10, gamma=0.5)
    epoch_losses = []


    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for batch in tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500"):
                # Loading data
                x0 = batch[0].to(device)
                a, eta, eta_theta = ddpm(x0)
                loss = get_loss(eta_theta, eta)
                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += loss.item() * len(x0) / len(loader.dataset)


        # Step the scheduler every epoch
        scheduler.step()

        # Record the epoch loss
        epoch_losses.append(epoch_loss)

        lr = scheduler.get_last_lr()[0]
        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}, Learning Rate: {lr:.6f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> saving best model"
        print(log_string)


    # Plotting the MSE loss for each epoch
    plt.figure()
    plt.plot(range(1, n_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title("MSE Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

import warnings
warnings.filterwarnings("ignore")


def evaluate(diffusion_model, test_loader, device="cuda"):
    diffusion_model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            n = len(data)
            n_steps = torch.randint(0, diffusion_model.timesteps, (n,)).to(device)
            noisy_imgs, noise = diffusion_model.add_noise(data, n_steps)
            eta_theta = diffusion_model.backward_process_model(noisy_imgs, n_steps.reshape(n, -1))

            loss = get_loss(noise, eta_theta)
            total_loss += loss.item() * n
            total_samples += n

    average_loss = total_loss / total_samples
    return average_loss

