import torch 
from torch import nn


class VariationalAutoencoder(nn.Module):


    def __init__(self, channels:int=1, latent_dim:int=2):

        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            self.conv_block(channels, 16, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            self.conv_block(16, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            self.conv_block(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            self.conv_block(64, 128, kernel_size=3, stride=1, padding=1)  # 7x7 -> 7x7
        )

        encoder_out_size = 7*7*128
        self.linear_1 = nn.Linear(in_features=encoder_out_size, out_features=latent_dim*2)
        self.linear_2 = nn.Linear(in_features=latent_dim, out_features=encoder_out_size)

        self.decoder = nn.Sequential(
            self.conv_block(128, 64, kernel_size=3, stride=1, padding=1, encoder=False),  # 7x7 -> 7x7
            self.conv_block(64, 32, kernel_size=4, stride=2, padding=1, encoder=False),  # 7x7 -> 14x14
            self.conv_block(32, 16, kernel_size=4, stride=2, padding=1, encoder=False),  # 14x14 -> 28x28
            nn.ConvTranspose2d(16, channels, kernel_size=3, stride=1, padding=1),  # Final layer without activation
            nn.Tanh()
        )
        

    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, encoder=True):

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding) if encoder else nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        )
    

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + (eps * std)
        return z
    

    def forward(self, x):

        encoded = self.encoder(x)
        B, C, H, W = encoded.shape
        encoded_flat = self.linear_1(encoded.view(B, -1))

        mean = encoded_flat[:, :encoded_flat.shape[1]//2]
        log_var = encoded_flat[:, encoded_flat.shape[1]//2:]
        latent_vector = self.reparameterize(mean, log_var)

        decoder_inp = self.linear_2(latent_vector).view(B, C, H, W)
        decoded = self.decoder(decoder_inp)

        return decoded, mean, log_var
        


class VAELoss(nn.Module):

    def __init__(self, alpha=0.001):
        super(VAELoss, self).__init__()
        self.alpha = alpha
        self.mse_loss_fn = nn.MSELoss()
    
    def forward(self, original, reconstruction, mean, log_var):
        reconstruction_loss = 0.5 * self.mse_loss_fn(reconstruction, original)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / original.shape[0]
        loss = reconstruction_loss + (self.alpha * kl_loss)
        return loss, reconstruction_loss, kl_loss