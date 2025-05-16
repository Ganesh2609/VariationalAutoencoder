import os
import torch 
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from logger import TrainingLogger
from tqdm import tqdm
import cv2


class ModularTrainer:
    """
    A modular trainer for VAE models with comprehensive visualization and logging capabilities.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader, 
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 loss_fn: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 log_path: Optional[str] = './train data/logs/training.log',
                 num_epochs: Optional[int] = 16,
                 checkpoint_path: Optional[str] = './train data/checkpoints',
                 graph_dir: Optional[str] = './train data/graphs/train metrics',
                 verbose: Optional[bool] = True,
                 device: Optional[torch.device] = None) -> None:
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(graph_dir, exist_ok=True)
        
        self.logger = TrainingLogger(log_path=log_path)
        self.graph_dir = graph_dir
        self.graph_path = os.path.join(graph_dir, 'metrics.png')

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss_fn = loss_fn
        self.optimizer = optimizer or torch.optim.Adam(params=self.model.parameters(), lr=1e-3)

        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        
        # Video paths
        self.grid_video_path = os.path.join(self.graph_dir, 'latent_space_1.avi')
        self.dist_video_path = os.path.join(self.graph_dir, 'latent_space_2.avi')
        
        self.verbose = verbose
        self.loss_update_step = 900

        self.current_epoch = 1
        self.current_step = 1
        self.best_metric = float('inf')

        self.history = {
            'Training Loss': [],
            'Training Reconstruction Loss': [],
            'Training KL Loss': [],
            'Testing Loss': [],
            'Testing Reconstruction Loss': [],
            'Testing KL Loss': []
        }

        self.step_history = {
            'Training Loss': [],
            'Training Reconstruction Loss': [],
            'Training KL Loss': [],
            'Testing Loss': [],
            'Testing Reconstruction Loss': [],
            'Testing KL Loss': []
        }
        
        # For video creation
        self.grid_video_writer = None
        self.grid_video_frame_count = 0
        self.video_fps = 4  # Set to 60 FPS
        
        # For latent distribution video
        self.dist_video_writer = None 
        self.dist_video_frame_count = 0


    def update_plot(self) -> None:
        """Update the training/testing loss plot with 2 rows and 3 columns for different loss components"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Training loss (top row)
        axes[0, 0].plot(self.step_history['Training Loss'], color='blue')
        axes[0, 0].set_title('Training Total Loss')
        axes[0, 0].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(False)
        
        axes[0, 1].plot(self.step_history['Training Reconstruction Loss'], color='green')
        axes[0, 1].set_title('Training Reconstruction Loss')
        axes[0, 1].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(False)
        
        axes[0, 2].plot(self.step_history['Training KL Loss'], color='red')
        axes[0, 2].set_title('Training KL Loss')
        axes[0, 2].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(False)
        
        # Testing loss (bottom row)
        if len(self.step_history['Testing Loss']) > 0:
            axes[1, 0].plot(self.step_history['Testing Loss'], color='blue')
            axes[1, 0].set_title('Testing Total Loss')
            axes[1, 0].set_xlabel(f'Steps [every {self.loss_update_step}]')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(False)
            
            axes[1, 1].plot(self.step_history['Testing Reconstruction Loss'], color='green')
            axes[1, 1].set_title('Testing Reconstruction Loss')
            axes[1, 1].set_xlabel(f'Steps [every {self.loss_update_step}]')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(False)
            
            axes[1, 2].plot(self.step_history['Testing KL Loss'], color='red')
            axes[1, 2].set_title('Testing KL Loss')
            axes[1, 2].set_xlabel(f'Steps [every {self.loss_update_step}]')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(False)
        else:
            for i in range(3):
                axes[1, i].set_title(f'Testing Loss Component {i+1} (No Data Yet)')
                axes[1, i].grid(False)
        
        plt.tight_layout()
        plt.savefig(self.graph_path)
        plt.close()


    def visualize_latent_space(self, epoch, step=None) -> None:
        """Create a grid visualization of the latent space and add it to the video"""
        # Set model to eval mode for visualization
        self.model.eval()
        
        # Define a grid in latent space
        n = 80  # Grid size
        latent_dim = self.model.latent_dim  # Get latent dimension from model
        
        if latent_dim == 2:
            # For 2D latent space, create a grid
            x = np.linspace(-3, 3, n)
            y = np.linspace(-3, 3, n)
            
            # Create empty canvas to fill with generated digits
            canvas = np.zeros((28*n, 28*n))
            
            # Generate images for each grid point
            for i, xi in enumerate(x):
                for j, yi in enumerate(y):
                    # Create latent vector
                    z = torch.tensor([[xi, yi]], dtype=torch.float).to(self.device)
                    
                    # Decode latent vector
                    with torch.no_grad():
                        # Need to pass through decoder part only
                        # First convert latent to the format expected by decoder
                        decoder_input = self.model.linear_2(z).view(1, 128, 7, 7)
                        decoded = self.model.decoder(decoder_input)
                    
                    # Convert to image
                    img = decoded.cpu().numpy().squeeze()
                    
                    # Place in canvas
                    canvas[i*28:(i+1)*28, j*28:(j+1)*28] = img
        else:
            # For higher-dimensional latent spaces, interpolate between random points
            # Create a grid where two dimensions vary and others are fixed
            canvas = np.zeros((28*n, 28*n))
            
            # Generate a base random vector
            base_z = torch.randn(1, latent_dim).to(self.device)
            
            # Choose two dimensions to vary
            dim1, dim2 = 0, 1
            
            for i, v1 in enumerate(np.linspace(-3, 3, n)):
                for j, v2 in enumerate(np.linspace(-3, 3, n)):
                    # Create a copy of the base vector
                    z = base_z.clone()
                    
                    # Modify the chosen dimensions
                    z[0, dim1] = v1
                    z[0, dim2] = v2
                    
                    # Decode
                    with torch.no_grad():
                        decoder_input = self.model.linear_2(z).view(1, 128, 7, 7)
                        decoded = self.model.decoder(decoder_input)
                    
                    img = decoded.cpu().numpy().squeeze()
                    canvas[i*28:(i+1)*28, j*28:(j+1)*28] = img
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(canvas, cmap='gnuplot2')
        title_text = f'Traversing {latent_dim}D latent space (Epoch {epoch})'
        if step is not None:
            title_text += f', Step {step}'
        plt.title(title_text)
        plt.tight_layout()
        
        # Save with epoch info as temporary file for video
        temp_grid_filename = os.path.join(self.graph_dir, f'temp_grid_epoch_{epoch}_step_{step}.png')
        plt.savefig(temp_grid_filename)
        plt.close()
        
        # Add frame to video
        if self.grid_video_writer is None:
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame = cv2.imread(temp_grid_filename)
            height, width = frame.shape[:2]
            self.grid_video_writer = cv2.VideoWriter(self.grid_video_path, fourcc, self.video_fps, (width, height))
        
        # Add frame to video
        frame = cv2.imread(temp_grid_filename)
        self.grid_video_writer.write(frame)
        self.grid_video_frame_count += 1
        
        # Remove temporary file
        if os.path.exists(temp_grid_filename):
            os.remove(temp_grid_filename)
        
    def visualize_reconstructions(self, epoch, step=None, num_samples=10):
        """Create a visualization of original vs reconstructed images"""
        self.model.eval()
        
        # Get a batch of images from the test loader
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        images = images.to(self.device)
        
        # Get reconstructions
        with torch.no_grad():
            reconstructed, _, _ = self.model(images)
        
        # Select a subset of images to display
        images = images[:num_samples]
        reconstructed = reconstructed[:num_samples]
        labels = labels[:num_samples]
        
        # Create a figure showing original and reconstructed side by side
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        
        for i in range(num_samples):
            # Original images on top row
            axes[0, i].imshow(images[i].cpu().squeeze().numpy(), cmap='gray')
            axes[0, i].set_title(f'Original: {labels[i]}')
            axes[0, i].axis('off')
            
            # Reconstructed images on bottom row
            axes[1, i].imshow(reconstructed[i].cpu().squeeze().numpy(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        recon_path = os.path.join(self.graph_dir, f'recon_epoch_{epoch}.png')
        plt.savefig(recon_path)
        plt.close()
        
    def visualize_latent_distributions(self, epoch, step=None):
        """Create a visualization of latent space colored by digit class"""
        self.model.eval()
        all_mu = []
        all_labels = []
        
        # Process batches from test loader
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                
                # Get the encoded representation
                encoded = self.model.encoder(images)
                B, C, H, W = encoded.shape
                encoded_flat = self.model.linear_1(encoded.view(B, -1))
                
                # Get mean values (first half of the output)
                mean = encoded_flat[:, :encoded_flat.shape[1]//2]
                
                # Store means and labels
                all_mu.append(mean.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all batches
        all_mu = np.concatenate(all_mu, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Create visualization with specified colors
        plt.figure(figsize=(10, 10))
        
        # Define custom colors
        colors = ['yellow', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta']
        
        # Plot each digit class with its own color
        for i in range(10):  # 10 digits (0-9)
            idx = all_labels == i
            plt.scatter(all_mu[idx, 0], all_mu[idx, 1], c=colors[i], label=f'Digit {i}', alpha=0.7)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        title_text = f'Latent Space Distribution (Epoch {epoch}'
        if step is not None:
            title_text += f', Step {step})'
        else:
            title_text += ')'
        plt.title(title_text)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save visualization as temporary file for video
        temp_dist_path = os.path.join(self.graph_dir, f'temp_dist_epoch_{epoch}_step_{step}.png')
        plt.savefig(temp_dist_path)
        plt.close()
        
        # Add frame to latent distribution video
        if self.dist_video_writer is None:
            # Initialize video writer for latent distributions
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame = cv2.imread(temp_dist_path)
            height, width = frame.shape[:2]
            self.dist_video_writer = cv2.VideoWriter(self.dist_video_path, fourcc, self.video_fps, (width, height))
        
        # Add frame to video
        frame = cv2.imread(temp_dist_path)
        self.dist_video_writer.write(frame)
        self.dist_video_frame_count += 1
        
        # Remove temporary file
        if os.path.exists(temp_dist_path):
            os.remove(temp_dist_path)


    def train_epoch(self) -> None:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Training)') as t:
            
            for i, (images, _) in t:  # MNIST returns (images, labels)
                
                # Move data to device
                images = images.to(self.device)
                
                # Forward pass
                reconstructed, mean, log_var = self.model(images)
                
                # Calculate loss
                loss, recon_loss, kl_loss = self.loss_fn(images, reconstructed, mean, log_var)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                self.current_step += 1

                t.set_postfix({
                    'Batch Loss': loss.item(),
                    'Train Loss': total_loss/(i+1)
                })

                # Update plots and visualizations at specified intervals
                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Training Loss'].append(total_loss / (i+1))
                    self.step_history['Training Reconstruction Loss'].append(total_recon_loss / (i+1))
                    self.step_history['Training KL Loss'].append(total_kl_loss / (i+1))
                    self.update_plot()
                    
                    # Generate visualizations for video
                    if self.test_loader:
                        self.visualize_reconstructions(self.current_epoch, i)
                        self.visualize_latent_space(self.current_epoch, i)
                        
                        if self.model.latent_dim == 2:  # Only for 2D latent spaces
                            self.visualize_latent_distributions(self.current_epoch, i)

        train_loss = total_loss / len(self.train_loader)
        train_recon_loss = total_recon_loss / len(self.train_loader)
        train_kl_loss = total_kl_loss / len(self.train_loader)
        
        self.history['Training Loss'].append(train_loss)
        self.history['Training Reconstruction Loss'].append(train_recon_loss)
        self.history['Training KL Loss'].append(train_kl_loss)
        
        self.logger.info(f"Training losses for epoch {self.current_epoch}: Total={train_loss:.4f}, Recon={train_recon_loss:.4f}, KL={train_kl_loss:.4f}")
        
        return


    def test_epoch(self) -> None:
        """Test for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        with tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Testing)') as t:
            
            for i, (images, _) in t:  # MNIST returns (images, labels)
                
                # Move data to device
                images = images.to(self.device)
                
                # Forward pass with no gradient calculation
                with torch.no_grad():
                    reconstructed, mean, log_var = self.model(images)
                    loss, recon_loss, kl_loss = self.loss_fn(images, reconstructed, mean, log_var)
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

                t.set_postfix({
                    'Batch Loss': loss.item(),
                    'Test Loss': total_loss/(i+1)
                })

                # Update plots at specified intervals
                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Testing Loss'].append(total_loss / (i+1))
                    self.step_history['Testing Reconstruction Loss'].append(total_recon_loss / (i+1))
                    self.step_history['Testing KL Loss'].append(total_kl_loss / (i+1))
                    self.update_plot()

        test_loss = total_loss / len(self.test_loader)
        test_recon_loss = total_recon_loss / len(self.test_loader)
        test_kl_loss = total_kl_loss / len(self.test_loader)
        
        self.history['Testing Loss'].append(test_loss)
        self.history['Testing Reconstruction Loss'].append(test_recon_loss)
        self.history['Testing KL Loss'].append(test_kl_loss)

        # Save if best model
        if test_loss < self.best_metric:
            self.best_metric = test_loss
            self.save_checkpoint(is_best=True)

        self.logger.info(f"Testing losses for epoch {self.current_epoch}: Total={test_loss:.4f}, Recon={test_recon_loss:.4f}, KL={test_kl_loss:.4f}\n")
        
        return
    

    def train(self, resume_from: Optional[str]=None) -> None:
        """Train the model for specified number of epochs"""
        
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed training from epoch {self.current_epoch}")
            self.logger.log_training_resume(
                epoch=self.current_epoch, 
                global_step=self.current_step, 
                total_epochs=self.num_epochs
            )
        else:
            self.logger.info(f"Starting training for {self.num_epochs} epochs from scratch")
    
        print(f"Starting training from epoch {self.current_epoch} to {self.num_epochs}")
        
        try:
            for epoch in range(self.current_epoch, self.num_epochs + 1):
                self.current_epoch = epoch
                self.train_epoch()
                
                if self.test_loader:
                    self.test_epoch()
                    
                    # Create end-of-epoch visualizations
                    self.visualize_reconstructions(epoch)
                    self.visualize_latent_space(epoch)
                    
                    if self.model.latent_dim == 2:  # Only for 2D latent spaces
                        self.visualize_latent_distributions(epoch)
        
                self.save_checkpoint()
        finally:
            # Ensure all video writers are released
            if self.grid_video_writer is not None:
                self.grid_video_writer.release()
                self.logger.info(f"Created latent space grid video with {self.grid_video_frame_count} frames at {self.grid_video_path}")
                
            if hasattr(self, 'dist_video_writer') and self.dist_video_writer is not None:
                self.dist_video_writer.release()
                self.logger.info(f"Created latent distribution video with {self.dist_video_frame_count} frames at {self.dist_video_path}")
        
        return
    
    
    def save_checkpoint(self, is_best:Optional[bool]=False):
        """Save model checkpoint"""

        checkpoint = {
            'epoch': self.current_epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_history': self.step_history,
            'history': self.history,
            'best_metric': self.best_metric
        }

        if is_best:
            path = os.path.join(self.checkpoint_path, 'best_model.pth')
        else:
            path = os.path.join(self.checkpoint_path, f'model_epoch_{self.current_epoch}.pth')

        torch.save(checkpoint, path)
        
        if self.verbose:
            save_type = "Best model" if is_best else "Checkpoint"
            self.logger.info(f"{save_type} saved to {path}")


    def load_checkpoint(self, checkpoint:Optional[str]=None, resume_from_best:Optional[bool]=False):
        """Load model checkpoint"""
        
        if resume_from_best:
            checkpoint_path = os.path.join(self.checkpoint_path, 'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch') + 1
        self.current_step = checkpoint.get('current_step')
        self.best_metric = checkpoint.get('best_metric')
        
        loaded_history = checkpoint.get('history')
        for key in self.history:
            if key in loaded_history:
                self.history[key] = loaded_history[key]

        loaded_step_history = checkpoint.get('step_history')
        for key in self.step_history:
            if key in loaded_step_history:
                self.step_history[key] = loaded_step_history[key]
        
        return