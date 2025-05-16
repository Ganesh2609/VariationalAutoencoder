# Variational Autoencoder

A PyTorch implementation of a Variational Autoencoder (VAE) with visualizations of the latent space.

## Overview

This project implements a Variational Autoencoder (VAE) with a focus on latent space visualization and exploration. The implementation features:

- A modular VAE architecture with configurable latent dimensions
- Latent space visualization and traversal
- Reconstruction quality monitoring
- Comprehensive training metrics tracking
- Multiple training runs with different latent space dimensions (2D and 8D)

## Architecture

The VAE consists of:

### Encoder
- Convolutional layers that progressively reduce spatial dimensions while increasing channels
- Final linear layer projecting to the latent space (mean and log variance)

### Decoder
- Linear layer to project from latent space to reshaped tensor
- Transposed convolutions to progressively increase spatial dimensions
- Final output layer with tanh activation

### Loss Function
- Reconstruction loss: Mean Squared Error (MSE) between input and output
- KL Divergence loss: Regularization term to enforce latent space distribution
- Combined loss with configurable weighting (alpha parameter)

## Results

### Training Run 1 (2D Latent Space)

This model was trained for 128 epochs on the EMNIST dataset with a 2-dimensional latent space.

#### Latent Space Visualization

The 2D latent space allows direct visualization of how the model organizes the data:

![Latent Space Visualization](./Train%20Data/Graphs/train%201/latent_space_2.avi)

#### Latent Space Traversal

By traversing the latent space, we can see how the digit representations change continuously:

![Latent Space Traversal](./Train%20Data/Graphs/train%201/latent_space_1.avi)

#### Training Metrics

The loss metrics show the training progression over time:

![Training Metrics](./Train%20Data/Graphs/train%201/metrics.png)

#### Reconstruction Examples

Examples of original images and their reconstructions at different epochs:

![Reconstruction at Epoch 1](./Train%20Data/Graphs/train%201/recon_epoch_1.png)
![Reconstruction at Epoch 64](./Train%20Data/Graphs/train%201/recon_epoch_64.png)
![Reconstruction at Epoch 128](./Train%20Data/Graphs/train%201/recon_epoch_128.png)

### Training Run 2 (8D Latent Space)

This model was trained for 32 epochs on the EMNIST dataset with an 8-dimensional latent space.

#### Latent Space Visualization

While the full 8D space cannot be directly visualized, we show projections of the latent space:

![Latent Space Visualization](./Train%20Data/Graphs/train%202/latent_space_1.avi)

#### Training Metrics

The loss metrics show the training progression over time:

![Training Metrics](./Train%20Data/Graphs/train%202/metrics.png)

#### Reconstruction Examples

Examples of original images and their reconstructions:

![Reconstruction at Epoch 16](./Train%20Data/Graphs/train%202/recon_epoch_16.png)
![Reconstruction at Epoch 32](./Train%20Data/Graphs/train%202/recon_epoch_32.png)

## Key Findings

- The 2D latent space provides an interpretable representation where digits cluster naturally
- Higher-dimensional latent spaces (8D) offer better reconstruction quality at the cost of direct interpretability
- KL divergence loss helps ensure a smooth, continuous latent space
- Reconstructions improve significantly over the course of training

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/Ganesh2609/VariationalAutoencoder.git
cd VariationalAutoencoder

# Install dependencies
pip install torch torchvision matplotlib numpy opencv-python tqdm
```

### Training

To train the model with default parameters:

```bash
python training_vae.py
```

To modify latent dimensions or other parameters, edit the `main()` function in `training_vae.py`:

```python
# Change latent_dim to experiment with different dimensions
model = VariationalAutoencoder(latent_dim=2).to(device)

# Adjust learning rate, batch size, or epochs
learning_rate = 1e-3
batch_size = 64
num_epochs = 32
```

### Resuming Training

To resume training from a checkpoint:

```python
# Uncomment and modify the path to resume training
trainer.train(resume_from="./Train Data/Checkpoints/train 1/model_epoch_12.pth")
```

## Code Structure

- `dataset.py`: Data loading utilities using EMNIST dataset
- `logger.py`: Custom logging functionality
- `trainer.py`: Training loop with visualization and metrics tracking
- `training_vae.py`: Main script to launch training
- `utils.py`: Model architecture and loss function definitions