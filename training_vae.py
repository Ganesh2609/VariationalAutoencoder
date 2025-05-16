import torch
from torch import nn
from trainer import ModularTrainer
from dataset import get_data_loaders
from utils import VariationalAutoencoder, VAELoss

def main():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 32

    model = VariationalAutoencoder(latent_dim=2).to(device)
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    loss_fn = VAELoss(alpha=0.001)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    log_path = './Train Data/Logs/train_3.log'
    checkpoint_path = './Train Data/Checkpoints/train 3'
    graph_dir = './Train Data/Graphs/train 3'

    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,  
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        log_path=log_path,
        num_epochs=num_epochs, 
        checkpoint_path=checkpoint_path,
        graph_dir=graph_dir,
        verbose=True,
        device=device 
    )

    trainer.train()
    #trainer.train(resume_from="./Train Data/Checkpoints/train 1/model_epoch_12.pth")

if __name__ == '__main__':
    main()
