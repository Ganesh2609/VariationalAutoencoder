# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms


# def get_data_loaders(batch_size:int=16, num_workers:int=10, prefetch_factor:int=2):

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5),(0.5))
#     ])
    
#     train_data = datasets.MNIST(
#         root='MNIST Dataset',
#         train=True,
#         download=True,
#         transform=transform
#     )

#     test_data = datasets.MNIST(
#         root='MNIST Dataset',
#         train=False,
#         download=True,
#         transform=transform
#     )

#     train_loader = DataLoader(
#         train_data, 
#         batch_size=batch_size, 
#         shuffle=True,
#         num_workers=num_workers, 
#         pin_memory=True, 
#         persistent_workers=True,  
#         prefetch_factor=prefetch_factor,
#         drop_last=True
#     )

#     test_loader = DataLoader(
#         test_data, 
#         batch_size=batch_size, 
#         shuffle=False,
#         num_workers=num_workers, 
#         pin_memory=True, 
#         persistent_workers=True,  
#         prefetch_factor=prefetch_factor,
#         drop_last=True
#     )

#     return train_loader, test_loader








from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size:int=16, num_workers:int=10, prefetch_factor:int=2):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])
    
    train_data = datasets.EMNIST(
        root='EMNIST Dataset',
        split='bymerge',
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.EMNIST(
        root='EMNIST Dataset',
        split='bymerge',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True,  
        prefetch_factor=prefetch_factor,
        drop_last=True
    )

    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True,  
        prefetch_factor=prefetch_factor,
        drop_last=True
    )

    return train_loader, test_loader