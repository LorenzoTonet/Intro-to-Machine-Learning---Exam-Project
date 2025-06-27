import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

class SiameseDataset(Dataset):
    """
    Dataset personalizzato per la Siamese Network.
    Genera coppie di immagini con etichette di similarità (1 se stessa classe, 0 altrimenti)
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.labels = [dataset[i][1] for i in range(len(dataset))]
        
        # Organizza gli indici per classe per facilitare il sampling
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Ottieni la prima immagine
        img1, label1 = self.dataset[idx]
        
        # Decidi se creare una coppia positiva (stessa classe) o negativa (classi diverse)
        should_get_same_class = random.random() > 0.5
        
        if should_get_same_class:
            # Coppia positiva: stessa classe
            idx2 = random.choice(self.label_to_indices[label1])
            img2, label2 = self.dataset[idx2]
            target = 1.0  # Similarità alta
        else:
            # Coppia negativa: classe diversa
            different_labels = [l for l in self.label_to_indices.keys() if l != label1]
            label2 = random.choice(different_labels)
            idx2 = random.choice(self.label_to_indices[label2])
            img2, _ = self.dataset[idx2]
            target = 0.0  # Similarità bassa
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(target, dtype=torch.float32)

def prepare_semisupervised_mnist_data(subset_ratio=0.2, train_ratio=0.75):
    """
    Prepara il dataset MNIST per il training della Siamese Network
    
    Args:
        subset_ratio: Percentuale del dataset MNIST da utilizzare come labeled (default: 20%)
        train_ratio: Percentuale del subset per il training (default: 75%)
    
    Returns:
        train_loader, test_loader, full_dataset
    """
    
    # Trasformazioni per normalizzare i dati
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # normalizzare i valori dei pixel(aiuta la convergenza)
    ])
    
    # Carica il dataset MNIST completo
    full_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Prendi solo una parte del dataset (20% di default)
    subset_size = int(len(full_mnist) * subset_ratio)
    subset_indices = torch.randperm(len(full_mnist))[:subset_size]
    mnist_subset = Subset(full_mnist, subset_indices)
    
    # Dividi il subset in train e test
    train_size = int(len(mnist_subset) * train_ratio)
    test_size = len(mnist_subset) - train_size
    
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(mnist_subset)))
    
    train_subset = Subset(mnist_subset, train_indices)
    test_subset = Subset(mnist_subset, test_indices)
    
    # Crea i dataset per la Siamese Network
    train_siamese_dataset = SiameseDataset(train_subset, transform=None)
    test_siamese_dataset = SiameseDataset(test_subset, transform=None)
    
    # Crea i DataLoader
    train_loader = DataLoader(
        train_siamese_dataset, 
        batch_size=32,
        shuffle=True,
    )
    
    test_loader = DataLoader(
        test_siamese_dataset, 
        batch_size=32, 
        shuffle=False,
    )
    
    print(f"Dataset preparato:")
    print(f"- Subset totale: {len(mnist_subset)} campioni")
    print(f"- Training set: {len(train_subset)} campioni")
    print(f"- Test set: {len(test_subset)} campioni")
    print(f"- Batch size: 32")
    
    return train_loader, test_loader, full_mnist


def visualize_pairs(dataset, num_pairs=5):
    """
    Visualizza alcune coppie di esempio dal dataset
    """
    fig, axes = plt.subplots(num_pairs, 2, figsize=(8, 2*num_pairs))
    
    for i in range(num_pairs):
        img1, img2, label = dataset[i]
        
        # Denormalizza per la visualizzazione
        img1 = img1 * 0.3081 + 0.1307
        img2 = img2 * 0.3081 + 0.1307
        
        axes[i, 0].imshow(img1.squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Immagine 1')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img2.squeeze(), cmap='gray')
        axes[i, 1].set_title(f'Immagine 2 (Sim: {label.item():.0f})')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
