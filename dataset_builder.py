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

class SiameseCoupleDataset(Dataset):
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

def prepare_mnist_data_couples(subset_ratio=0.2, train_ratio=0.75, batchsize = 32):
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
    train_siamese_dataset = SiameseCoupleDataset(train_subset, transform=None)
    test_siamese_dataset = SiameseCoupleDataset(test_subset, transform=None)
    
    # Crea i DataLoader
    train_loader = DataLoader(
        train_siamese_dataset, 
        batch_size=batchsize,
        shuffle=True,
    )
    
    test_loader = DataLoader(
        test_siamese_dataset, 
        batch_size=batchsize, 
        shuffle=False,
    )
    
    print(f"Dataset preparato:")
    print(f"- Subset totale: {len(mnist_subset)} campioni")
    print(f"- Training set: {len(train_subset)} campioni")
    print(f"- Test set: {len(test_subset)} campioni")
    print(f"- Batch size: {batchsize}")
    
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

###################################################################

class SiameseTripletsDataset(Dataset):
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
        # Ottieni l'immagine anchor
        anchor_img, anchor_lab = self.dataset[idx]

        equal_labels = [l for l in self.label_to_indices.keys() if l == anchor_lab]
        different_labels = [l for l in self.label_to_indices.keys() if l != anchor_lab]

        positive_lab = random.choice(equal_labels)
        positive_idx = random.choice(self.label_to_indices[positive_lab])
        positive_img, _ = self.dataset[positive_idx]
        
        negative_lab = random.choice(different_labels)
        negative_idx = random.choice(self.label_to_indices[negative_lab])
        negative_img, _ = self.dataset[negative_idx]

        if self.transform:
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            anchor_img = self.transform(anchor_img)
        
        return anchor_img, positive_img, negative_img
    
def prepare_mnist_data_triplets(subset_ratio=0.2, train_ratio=0.75, batchsize = 32):
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
    train_siamese_dataset = SiameseTripletsDataset(train_subset, transform=None)
    test_siamese_dataset = SiameseTripletsDataset(test_subset, transform=None)
    
    # Crea i DataLoader
    train_loader = DataLoader(
        train_siamese_dataset, 
        batch_size=batchsize,
        shuffle=True,
    )
    
    test_loader = DataLoader(
        test_siamese_dataset, 
        batch_size=batchsize, 
        shuffle=False,
    )
    
    print(f"Dataset preparato:")
    print(f"- Subset totale: {len(mnist_subset)} campioni")
    print(f"- Training set: {len(train_subset)} campioni")
    print(f"- Test set: {len(test_subset)} campioni")
    print(f"- Batch size: {batchsize}")
    
    return train_loader, test_loader, full_mnist

def visualize_triplets(dataset, num_triplets=5):
    """
    Visualizza alcune coppie di esempio dal dataset
    """
    fig, axes = plt.subplots(num_triplets, 3, figsize=(12, 2*num_triplets))
    
    for i in range(num_triplets):
        anch, pos, neg = dataset[i]
        
        # Denormalizza per la visualizzazione
        anch = anch * 0.3081 + 0.1307
        pos = pos * 0.3081 + 0.1307
        neg = neg * 0.3081 + 0.1307
        
        axes[i, 0].imshow(anch.squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Anchor')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pos.squeeze(), cmap='gray')
        axes[i, 1].set_title(f'Positive')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(neg.squeeze(), cmap='gray')
        axes[i, 2].set_title(f'Negative')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def MNIST_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    return data_loader

def create_embedding_dataset(model, data_loader):
    """
    Crea un dataset di embedding usando la rete siamese allenata
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            
            # Genera embedding usando forward_once
            embedding = model.forward_once(data)
            
            embeddings.append(embedding.cpu().numpy())
            labels.extend(target.numpy())
            
            if batch_idx % 100 == 0:
                print(f'Processato batch {batch_idx}/{len(data_loader)}')
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    return embeddings, labels


if __name__ == "__main__":

    train_loader_triplet, test_loader_triplet, full_dataset_triplet = prepare_mnist_data_triplets()
    visualize_triplets(train_loader_triplet.dataset, num_triplets=5)