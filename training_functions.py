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
import pickle

from sklearn.cluster import SpectralClustering

from torchvision.datasets import MNIST, KMNIST

from dataset_builder import *
from clustering_utils import *
from Siamese_networks import *


def train_couple_model(model, train_loader, optimizer, criterion, epochs):

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(epochs):

        for i, (img0, img1, label) in enumerate(train_loader, 0):

            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 50 == 0 :
                print(f"Epoch {epoch} batch {i}\nloss {loss_contrastive.item()}\n")
                iteration_number += 10

                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    
    return (counter, loss_history)

def test_couple_model(model, dataset, num_examples=5, verbose = True):
    """
    Testa il modello trainato su alcuni esempi
    """
    giuste = 0
    sbagliate = 0
    model.eval()
    device = next(model.parameters()).device
    
    if verbose:
        print(f"Test del modello su {num_examples} esempi:")
        print("-" * 60)
    
    for i in range(num_examples):
        img1, img2, true_label = dataset[i]
        
        # Aggiungi dimensione batch
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output1, output2 = model(img1, img2)
            euclidean_distance = F.pairwise_distance(output1, output2)
            similarity = torch.exp(-euclidean_distance)
            predicted_label = 1 if similarity.item() > 0.5 else 0
        
        if predicted_label == true_label:
            giuste +=1
        else:
            sbagliate +=1
    
    if verbose:
        print("Il modello ne ha beccate: " + str(giuste))
        print("Il modello ne ha sbagliate: " + str(sbagliate))

    accuracy = giuste/num_examples
    return accuracy

def train_triplet_model(model, train_loader, optimizer, criterion, epochs):

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(epochs):

        for i, (a, p, n) in enumerate(train_loader, 0):

            optimizer.zero_grad()
            output1, output2, output3 = model(a,p,n)
            loss_triplet = criterion(output1, output2, output3)
            loss_triplet.backward()
            optimizer.step()
            if i % 50 == 0 :
                print(f"Epoch {epoch} | batch {i} | Loss: {loss_triplet.item():.5f}\n")
                iteration_number += 10

                counter.append(iteration_number)
                loss_history.append(loss_triplet.item())
    
    return (counter, loss_history)

def test_triplet_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for a, p, n in test_loader:
            output1, output2, output3 = model(a, p, n)
            
            distances_ap = F.pairwise_distance(output1, output2)
            distances_an = F.pairwise_distance(output1, output3)
            correct += (distances_ap < distances_an).sum().item()
            total += len(a)

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

