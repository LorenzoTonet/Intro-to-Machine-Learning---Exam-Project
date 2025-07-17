import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

def analyze_clusters(true_labels, cluster_labels):
    """
    Analizza la corrispondenza tra cluster e classi vere
    """
    # Matrice di confusione tra cluster e classi vere
    cm = confusion_matrix(true_labels, cluster_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Matrice di Confusione: Classi Vere vs Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Classi Vere (MNIST)')
    plt.show()
    
    return cm

def visualize_embeddings(embeddings, true_labels, cluster_labels, embedding_size, perplexity=30):
    """
    Visualizza gli embedding in 2D
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    if embedding_size == 2:
        # Se gli embedding sono già 2D, usali direttamente
        embeddings_2d = embeddings
    else:
        # Altrimenti usa t-SNE per riduzione dimensionalità
        print("Applicando t-SNE per visualizzazione...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot delle vere etichette
    scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=true_labels, cmap='tab10', alpha=0.7, s=1)
    axes[0].set_title('Embedding - Vere Etichette')
    axes[0].set_xlabel('Dimensione 1')
    axes[0].set_ylabel('Dimensione 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot dei cluster predetti
    scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=cluster_labels, cmap='tab10', alpha=0.7, s=1)
    axes[1].set_title('Embedding - Labels')
    axes[1].set_xlabel('Dimensione 1')
    axes[1].set_ylabel('Dimensione 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()