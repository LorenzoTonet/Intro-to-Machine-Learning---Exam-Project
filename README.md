# Intro to Machine Learning - Exam Project

## Semi-Supervised Spectral Clustering with Siamese Networks

This project explores a hybrid approach that combines supervised representation learning with unsupervised spectral clustering, in order to improve clustering performance on datasets where only a small portion of the data is labeled.

Spectral clustering typically relies on building a similarity graph from a dataset, where each node corresponds to a data point and edges are weighted based on a predefined similarity (or distance) function, such as the Euclidean distance or the RBF kernel. However, choosing the right similarity function is often non-trivial and can greatly affect the clustering results.

To address this, the idea is to **learn a task-specific similarity metric** using a **Siamese Neural Network**, leveraging the limited labeled data available. This network is trained to embed the data into a new space where similar points (i.e., points with the same label) are closer together, and dissimilar ones are further apart.

Once the Siamese Network has been trained on the labeled portion of the dataset, it is used to embed the entire dataset (including the unlabeled points). A similarity graph is then constructed by computing pairwise distances in this learned embedding space. Finally, **spectral clustering** is applied to this graph to discover clusters that reflect the learned notion of similarity.

## Pipeline

1. **Input**: A dataset where a small subset of points is labeled.
2. **Training**: Use the labeled data to train a Siamese Neural Network that learns an embedding space where similar points are close.
3. **Embedding**: Apply the trained network to the entire dataset to get embeddings for all data points.
4. **Graph Construction**: Build a weighted graph using pairwise distances (e.g., Euclidean) in the learned embedding space.
5. **Spectral Clustering**: Perform spectral clustering on the graph to obtain the final clusters.

