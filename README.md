The following files are included:
	1. create_cifar_data.py -> Preprocesses the CIFAR10 dataset and converts them to numpy arrays
	2. tsne_visualization.py -> Performs t-SNE on both CIFAR10 and MNIST datasets and plots the t-SNE representations
	3. encoders.py -> Implementation of Regular Autoencoder and Variational Autoencoder 
	4. cnn.py -> CNN implementation for CIFAR10 and MNIST dataset

Run the above files as:
	1. create_cifar_data.py -> python create_cifar_data.py
	2. tsne_visualization.py -> python tsne_visualization.py
	3. encoders.py -> python encoders.py cifar|mnist (To run for CIFAR/MNIST respectively)
	4. cnn.py -> python cnn.py cifar|mnist (To run for CIFAR/MNIST respectively)