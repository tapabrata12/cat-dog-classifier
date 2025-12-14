from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# CONFIGURATION
IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 2


def get_transforms():
	"""Defines how we process the images."""
	# Training: Resize, Flip (augment), Convert to Tensor, Normalize
	train_transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	
	# Validation: Resize, Convert to Tensor, Normalize (No flipping)
	val_transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	
	return train_transform, val_transform


def get_dataloaders(data_dir):
	"""Creates the loaders that feed data to the training loop."""
	train_dir = os.path.join(data_dir, 'train')
	val_dir = os.path.join(data_dir, 'val')
	
	train_transform, val_transform = get_transforms()
	
	# Create datasets
	train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
	val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
	
	# Create loaders
	train_loader = DataLoader(
		train_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=NUM_WORKERS,
		pin_memory=True
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=True
	)
	
	return train_loader, val_loader, train_dataset.classes