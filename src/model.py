import torch
import torch.nn as nn
import torch.nn.functional as f


class CatDogCNN(nn.Module):
	def __init__(self):
		super(CatDogCNN, self).__init__()
		
		# 1. Convolutional Block 1
		# Input: 3 channels (RGB), Output: 32 feature maps
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Cuts size in half
		
		# 2. Convolutional Block 2
		# Input: 32 features, Output: 64 features
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		
		# 3. Convolutional Block 3
		# Input: 64 features, Output: 128 features
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		
		# 4. Flattening & Fully Connected Layers (The Decision Maker)
		# We need to calculate the flattened size:
		# Image starts at 128x128 -> pool -> 64x64 -> pool -> 32x32 -> pool -> 16x16
		# Final volume = 128 features * 16 * 16 pixels
		self.fc1 = nn.Linear(128 * 16 * 16, 512)
		self.fc2 = nn.Linear(512, 2)  # Output: 2 scores (Cat, Dog)
		self.dropout = nn.Dropout(0.5)  # Prevents overfitting
	
	def forward(self, x):
		# Pass through Conv blocks with ReLU activation (makes math non-linear/smarter)
		x = self.pool(f.relu(self.conv1(x)))
		x = self.pool(f.relu(self.conv2(x)))
		x = self.pool(f.relu(self.conv3(x)))
		
		# Flatten: Turn the 3D cube of features into a 1D line of numbers
		x = x.view(-1, 128 * 16 * 16)
		
		# Decision-making
		x = f.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		return x


# Quick test to make sure the math works
if __name__ == "__main__":
	# Create a dummy image (Batch Size 1, RGB 3, 128x128)
	dummy_input = torch.randn(1, 3, 128, 128)
	model = CatDogCNN()
	output = model(dummy_input)
	print(f"✅ Model created successfully!")
	print(f"   Input shape:  {dummy_input.shape}")
	print(f"   Output shape: {output.shape}")