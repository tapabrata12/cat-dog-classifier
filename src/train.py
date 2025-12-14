import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import get_dataloaders
from model import CatDogCNN

# CONFIGURATION
# We use a lower learning rate to be careful not to overshoot the best spot
LEARNING_RATE = 0.001
EPOCHS = 10  # How many times we show the entire dataset to the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "../models/cat_dog_model.pth"


def train_model():
	print(f"🚀 Training on device: {DEVICE}")
	
	# 1. Prepare Data and Model
	# We step out to the main folder to find 'data'
	train_loader, val_loader, class_names = get_dataloaders("../data")
	
	model = CatDogCNN().to(DEVICE)  # Move the brain to the GPU
	
	# 2. Set up the "Critic" and "Coach"
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	
	best_val_accuracy = 0.0
	
	# 3. The Loop
	for epoch in range(EPOCHS):
		print(f"\nEpoch {epoch + 1}/{EPOCHS}")
		print("-" * 30)
		
		# --- TRAINING PHASE ---
		model.train()  # Set model to 'learning mode' (enables dropout)
		running_loss = 0.0
		
		for images, labels in train_loader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			
			# Zero the gradients (clear previous calculations)
			optimizer.zero_grad()
			
			# Forward pass (Guess)
			outputs = model(images)
			
			# Calculate error
			loss = criterion(outputs, labels)
			
			# Backward pass (Learn)
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
		
		avg_train_loss = running_loss / len(train_loader)
		print(f"   Train Loss: {avg_train_loss:.4f}")
		
		# --- VALIDATION PHASE ---
		model.eval()  # Set model to 'test mode' (freezes layers)
		correct = 0
		total = 0
		
		with torch.no_grad():  # Don't calculate gradients (saves memory)
			for images, labels in val_loader:
				images, labels = images.to(DEVICE), labels.to(DEVICE)
				outputs = model(images)
				
				# Get predictions (the index with the highest score is the winner)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		
		val_accuracy = 100 * correct / total
		print(f"   Val Accuracy: {val_accuracy:.2f}%")
		
		# Save the model if it's the best one so far
		if val_accuracy > best_val_accuracy:
			best_val_accuracy = val_accuracy
			os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
			torch.save(model.state_dict(), MODEL_SAVE_PATH)
			print(f"   💾 New best model saved!")
	
	print("\n✅ Training Complete!")


if __name__ == "__main__":
	train_model()