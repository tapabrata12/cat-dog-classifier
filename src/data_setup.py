"""
Action: Create the Split Script
We need to take the images from PetImages and move them into train (80% of images)
and val (20% of images) folders.
"""
import os
import shutil
import random
from pathlib import Path

# Config
RAW_DATA_PATH = Path("../data/PetImages")
TRAIN_PATH = Path("../data/train")
VAL_PATH = Path("../data/val")
SPLIT_RATIO = 0.8  # 80% for training, 20% for validation

"""
Creates train and val directories with category subfolders.
"""

def setup_directories():
	for categories in ['Cat', 'Dog']:
		os.makedirs(TRAIN_PATH / categories, exist_ok=True)
		os.makedirs(VAL_PATH / categories, exist_ok=True)

"""
We need to make sure the photos aren't broken before we use them
"""

def valid_image(file_path):
	# This asks, "Is this file empty?" If it has 0 bytes, it's a ghost file!
	if file_path.stat().st_size == 0:
		return False
	
	# This asks, "Is this actually a picture?" If the file name doesn't end in
	# .jpg or .png, it might be a text file or a virus. We skip it.
	elif file_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
		return False
	
	else:
		return True

"""
Splits data into training and validation sets.
"""
def split_data():
	print("🚀 Starting data split...")
	setup_directories()
	total_moved = 0
	for category in ["Cat", "Dog"]:
		source_dir = RAW_DATA_PATH / category
		images = [f for f in source_dir.iterdir() if f.is_file() and valid_image(f)]
		# Shuffle to ensure random split
		random.shuffle(images)
		split_point = int(len(images) * SPLIT_RATIO)
		train_images = images[:split_point]
		val_images = images[split_point:]
		print(f"   Processing {category}: {len(train_images)} train, {len(val_images)} val")
		
		for img in train_images:
			shutil.copy(img,TRAIN_PATH / category / img.name)
		
		for img in val_images:
			shutil.copy(img, VAL_PATH / category / img.name)
		total_moved  += len(images)
		
		print(f"✅ Done! {total_moved} images processed.")
		print(f"📂 Training data located at: {TRAIN_PATH.resolve()}")
		print(f"📂 Validation data located at: {VAL_PATH.resolve()}")


if __name__ == "__main__":
	# Ensure we are running from the src directory or adjust paths accordingly
	# This check helps if you run from the root folder
		if not os.path.exists(RAW_DATA_PATH):
			print(f"⚠️  Error: Could not find {RAW_DATA_PATH}. Make sure you run this script from the 'src' folder!")
		else:
			split_data()