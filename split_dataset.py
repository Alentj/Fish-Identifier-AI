import os
import random
import shutil

source_dir = "kerala_fish_dataset"
train_dir = "dataset/train"
test_dir = "dataset/test"

# create train and test folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for fish in os.listdir(source_dir):

    fish_path = os.path.join(source_dir, fish)

    if os.path.isdir(fish_path):

        images = os.listdir(fish_path)
        random.shuffle(images)

        split_index = int(len(images) * 0.8)

        train_images = images[:split_index]
        test_images = images[split_index:]

        os.makedirs(os.path.join(train_dir, fish), exist_ok=True)
        os.makedirs(os.path.join(test_dir, fish), exist_ok=True)

        for img in train_images:
            shutil.copy(
                os.path.join(fish_path, img),
                os.path.join(train_dir, fish, img)
            )

        for img in test_images:
            shutil.copy(
                os.path.join(fish_path, img),
                os.path.join(test_dir, fish, img)
            )

print("Dataset split completed!")
