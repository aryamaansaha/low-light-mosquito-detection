import os
import cv2
import torch
from torch.utils.data import Dataset

class MosquitoDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpeg")]
        self.img_size = img_size  # Resize images to this size

    def __len__(self):
        return len(self.image_files)

    def convert_yolo_to_pascal(self, label_file, img_width, img_height, new_width, new_height):
        """Convert YOLO annotations to Pascal VOC format and adjust for resized dimensions."""
        converted_labels = []
        scale_x = new_width / img_width
        scale_y = new_height / img_height

        with open(label_file, 'r') as file:
            for line in file:
                class_id, center_x, center_y, width, height = map(float, line.strip().split())

                # Convert YOLO (center_x, center_y, width, height) to Pascal VOC (x_min, y_min, x_max, y_max)
                x_min = (center_x - width / 2) * img_width
                y_min = (center_y - height / 2) * img_height
                x_max = (center_x + width / 2) * img_width
                y_max = (center_y + height / 2) * img_height

                # Adjust bounding box to the new image size
                x_min *= scale_x
                y_min *= scale_y
                x_max *= scale_x
                y_max *= scale_y

                converted_labels.append([class_id, x_min, y_min, x_max, y_max])

        return converted_labels

    def __getitem__(self, idx):
        # Image file path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace(".jpeg", ".txt"))

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get original dimensions
        orig_height, orig_width, _ = img.shape

        # Resize the image
        img = cv2.resize(img, (self.img_size, self.img_size))
        new_height, new_width = self.img_size, self.img_size

        # Convert YOLO annotations to Pascal VOC and adjust for resized dimensions
        pascal_labels = self.convert_yolo_to_pascal(label_path, orig_width, orig_height, new_width, new_height)

        boxes = []
        labels = []
        for label in pascal_labels:
            class_id, x_min, y_min, x_max, y_max = label
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]

        return img, target
