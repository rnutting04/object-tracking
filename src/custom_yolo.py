import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import glob

# --- CONFIGURATION (NO MISTAKES) ---
# Image size: 448x448 is standard for YOLO v1, giving us a clean 7x7 grid (64px stride)
IMG_SIZE = 448 
GRID_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 1 # Just "Person"
# Output depth: (x, y, w, h, conf) * 2 boxes + 1 class = 5*2 + 1 = 11
OUTPUT_DEPTH = NUM_BOXES * 5 + NUM_CLASSES 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. THE ARCHITECTURE (TINY DARKNET) ---
class TinyYOLO(nn.Module):
    def __init__(self):
        super(TinyYOLO, self).__init__()
        # Features: A simplified Darknet backbone
        self.features = nn.Sequential(
            # Block 1: Edges
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # 224x224

            # Block 2: Shapes
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # 112x112

            # Block 3: Textures/Parts
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # 56x56

            # Block 4: High Level Features
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # 28x28

            # Block 5: The Semantic Squeeze
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # 14x14
            
            # Block 6: Final Grid Sizing
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # 7x7
        )

        # Detection Head: 1x1 Convs to flatten depth to target size
        self.detector = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, OUTPUT_DEPTH, kernel_size=1), # Output: 11x7x7
            nn.Sigmoid() # Force all outputs to 0-1 range for stability
        )

    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        # Permute to (Batch, 7, 7, 11) for easier processing
        return x.permute(0, 2, 3, 1)

# --- 2. THE LOSS FUNCTION (CRITICAL PART) ---
class SimpleYOLOLoss(nn.Module):
    def __init__(self):
        super(SimpleYOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        
        # Hyperparameters to balance the loss
        self.lambda_coord = 5.0 
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        # predictions: (N, 7, 7, 11)
        # target: (N, 7, 7, 11) -- pre-encoded ground truth
        
        # 1. Identify which cells have objects
        # In our target format, the last channel (index 10) is class probability (1 if object exists)
        # Or better: use the confidence score of the first box slot in target (index 4)
        exists_box = target[..., 4].unsqueeze(3) # (N, 7, 7, 1)

        # ----------------------------------
        # BOX COORDINATES LOSS (x, y, w, h)
        # ----------------------------------
        # We only penalize coordinates if an object exists in that cell.
        # Box 1 prediction vs Target Box
        box1_pred = predictions[..., 0:4]
        box1_target = target[..., 0:4]
        
        # Box 2 prediction (We train both slots to predict the same object if one exists)
        box2_pred = predictions[..., 5:9]
        box2_target = target[..., 0:4] # Target is same for both slots

        # Calculate coordinate loss only where object exists
        box_loss = self.mse(box1_pred * exists_box, box1_target * exists_box) + \
                   self.mse(box2_pred * exists_box, box2_target * exists_box)

        # ----------------------------------
        # OBJECT CONFIDENCE LOSS
        # ----------------------------------
        # If object exists, we want confidence = 1.0
        # Box 1 Conf
        conf1_pred = predictions[..., 4:5]
        # Box 2 Conf
        conf2_pred = predictions[..., 9:10]
        
        object_loss = self.mse(conf1_pred * exists_box, exists_box) + \
                      self.mse(conf2_pred * exists_box, exists_box)

        # ----------------------------------
        # NO OBJECT LOSS (Background)
        # ----------------------------------
        # If NO object, we want confidence = 0.0
        # We penalize this less (lambda_noobj) because most cells are empty
        no_object_loss = self.mse(conf1_pred * (1 - exists_box), torch.zeros_like(exists_box)) + \
                         self.mse(conf2_pred * (1 - exists_box), torch.zeros_like(exists_box))

        # ----------------------------------
        # CLASS LOSS
        # ----------------------------------
        # Class prob is at index 10.
        class_pred = predictions[..., 10:11]
        class_target = target[..., 10:11]
        class_loss = self.mse(class_pred * exists_box, class_target * exists_box)

        # Total Loss
        loss = (self.lambda_coord * box_loss) + \
               object_loss + \
               (self.lambda_noobj * no_object_loss) + \
               class_loss
               
        return loss / predictions.size(0) # Normalize by batch size

# --- 3. DATASET HANDLING ---
class PersonDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = glob.glob(os.path.join(img_dir, "*.jpg"))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Load Image
        img_path = self.img_files[index]
        image = Image.open(img_path).convert("RGB")
        
        # Load Label
        # Expects a text file: class_id x_center y_center width height (normalized 0-1)
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    # Class 0 is Person
                    class_label, x, y, w, h = map(float, line.strip().split())
                    boxes.append([class_label, x, y, w, h])

        # Encode to Grid Tensor (7x7x11)
        label_matrix = torch.zeros((GRID_SIZE, GRID_SIZE, OUTPUT_DEPTH))
        
        for box in boxes:
            cls, x, y, w, h = box
            
            # Determine which grid cell (i, j) gets this object
            # x, y are normalized 0-1. Multiply by GRID_SIZE to get cell location.
            # Example: 0.5 * 7 = 3.5. Cell is index 3.
            i, j = int(GRID_SIZE * y), int(GRID_SIZE * x)
            x_cell, y_cell = GRID_SIZE * x - j, GRID_SIZE * y - i
            
            # If cell is already taken, we skip (simplification for demo)
            if label_matrix[i, j, 4] == 0:
                # Set Box 1 (Indices 0-4)
                label_matrix[i, j, 0] = x_cell # x relative to cell
                label_matrix[i, j, 1] = y_cell # y relative to cell
                label_matrix[i, j, 2] = w      # w relative to image
                label_matrix[i, j, 3] = h      # h relative to image
                label_matrix[i, j, 4] = 1.0    # Confidence
                
                # Set Box 2 (Indices 5-9) - Identical for training
                label_matrix[i, j, 5] = x_cell
                label_matrix[i, j, 6] = y_cell
                label_matrix[i, j, 7] = w
                label_matrix[i, j, 8] = h
                label_matrix[i, j, 9] = 1.0

                # Set Class (Index 10)
                label_matrix[i, j, 10] = 1.0   # Is Person

        if self.transform:
            image = self.transform(image)

        return image, label_matrix

# --- 4. TRAINING LOOP ---
def train_model():
    print("--- Initializing Demo Training ---")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # 1. Geometry Augmentation (Simulates Camera Angle Changes)
        # Randomly flip horizontal (Mirror view)
        transforms.RandomHorizontalFlip(p=0.5),
        # Randomly rotate +/- 10 degrees (Camera tilt)
        transforms.RandomRotation(degrees=10),
        # Randomly zoom in/out and squish (Simulates distance/perspective)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        
        # 2. Lighting Augmentation (Simulates Shadows/Different Room Spots)
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
    ])

    # FOLDER STRUCTURE EXPECTATION:
    # ./data/images/ (put your jpgs here)
    # ./data/labels/ (put your txts here)
    train_dataset = PersonDataset(img_dir="./data/images", label_dir="./data/labels", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = TinyYOLO().to(DEVICE)
    loss_fn = SimpleYOLOLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Training on {len(train_dataset)} images...")

    # Overfitting Loop (100 epochs is usually plenty for a demo on small data)
    model.train()
    for epoch in range(200):
        total_loss = 0
        for batch_idx, (img, target) in enumerate(train_loader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(img)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/100] Loss: {total_loss:.4f}")

    print("--- Training Complete. Saving weights. ---")
    torch.save(model.state_dict(), "simple_yolo_weights.pth")
    print("Saved to 'simple_yolo_weights.pth'")

if __name__ == "__main__":
    os.makedirs("./data/images", exist_ok=True)
    os.makedirs("./data/labels", exist_ok=True)
    if len(glob.glob("./data/images/*.jpg")) > 0:
        train_model()
    else:
        print("Dataset not found. Please put images in ./data/images and labels in ./data/labels")