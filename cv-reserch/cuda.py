import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from prometheus_client import start_http_server, Counter, Gauge
import albumentations as A
import psutil
import GPUtil
import os

GPU_USAGE = Gauge('gpu_usage', 'GPU Usage Percentage')
CPU_USAGE = Gauge('cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('ram_usage', 'RAM Usage Percentage')
DISK_USAGE = Gauge('disk_usage', 'Disk Usage Percentage')
DISK_READ = Gauge('system_disk_read', 'Disk read in bytes per second')
DISK_WRITE = Gauge('system_disk_write', 'Disk write in bytes per second')


def update_metrics():
    GPU_USAGE.set(GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0)
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    disk_io = psutil.disk_io_counters()
    DISK_READ.set(disk_io.read_bytes)
    DISK_WRITE.set(disk_io.write_bytes)
    gpus = GPUtil.getGPUs()
    if gpus:
        GPU_USAGE.set(gpus[0].load)

class FootballFieldDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_pil_image(image)
        mask = cv2.imread(mask_path, 0) 
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) 

        mask = np.where(mask == 113, 1, mask)
        mask = np.where(mask == 169, 2, mask)
        mask = np.where(mask == 227, 3, mask)

        unique_values = np.unique(mask)
        print(f"Unique mask values: {unique_values}")


        mask = torch.from_numpy(mask).long()

        if self.transform is not None:
            image = self.transform(image)

        return image, mask


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def get_model(num_classes=4):
    # model = smp.PSPNet(
    model = smp.DeepLabV3(
    # model = smp.Unet(
        encoder_name="resnet50", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=num_classes)

    return model


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def iou_score(output, target):
    output = torch.sigmoid(output)
    output = torch.argmax(output, dim=1)
    output = output.view(-1)
    target = target.view(-1)

    intersection = (output == target).float().sum()
    union = output.numel()

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def dice_score(output, target):
    output = torch.sigmoid(output)
    output = torch.argmax(output, dim=1)
    output = output.view(-1)
    target = target.view(-1)

    intersection = (output == target).float().sum()
    dice = (2. * intersection + 1e-6) / (output.numel() + target.numel() + 1e-6)
    return dice.item()


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    iou_total = 0
    dice_total = 0
    count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            iou_total += iou_score(outputs, masks)
            dice_total += dice_score(outputs, masks)
            count += 1
    return total_loss / count, iou_total / count, dice_total / count


def post_process_mask(mask):
    mask = mask.astype(np.uint8)

    kernel = np.ones((6, 6), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing


def visualize(model, data_loader, device, num_images=5):
    model.eval()
    fig, ax = plt.subplots(nrows=num_images, ncols=4, figsize=(20, num_images * 5))

    for i, (images, gt_masks) in enumerate(data_loader):
        if i >= num_images:
            break

        images = images.to(device)
        gt_masks = gt_masks.cpu().numpy()

        with torch.no_grad():
            preds = model(images)
        preds = torch.argmax(preds, dim=1).cpu().numpy()
        images = images.cpu().numpy()

        for j in range(len(images)):
            ax[i, 0].imshow(np.transpose(images[j], (1, 2, 0)))
            ax[i, 1].imshow(gt_masks[j], cmap='gray')
            ax[i, 2].imshow(preds[j], cmap='gray')
            processed_mask = post_process_mask(preds[j])
            ax[i, 3].imshow(processed_mask, cmap='gray')

            ax[i, 0].set_title("Original Image")
            ax[i, 1].set_title("Ground Truth Mask")
            ax[i, 2].set_title("Predicted Mask")
            ax[i, 3].set_title("Post Processed Mask")
            for k in range(4):
                ax[i, k].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    update_metrics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = 'segmentation_labeled_dataset'
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])

    train_dataset = FootballFieldDataset(
        image_dir=os.path.join(dataset_root, 'train', 'JPEGImages'),
        mask_dir=os.path.join(dataset_root, 'train', 'SegmentationClass'),
        transform=transform
    )
    val_dataset = FootballFieldDataset(
        image_dir=os.path.join(dataset_root, 'val', 'JPEGImages'),
        mask_dir=os.path.join(dataset_root, 'val', 'SegmentationClass'),
        transform=transform
    )


    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=10, pin_memory=True)


    model = get_model(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    best_val_loss = float('inf')
    best_model_path = ""
    epoch = 0
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    try:
        while True:
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            update_metrics()
            val_loss = validate(model, val_loader, criterion)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(models_dir, f'model_checkpoint_epoch_{epoch}.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"Checkpoint saved for epoch {epoch} with validation loss: {val_loss} at {best_model_path}")

            epoch += 1

    except KeyboardInterrupt:
        print("Training interrupted.")

    print("Training complete.")

    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
    
    test_dataset = FootballFieldDataset(
        image_dir=os.path.join(dataset_root, 'test', 'JPEGImages'),
        mask_dir=os.path.join(dataset_root, 'test', 'SegmentationClass'),
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    test_loss, test_iou, test_dice = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss}, Test IoU: {test_iou}, Test Dice: {test_dice}")


    visualize_results = True
    if visualize_results:
        visualize(model, test_loader, device, num_images=5)


if __name__ == "__main__":
    start_http_server(8083)
    main()