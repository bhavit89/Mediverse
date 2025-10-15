import os 
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms 

class TumorDataset(Dataset):
    def __init__(self, val=True):
        super().__init__()
        if val:
            self.image_dir = "Tumor_dataset/train/images/"
            self.mask_dir = "Tumor_dataset/train/masks/"
            self.image_files = sorted(os.listdir(self.image_dir))
            self.mask_files = sorted(os.listdir(self.mask_dir))
        else:
            self.image_dir = "Tumor_dataset/val/images/"
            self.mask_dir = "Tumor_dataset/val/masks/"
            self.image_files = sorted(os.listdir(self.image_dir))
            self.mask_files = sorted(os.listdir(self.mask_dir))
        
        
    
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        return self.transform(img), self.transform(mask)
    
    def __len__(self):
        return len(self.image_files)