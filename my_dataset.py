from PIL import Image
from torch.utils.data import Dataset


class BatchDataset(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None,num_classes=45):
        self.imglist = images_path
        self.labels = images_class
        self.transform = transform
        self.num_classes = num_classes
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, item):
        img = Image.open(self.imglist[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.imglist[item]))
        label = self.labels[item]
        img = self.transform(img)
        return [img, label]