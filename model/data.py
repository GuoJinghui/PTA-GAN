import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AP_TEM_Dataset(Dataset):
    def __init__(self, pp, ap, tem, mask):
        samples = []
        # self.PP = np.array(pp).reshape(len(pp), 1, 146, 199)
        self.PP = pp
        self.AP = ap
        self.TEM = tem
        self.MASK = mask
        self.transform = self.get_transform()
        for i in range(len(self.PP)):
            samples.append((self.PP[i], self.AP[i], self.TEM[i], self.MASK[i]))
        self.samples = samples

    def __len__(self):
        return len(self.PP)

    def __getitem__(self, idx):
        pp, ap, tem, mask = self.samples[idx]
        return self.one_hot_sample(pp, ap, tem, mask)

    def one_hot_sample(self, pp, ap, tem, mask):
        pp = torch.Tensor(np.reshape(pp, (1, 256, 256)))
        ap = torch.Tensor(np.reshape(ap, (1, 256, 256)))
        tem = torch.Tensor(np.reshape(tem, (1, 512, 512)))
        # if self.transform:
        pp = self.transform(pp)
        ap = self.transform(ap)
        tem = self.transform(tem)
        mask = np.reshape(mask, (1, 256, 256))
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        return pp, ap, tem, mask

    def get_transform(self, grayscale=True, convert=True):
        transform_list = []
        # if grayscale:
        #     transform_list.append(transforms.Grayscale(1))
        if convert:
            # transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize([0.5], [0.5])]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)



