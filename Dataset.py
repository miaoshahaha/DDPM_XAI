import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import Dataset, DataLoader, Subset
import os


class CUSTOM_DATASET:
    def __init__(self, args, split=True):

        self.curr_dataset = args.dataset
        self.split_class = args.split_class
        self.split_flag = True if args.split_class and split else False
        
    def load_dataset(self, custom_trasform=True):
        if custom_trasform:
            CIFAR10_DatasetTransform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.RandomCrop((26,26)),
                transforms.Resize((32,32)),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.ColorJitter(brightness=0.2),
                #transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
                ])
            
            MNIST_DatasetTransform = transforms.Compose([
                transforms.Resize(32),
                torchvision.transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                ])
            
            Imagenet_DatasetTransform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.CenterCrop((48,48)),
                transforms.Resize((64,64)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        else:
            CIFAR10_DatasetTransform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                ])
            
            MNIST_DatasetTransform = transforms.Compose([
                transforms.Resize(32),
                torchvision.transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                ])
            
            Imagenet_DatasetTransform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                ])
            
            


        if self.curr_dataset == 'MNIST':
            train_dataset = MNIST(root='./MNIST_ds', train=True, download=True, transform=MNIST_DatasetTransform)
            test_dataset = MNIST(root='./MNIST_ds', train=False, download=True, transform=MNIST_DatasetTransform)
            
        elif self.curr_dataset == 'CIFAR10':
            train_dataset = CIFAR10(root='./CIFAR10_ds', train=True, download=True, transform=CIFAR10_DatasetTransform)
            test_dataset = CIFAR10(root='./CIFAR10_ds', train=False, download=True, transform=CIFAR10_DatasetTransform)

        elif self.curr_dataset == 'Imagenet':
            train_dataset = torchvision.datasets.ImageFolder(root='Imagenet/train', transform=Imagenet_DatasetTransform)
            test_dataset = torchvision.datasets.ImageFolder(root='Imagenet/val', transform=Imagenet_DatasetTransform)
            
        elif self.curr_dataset == 'LSUN_bedroom':
            train_dataset = torchvision.datasets.ImageFolder(root='LSUN_bedroom/train', transform=Imagenet_DatasetTransform)
            test_dataset = torchvision.datasets.ImageFolder(root='LSUN_bedroom/val', transform=Imagenet_DatasetTransform)
            
    
        #elif self.curr_dataset == 'CelebA':
            #train_dataset = torchvision.datasets.ImageFolder(root='celeba/train', transform=DatasetTransform)
            #test_dataset = torchvision.datasets.ImageFolder(root='celeba/val', transform=DatasetTransform)
    
        else:
            SyntaxError('Unrecongnize dataaset.')
        """
        elif self.curr_dataset == 'LSUN_cat':
            train_dataset = torchvision.datasets.ImageFolder(root='LSUN_cat/train', transform=Imagenet_DatasetTransform)
            test_dataset = torchvision.datasets.ImageFolder(root='LSUN_cat/val', transform=Imagenet_DatasetTransform)
        """
        
        if self.split_flag:
            train_dataset, test_dataset = self.__split_dataset(train_dataset, test_dataset, self.split_class)


        return train_dataset, test_dataset
    
    def __split_dataset(self, train_dataset, test_dataset, split_class):
        train_idx = []
        test_idx = []
        
        for n, idx in enumerate(train_dataset):
            if idx[1] in set(split_class):
                train_idx.append(n)

        new_train_dataset = Subset(train_dataset, list(train_idx))
        
        for n, idx in enumerate(test_dataset):
            if idx[1] in set(split_class):
                test_idx.append(n)

        new_test_dataset = Subset(test_dataset, list(test_idx))


        return new_train_dataset, new_test_dataset
    
    
class FixedLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, fixed_label):
        self.dataset = dataset
        self.fixed_label = fixed_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, _ = self.dataset[idx]
        return data, self.fixed_label
    
            
# 建立一個打亂後的 dataset view
class ShuffledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
def dataset_transform(transform='32p', noise_data=False):
    """
    Returns a composed transform depending on the specified type.

    Parameters:
        transform (str): Type of transformation, e.g., '32p', '32p_mnist', '64p', '64p_mnist'.

    Returns:
        torchvision.transforms.Compose: Transformation pipeline.
    """
    size_map = {
        '32p': (32, False),
        '32p_mnist': (32, True),
        '64p': (64, False),
        '64p_mnist': (64, True)
    }

    if transform not in size_map:
        raise ValueError(f"Unsupported transform type: {transform}")

    size, grayscale = size_map[transform]
    transform_list = [transforms.Resize((size, size))]

    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
        
    if noise_data:
        transform_list.append(transforms.CenterCrop((size*0.75,size*0.75))),
        transform_list.append(transforms.Resize((size,size))),
        #transform_list.append(transforms.RandomHorizontalFlip(p=0.5)),
        

    transform_list.append(transforms.ToTensor())
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    return transforms.Compose(transform_list)

def create_combined_dataset(args=None, pixel='32'):
    """
    Creates a combined dataset from multiple sources, transforming all datasets to a consistent image size.

    Parameters:
        args: Optional arguments (not currently used).
        pixel (str): Target pixel size, '32' or '64'.

    Returns:
        torch.utils.data.ConcatDataset: Combined dataset with fixed labels.
    """
    dataset_sources = {
        'CIFAR10': {
            'class': CIFAR10,
            'args': {'root': './CIFAR10_ds', 'train': False, 'download': True},
            'label': 1,
            'transform_key': f'{pixel}p'
        },
        'MNIST': {
            'class': MNIST,
            'args': {'root': './MNIST_ds', 'train': True, 'download': True},
            'label': 0,
            'transform_key': f'{pixel}p_mnist'
        },
        'LSUN_bedroom': {
            'class': torchvision.datasets.ImageFolder,
            'args': {'root': os.path.join('LSUN_bedroom', 'train')},
            'label': 2,
            'transform_key': f'{pixel}p'
        },
        'AFHQ': {
            'class': torchvision.datasets.ImageFolder,
            'args': {'root': os.path.join('AFHQ', 'train')},
            'label': 3,
            'transform_key': f'{pixel}p'
        },
        'STL-10': {
            'class': torchvision.datasets.ImageFolder,
            'args': {'root': os.path.join('STL-10', 'train')},
            'label': 4,
            'transform_key': f'{pixel}p'
        },
        'Imagenet': {
            'class': torchvision.datasets.ImageFolder,
            'args': {'root': os.path.join('Imagenet', 'train')},
            'label': 5,
            'transform_key': f'{pixel}p'
        }
    }

    labeled_datasets = []

    for name, config in dataset_sources.items():
        if (name == args.dataset) or (name not in args.test_dataset):
            continue
        transform = dataset_transform(config['transform_key'], noise_data=True)
        dataset = config['class'](transform=transform, **config['args'])
        labeled_dataset = FixedLabelDataset(dataset, config['label'])
        labeled_datasets.append(labeled_dataset)

    combined_dataset = torch.utils.data.ConcatDataset(labeled_datasets)
    return combined_dataset