import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import Dataset, DataLoader, Subset


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
                transforms.Resize((32,32)),
                transforms.CenterCrop((26,26)),
                transforms.Resize((32,32)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            
        if self.curr_dataset == 'CIFAR10':
            train_dataset = CIFAR10(root='./CIFAR10_ds', train=True, download=True, transform=CIFAR10_DatasetTransform)
            test_dataset = CIFAR10(root='./CIFAR10_ds', train=False, download=True, transform=CIFAR10_DatasetTransform)

        elif self.curr_dataset == 'Imagenet':
            train_dataset = torchvision.datasets.ImageFolder(root='Imagenet/train', transform=Imagenet_DatasetTransform)
            test_dataset = torchvision.datasets.ImageFolder(root='Imagenet/val', transform=Imagenet_DatasetTransform)

        #elif self.curr_dataset == 'CelebA':
            #train_dataset = torchvision.datasets.ImageFolder(root='celeba/train', transform=DatasetTransform)
            #test_dataset = torchvision.datasets.ImageFolder(root='celeba/val', transform=DatasetTransform)
    
        else:
            SyntaxError('Unrecongnize dataaset.')
        
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