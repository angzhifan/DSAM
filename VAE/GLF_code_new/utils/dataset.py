import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import torch
import numpy as np

"""class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (img,index)"""


"""class CustomTensorDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)

        return x"""

def adjust_image(img, factor):
    img = img*1.0/factor
    return img #np.clip(img, 0, 1)
    
def inv_adjust_image(img, factor):
    img = img*factor
    return img #np.clip(img, 0, 1)

def return_data(args, shuffled=True):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers

    if name.lower() == 'cifar':
        root = os.path.join(dset_dir, 'cifar')
        if args.loss_type == "MSE":
            training_data = datasets.CIFAR10(root=root, train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                            ]))
        else:
            training_data = datasets.CIFAR10(root=root, train=True, download=True,
                                        transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ]))
    
    elif name.lower() == 'mnist':
        root = os.path.join(dset_dir, 'mnist')
        if not args.loss_type == 'Perceptual':
            training_data = datasets.MNIST(root=root, download=True, transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          #transforms.Normalize((0.5,), (0.5,))
                                      ]))
        else:
            training_data = datasets.MNIST(root=root, download=True, transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                      ]))
    elif name.lower() == 'fashion-mnist':
        root = os.path.join(dset_dir, 'fmnist')
        if not args.loss_type == 'Perceptual':
            training_data = datasets.FashionMNIST(root=root, download=True, transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          #transforms.Normalize((0.5,), (0.5,))
                                      ]))
        else:
            training_data = datasets.FashionMNIST(root=root, download=True, transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                      ]))    
        
    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        if args.loss_type == 'MSE':
            training_data = datasets.CelebA(root=root, download=True, transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ]))
        else:
            training_data = datasets.CelebA(root=root, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    elif name.lower() == 'mask':
        root = '/home/fana/MaskRCNN/MaskRCNN/VAE/VAE_train_images_nocrop49594.npy'
        if args.loss_type == 'MSE' or args.loss_type == 'cross_entropy':
            data_np = np.load(root)
            data_np = np.moveaxis(data_np, 3,1)
            #print(data_np.shape)
            resize_func = transforms.Resize(48)
            data_tensor = torch.from_numpy(data_np)
            data_resized = resize_func(data_tensor)
            training_data = TensorDataset(data_resized)
            #print(data_np.shape)

        """else:
            training_data = datasets.CelebA(root=root, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                                               [0.229, 0.224, 0.225])]))
            #train_kwargs = {'root':root, 'transform':transform}
            #dset = CustomImageFolder
            #training_data = dset(**train_kwargs)"""
    elif name.lower() == 'mask-bg':
        root = '/home/fana/DSA_mask/VAE/VAE_train_fp118609.npy'
        if args.loss_type == 'MSE' or args.loss_type == 'cross_entropy':
            data_np = np.load(root)
            data_np = np.moveaxis(data_np, 3, 1)
            resize_func = transforms.Resize(48)
            data_tensor = torch.from_numpy(data_np)
            data_resized = resize_func(data_tensor)
            print(data_resized.shape)
            labels_tensor = torch.zeros(data_np.shape[0])
            training_data = TensorDataset(data_resized, labels_tensor)
    
    elif name.lower() == 'mask-label':
        root = '/home/fana/MaskRCNN/MaskRCNN/VAE/VAE_train_images_nocrop49594.npy'
        root_label = '/home/fana/MaskRCNN/MaskRCNN/VAE/VAE_train_labels_nocrop49594.npy'
        if args.loss_type == 'MSE' or args.loss_type == 'cross_entropy':
            data_np = np.load(root)
            data_np = np.moveaxis(data_np, 3, 1)
            #print(data_np.shape)
            resize_func = transforms.Resize(48)
            data_tensor = torch.from_numpy(data_np)
            data_resized = resize_func(data_tensor)

            labels_np = np.load(root_label)
            #labels_tensor = torch.from_numpy(labels_np)

            if(args.split != 'none'):
                splitVals = []
                if(args.split == 'cars'):
                    splitVals = [4,5,6,7]
                elif (args.split == 'bikes'):
                    splitVals = [1,8]
                elif (args.split == 'person'):
                    splitVals = [2]
                else:
                    for i in args.split:
                        splitVals.append(int(i))
                indices = []
                for i in range(len(labels_np)):
                    if(labels_np[i] in splitVals):
                        indices.append(True)
                    else:
                        indices.append(False)
                indices = np.array(indices)
                labels_np_split = labels_np[indices]
                print(labels_np_split)
                data_resized_split = data_resized[indices]
            else:
                labels_np_split = labels_np
                data_resized_split = data_resized

            print(np.unique(labels_np))
            print(labels_np_split)
            print(data_resized_split.shape)

            labels_tensor_split = torch.from_numpy(labels_np_split)
            training_data = TensorDataset(data_resized_split, labels_tensor_split)
            # print(data_np.shape)

    elif name.lower() == 'mask-all':
        root = '/home/fana/DSA_mask/VAE/VAE_train_images49594.npy'
        root_mask = '/home/fana/DSA_mask/VAE/VAE_train_masks49594.npy'
        root_label = '/home/fana/DSA_mask/VAE/VAE_train_labels49594.npy'
        root_bg = '/home/fana/DSA_mask/VAE/VAE_train_bg95456.npy' 
        data_np = np.load(root)
        data_np = np.moveaxis(data_np, 3, 1)
        mask_np = np.load(root_mask)
        labels_np = np.load(root_label)
        bg_np = np.load(root_bg)
        bg_np = np.moveaxis(bg_np, 3, 1)
        resize_func = transforms.Resize(48)
        bg_tensor = torch.from_numpy(bg_np)
        bg_resized = resize_func(bg_tensor)
        
        splitVals = sorted(np.unique(labels_np))
        train_loaders = []
        if args.loss_type == 'MSE' or args.loss_type == 'cross_entropy':
            print("data_np.shape:", data_np.shape, mask_np.shape)
            data_tensor = torch.from_numpy(data_np)
            data_resized = resize_func(data_tensor)
            for i in splitVals:
                labels_np_split = labels_np[labels_np==i]
                data_resized_split = data_resized[labels_np==i]
                mask_np_split = np.moveaxis(np.expand_dims(mask_np[labels_np==i], 3), 3, 1)
                data_resized_split = torch.cat((data_resized_split, torch.from_numpy(mask_np_split)), 1)
                splitsize = labels_np_split.shape[0]
                print(i, splitsize, labels_np_split.shape, data_resized_split.shape)

                labels_tensor_split = torch.from_numpy(labels_np_split)
                training_data = TensorDataset(data_resized_split, labels_tensor_split)
                train_loader = DataLoader(training_data,
                            batch_size=batch_size,
                            shuffle=shuffled,
                            num_workers=num_workers,
                            drop_last=False)
                train_loaders.append(train_loader)

                bg_indices = np.random.choice(bg_resized.shape[0], splitsize, replace=False)
                print(bg_resized.shape[0], splitsize)
                bg_data = bg_resized[bg_indices,:,:,:]
                bg_data = torch.cat((bg_data, torch.ones(splitsize,1,48,48)), 1)
                bg_data = TensorDataset(bg_data, -torch.ones(splitsize)*i)
                train_loader = DataLoader(bg_data,
                            batch_size=batch_size,
                            shuffle=shuffled,
                            num_workers=num_workers,
                            drop_last=False)
                train_loaders.append(train_loader)
        return train_loaders
    elif name.lower() == 'mask-small':
        root = '/home/fana/MaskRCNN/MaskRCNN/VAE/VAE_train_images_nocrop1000.npy'
        if args.loss_type == 'MSE' or args.loss_type == 'cross_entropy':
            data_np = np.load(root)
            data_np = np.moveaxis(data_np, 3,1)
            #print(data_np.shape)
            resize_func = transforms.Resize(48)
            data_tensor = torch.from_numpy(data_np)
            data_resized = resize_func(data_tensor)
            training_data = TensorDataset(data_resized)
            #print(data_np.shape)
    else:
        raise NotImplementedError
    train_loader = DataLoader(training_data,
                            batch_size=batch_size,
                            shuffle=shuffled,
                            num_workers=num_workers,
                            drop_last=True)
    return [train_loader]

