import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import numpy as np
from vae.VAE_infoGAN import ConvVAE, reparametrize
from AE.convAE_infoGAN import ConvAE
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os

from utils import perceptual_loss
from utils.dataset import adjust_image, inv_adjust_image
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import RNVPCouplingBlock, PermuteRandom
import torchvision.utils as t_utils


# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

def distance_metric(sz, force_l2=False):
    if sz == 32 or sz == 28:
        return perceptual_loss._VGGDistance(3, device)
    elif sz == 64:
        return perceptual_loss._VGGDistance(4, device)


def gaussian_nice_loglkhd(h, device):
    return - 0.5 * torch.sum(torch.pow(h, 2), dim=1) - h.size(1) * 0.5 * torch.log(torch.tensor(2 * np.pi).to(device))


from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    #print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def return_data(args, shuffled=True):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers

    if name.lower() == 'cifar':
        pass
    elif name.lower() == 'mask-all':
        root = '/home/fana/DSA_mask/VAE/VAE_test_images48613.npy'
        root_mask = '/home/fana/DSA_mask/VAE/VAE_test_masks48613.npy'
        root_label = '/home/fana/DSA_mask/VAE/VAE_test_labels48613.npy'
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
    return [train_loader]


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VAE with flow prior or train with GLF")
    parser.add_argument("--dataset", default='mask-all', dest='dataset',
                        choices=('mask-all'),
                        help="Dataset to train the model on.")
    parser.add_argument("--dset_dir", dest='dset_dir', default="./data",
                        help="Where you store the dataset.")
    parser.add_argument("--epochs", dest='num_epochs', default=201, type=int, #101
                        help="Number of epochs to train on.")
    parser.add_argument("--batch_size", dest="batch_size", default=256, type=int, #256  # made it bigger
                        help="Number of examples per batch.")

    parser.add_argument('--device', dest='device', default=0, type=int,
                        help='Index of device')
    parser.add_argument("--savedir", dest='savedir', default="./saved_models/mask-all",
                        help="Where to save the trained model.")
    parser.add_argument('--loss_type', default='MSE', type=str,
                        help='Type of loss', choices=('MSE', 'Perceptual', 'cross_entropy'))
    parser.add_argument('--model_type', default='GLF', type=str,
                        help='Type of model', choices=('VAEflow', 'GLF'))
    parser.add_argument('--split', default='none', type=str,
                        help='label value to train on, default all labels')

    # Variational Auto Encoder settings:
    parser.add_argument("--num_latent", default=64, type=int, #64
                        help="dimension of latent code z")
    parser.add_argument("--image_size", default=50, type=int,
                        help="size of training image")
    parser.add_argument("--beta", dest='beta', default=400, #400,
                        help="weight of reconstruction loss")

    # Flow settings:
    parser.add_argument("--fc_dim", default=256, type=int,
                        help="dimension of FC layer in the flow")
    parser.add_argument("--num_block", default=1, type=int, #4
                        help="number of affine coupling layers in the flow")

    # optimization settings
    parser.add_argument("--lr", default=1e-3, dest='lr', type=float,
                        help="Learning rate for ADAM optimizer.")
    parser.add_argument("--beta1", default=0.8, dest='beta1', type=float,
                        help="beta1 for adam optimizer")
    parser.add_argument("--beta2", default=0.9, dest='beta2', type=float,
                        help="beta2 for adam optimizer")
    parser.add_argument("--decay", default=50, dest='decay', type=float,
                        help="number of epochs to decay the lr by half")
    # run experiments w new params
    parser.add_argument("--num_workers", dest="num_workers", default=8, type=int,
                        help="Number of workers when load in dataset.")
    args = parser.parse_args()
    print(args)

    use_gpu = torch.cuda.is_available()
    print(use_gpu)

    torch.manual_seed(123)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:{}".format(args.device) if use_gpu else "cpu")
    print(args.dataset, args.model_type, device)
    if args.dataset in ['mask-all']:
        args.image_size = 48
        image_dim = 48*48*3
        args.cls = ['1','-1','2','-2','4','-4','5','-5','6','-6','7','-7','8','-8']
    training_loaders = return_data(args)
    assert len(training_loaders) == len(args.cls)

    if args.loss_type == 'MSE':
        recon_loss_fn = nn.MSELoss(reduction='sum')

    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, args.fc_dim), nn.ReLU(), nn.BatchNorm1d(args.fc_dim),
                             nn.Linear(args.fc_dim, args.fc_dim), nn.ReLU(), nn.BatchNorm1d(args.fc_dim),
                             nn.Linear(args.fc_dim, c_out))

    modbases = []
    epoch = 200
    for index in [0,1]:
        if args.model_type == 'VAEflow':
            modbase = ConvVAE(args)
        elif args.model_type == 'GLF':
            modbase = ConvAE(args)
        ae_name = args.model_type + '_VAEModel_epo'+str(epoch)
        state_ae = torch.load(os.path.join(args.savedir, ae_name+'_'+str(index)))
        modbase.load_state_dict(state_ae)
        modbase.eval()
        modbases.append(modbase.to(device))
    modFlows = []
    for index in range(len(training_loaders)):
        nodes = [InputNode(args.num_latent, name='input')]
        for k in range(args.num_block):
            nodes.append(Node(nodes[-1],
                              RNVPCouplingBlock,
                              {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                              name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed': k},
                              name=F'permute_{k}'))
        nodes.append(OutputNode(nodes[-1], name='output'))
        modFlow = ReversibleGraphNet(nodes, verbose=False)
        flowModel_name = args.model_type + '_flowModel_epo'+str(epoch)+'_' + args.cls[index]
        state_flow = torch.load(os.path.join(args.savedir, flowModel_name))
        modFlow.load_state_dict(state_flow)
        modFlow.eval()
        modFlows.append(modFlow.to(device))

    # recon_ll = []
    # gauss_ll = []
    # log_jacob = []
    # entr = []
    ELBOs = np.zeros((len(args.cls)//2+1, len(args.cls)//2+1))
    flow_lls = np.zeros((len(args.cls)//2+1, len(args.cls)//2+1))
    entrs = np.zeros((len(args.cls)//2+1, len(args.cls)//2+1))

    print("Base model (VAE or AE) params:")
    count_parameters(modbases[0])
    print("Flow params:")
    count_parameters(modFlows[0])

    args.savedir += args.dataset
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    for cls1 in range(len(args.cls)//2+1):
        for cls2 in range(len(args.cls)//2+1):
            if cls1==len(args.cls)//2 and cls2==len(args.cls)//2:
                modbase = modbases[1]
                modFlow = modFlows[5]
                training_loader = training_loaders[5]
            elif cls1==len(args.cls)//2:
                modbase = modbases[1]
                modFlow = modFlows[cls2*2+1]
                training_loader = training_loaders[cls2*2]
            elif cls2==len(args.cls)//2:
                modbase = modbases[0]
                modFlow = modFlows[cls1*2]
                training_loader = training_loaders[cls1*2+1]
            else:
                modbase = modbases[0]
                modFlow = modFlows[cls1*2]
                training_loader = training_loaders[cls2*2]
            
            # if cls1==len(args.cls)//2 and cls2==len(args.cls)//2:
            #     modbase = modbases[0]
            #     modFlow = modFlows[1]
            #     training_loader = training_loaders[5]
            # elif cls1==len(args.cls)//2:
            #     modbase = modbases[0]
            #     modFlow = modFlows[1]
            #     training_loader = training_loaders[cls2*2]
            # elif cls2==len(args.cls)//2:
            #     modbase = modbases[0]
            #     modFlow = modFlows[cls1*2]
            #     training_loader = training_loaders[cls1*2+1]
            # else:
            #     modbase = modbases[0]
            #     modFlow = modFlows[cls1*2]
            #     training_loader = training_loaders[cls2*2]
            
            recon_ll_cls = 0.0
            gauss_ll_cls = 0.0
            log_jacob_cls = 0.0
            entr_cls = 0.0
            ELBO_cls = 0.0
            cnt_cls = 0
            for batch_idx, (dat, label) in enumerate(training_loader):
                    assert dat.shape[1] == 4
                    assert dat.shape[2] == dat.shape[3] == 48
                    mask = dat[:,3:,:,:]
                    dat = dat[:,:3,:,:]
    
                    adjust = 1.0 #torch.max(dat.view(-1, args.image_size*args.image_size*3), 1).values.view(-1,1,1,1)
                    dat = torch.tensor(adjust_image(dat.cpu().numpy(), adjust))
                    
                    mask = mask.to(device)
                    dat = dat.to(device)
                    if args.model_type == 'VAEflow':
                        z, mu, logvar = modbase.encode(dat)
                        recon_batch = modbase.decode(mu)
                        latent_ll, logd = 0, 0
                        k = 1
                        for _ in range(k):
                            z1 = reparametrize(mu, logvar).to(device)
                            zhat1, logd1 = modFlow(z1)
                            latent_ll += gaussian_nice_loglkhd(zhat1, device).mean()/k
                            logd += logd1.mean()/k

                        # recon_batch, z, mu, logvar = modbase(dat)
                        # zhat, logd = modFlow(z)  
                        value1 = -0.5*args.beta*image_dim*(((recon_batch-dat)*mask)**2).sum()/mask.sum()-0.5*image_dim*torch.log(torch.tensor(2*np.pi/args.beta).to(device))
                        value2 = latent_ll #gaussian_nice_loglkhd(zhat, device).mean()
                        value3 = logd #logd.mean()
                        value4 = 0.5*logvar.size(1)*(1.0+torch.log(torch.tensor(2*np.pi))).to(device)+torch.mean(0.5*torch.sum(logvar, 1))
                    elif args.model_type == 'GLF':
                        recon_batch, z = modbase(dat)
                        zhat, logd = modFlow(z.clone().detach())  
                        value1 = -0.5*args.beta*image_dim*(((recon_batch-dat)*mask)**2).sum()/mask.sum()-0.5*image_dim*torch.log(torch.tensor(2*np.pi/args.beta).to(device))
                        value2 = gaussian_nice_loglkhd(zhat, device).mean()
                        value3 = logd.mean()
                        value4 = torch.zeros(1).to(device)
                    value5 = value1+value2+value3+value4
                    recon_ll_cls += value1.item()*dat.size(0)
                    gauss_ll_cls += value2.item()*dat.size(0)
                    log_jacob_cls += value3.item()*dat.size(0)
                    entr_cls += value4.item()*dat.size(0)
                    ELBO_cls += value5.item()*dat.size(0)
                    cnt_cls += dat.size(0)
            ELBOs[cls1, cls2] = ELBO_cls/cnt_cls
            flow_lls[cls1, cls2] = (gauss_ll_cls+log_jacob_cls)/cnt_cls
            entrs[cls1, cls2] = entr_cls/cnt_cls
    print(np.round(ELBOs,1))
    print(np.round(flow_lls,1))
    print(np.round(entrs,1))
                        

