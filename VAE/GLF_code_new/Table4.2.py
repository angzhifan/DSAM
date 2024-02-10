import torch
from torch import optim
import torch.nn as nn
import numpy as np
from vae.VAE_infoGAN import ConvVAE
from AE.convAE_infoGAN import ConvAE
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os

from utils import perceptual_loss
from utils.dataset import return_data, adjust_image, inv_adjust_image
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

    recon_ll = []
    gauss_ll = []
    log_jacob = []
    entr = []
    ELBO = []

    print("Base model (VAE or AE) params:")
    count_parameters(modbases[0])
    print("Flow params:")
    count_parameters(modFlows[0])

    args.savedir += args.dataset
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    for epoch in range(1):
        for cls, training_loader in enumerate(training_loaders):
            recon_ll_cls = 0.0
            gauss_ll_cls = 0.0
            log_jacob_cls = 0.0
            entr_cls = 0.0
            ELBO_cls = 0.0
            cnt_cls = 0
            
            modbase = modbases[cls%2]
            modFlow = modFlows[cls]
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
                        recon_batch, z, mu, logvar = modbase(dat)
                        zhat, logd = modFlow(z)  
                        value1 = -0.5*args.beta*image_dim*(((recon_batch-dat)*mask)**2).sum()/mask.sum()-0.5*image_dim*torch.log(torch.tensor(2*np.pi/args.beta).to(device))
                        value2 = gaussian_nice_loglkhd(zhat, device).mean()
                        value3 = logd.mean()
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
                        
            recon_ll.append(recon_ll_cls/cnt_cls)
            gauss_ll.append(gauss_ll_cls/cnt_cls)
            log_jacob.append(log_jacob_cls/cnt_cls)
            entr.append(entr_cls/cnt_cls)
            ELBO.append(ELBO_cls/cnt_cls)
    print([np.round(recon_ll[index],1) for index in range(0,14,2)])
    print([np.round(gauss_ll[index],1) for index in range(0,14,2)])
    print([np.round(log_jacob[index],1) for index in range(0,14,2)])
    print([np.round(entr[index],1) for index in range(0,14,2)])
    print([np.round(ELBO[index],1) for index in range(0,14,2)])
    
    # print([np.round(recon_ll[index],1) for index in range(1,14,2)])
    # print([np.round(gauss_ll[index],1) for index in range(1,14,2)])
    # print([np.round(log_jacob[index],1) for index in range(1,14,2)])
    # print([np.round(entr[index],1) for index in range(1,14,2)])
    # print([np.round(ELBO[index],1) for index in range(1,14,2)])
                        

