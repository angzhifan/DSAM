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
    print(table)
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
    parser.add_argument("--savedir", dest='savedir', default="./saved_models/",
                        help="Where to save the trained model.")
    parser.add_argument('--loss_type', default='MSE', type=str,
                        help='Type of loss', choices=('MSE', 'Perceptual', 'cross_entropy'))
    parser.add_argument('--model_type', default='GLF', type=str,
                        help='Type of model', choices=('VAEflow', 'GLF'))
    parser.add_argument('--split', default='none', type=str,
                        help='label value to train on, default all labels')

    # Variational Auto Encoder settings:
    parser.add_argument("--num_latent", default=32, type=int, #64
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

    if args.loss_type == 'cross_entropy':
        assert (
                    args.dataset == 'mnist' or args.dataset == 'fashion-mnist'), "Cross entropy should only be used for mnist or Fashion-mnist."

    use_gpu = torch.cuda.is_available()
    print(use_gpu)

    torch.manual_seed(123)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:{}".format(args.device) if use_gpu else "cpu")
    print(device)
    print(args.dataset)
    if args.dataset in ['mask-all']:
        args.image_size = 48
        args.cls = ['1','-1','2','-2','4','-4','5','-5','6','-6','7','-7','8','-8']
    training_loaders = return_data(args)
    assert len(training_loaders) == len(args.cls)

    if args.loss_type == 'MSE':
        recon_loss_fn = nn.MSELoss(reduction='sum')

    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, args.fc_dim), nn.ReLU(), nn.BatchNorm1d(args.fc_dim),
                             nn.Linear(args.fc_dim, args.fc_dim), nn.ReLU(), nn.BatchNorm1d(args.fc_dim),
                             nn.Linear(args.fc_dim, c_out))

    if args.model_type == 'VAEflow':
        modbases = [ConvVAE(args).to(device), ConvVAE(args).to(device)]
    elif args.model_type == 'GLF':
        modbases = [ConvAE(args).to(device), ConvAE(args).to(device)]
    modFlows = []
    for _ in range(len(training_loaders)):
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
        modFlow = ReversibleGraphNet(nodes, verbose=False).to(device)
        modFlows.append(modFlow)

    optimizers1 = []
    schedulers1 = []
    for cls in range(2):
        modbase = modbases[cls]
        optimizer1 = optim.Adam(modbase.parameters(), lr=args.lr)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, args.decay, gamma=0.5, last_epoch=-1)
        optimizers1.append(optimizer1)
        schedulers1.append(scheduler1)
    
    optimizers2 = []
    schedulers2 = []
    for cls in range(len(training_loaders)):
        modFlow = modFlows[cls]
        trainable_parameters = [p for p in modFlow.parameters() if p.requires_grad]
        optimizer2 = torch.optim.Adam(trainable_parameters, lr=1e-5, betas=(0.8, 0.9),
                                      eps=1e-6, weight_decay=2e-5) #2e-5
        for param in trainable_parameters:
            param.data = 0.05 * torch.randn_like(param)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, args.decay, gamma=0.5, last_epoch=-1)
        optimizers2.append(optimizer2)
        schedulers2.append(scheduler2)

    recon_loss_vector = []
    lik_loss_vector = []
    entr_loss_vector = []
    total_loss_vector = []

    print("Base model (VAE or AE) params:")
    count_parameters(modbases[0])
    print("Flow params:")
    count_parameters(modFlows[0])

    args.savedir += args.dataset
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    for epoch in range(args.num_epochs):
        recon_losses = []
        like_losses = []
        log_D = []
        entr_L = []
        total_losses = []
        for cls, training_loader in enumerate(training_loaders):
            modbase = modbases[cls%2]
            modbase.train()
            optimizer1 = optimizers1[cls%2]
            scheduler1 = schedulers1[cls%2]
            modFlow = modFlows[cls]
            modFlow.train()
            optimizer2 = optimizers2[cls]
            scheduler2 = schedulers2[cls]
            with tqdm(total=len(training_loader.dataset)) as progress_bar:   #      
                for batch_idx, (dat, label) in enumerate(training_loader):
                    assert dat.shape[1] == 4
                    mask = dat[:,3:,:,:]
                    dat = dat[:,:3,:,:]
                    #if batch_idx >= 14: break
                    adjust = 1.0 #torch.max(dat.view(-1, args.image_size*args.image_size*3), 1).values.view(-1,1,1,1)
                    dat = torch.tensor(adjust_image(dat.cpu().numpy(), adjust))
                    mask = mask.to(device)
                    dat = dat.to(device)
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    if args.model_type == 'VAEflow':
                        recon_batch, z, mu, logvar = modbase(dat)
                        zhat, logd = modFlow(z)  
                        loss_recon = recon_loss_fn(recon_batch*mask, dat*mask)/mask.mean()
                        # entropy loss
                        en_loss = -torch.mean(torch.sum(logvar, 1)).div(2)
                        loss_ll = gaussian_nice_loglkhd(zhat, device) + logd
                        loss_ll = -loss_ll.mean()
                        log_D.append(torch.mean(logd).item())
                        like_losses.append(loss_ll.item())
                        recon_losses.append(loss_recon.item() / dat.size(0))
                        entr_L.append(en_loss.item())
        
                        total_loss = args.beta * loss_recon / dat.size(0) + loss_ll + en_loss
                    elif args.model_type == 'GLF':
                        recon_batch, z = modbase(dat)
                        loss_recon = recon_loss_fn(recon_batch*mask, dat*mask)/mask.mean()
                        recon_losses.append(loss_recon.item() / dat.size(0))
                        # zhat = modFlow(z.data)
                        # logd = modFlow.log_jacobian(z.data)
                        zhat, logd = modFlow(z.clone().detach())  
                        loss_ll = gaussian_nice_loglkhd(zhat, device) + logd
                        loss_ll = -loss_ll.mean()
                        log_D.append(torch.mean(logd).item())
                        like_losses.append(loss_ll.item())
                        total_loss = args.beta * loss_recon / dat.size(0) + loss_ll
                        
                    total_loss.backward(retain_graph=True)
                    total_losses.append(total_loss.item())
                    optimizer1.step()
                    optimizer2.step()
                    if args.model_type == 'VAEflow':
                        progress_bar.set_postfix(loss=np.mean(recon_losses), logd=np.mean(log_D),
                                             likloss=np.mean(like_losses), entropy_loss=np.mean(entr_L))
                    elif args.model_type == 'GLF':
                        progress_bar.set_postfix(loss=np.mean(recon_losses), logd=np.mean(log_D),
                                             likloss=np.mean(like_losses))
                    progress_bar.update(dat.size(0))
            scheduler1.step()
            scheduler2.step()

        if args.model_type == 'VAEflow':
            print('Train Epoch: {} Reconstruction-Loss: {:.4f} loglikelihood loss: {}  entropy loss: {}'.format(
                epoch, np.mean(recon_losses), np.mean(like_losses), np.mean(entr_L)))
            entr_loss_vector.append(np.mean(entr_L))
        elif args.model_type == 'GLF':
            print('Train Epoch: {} Reconstruction-Loss: {:.4f} loglikelihood loss: {}  log det: {}'.format(
                    epoch, np.mean(recon_losses), np.mean(like_losses),np.mean(log_D)))
        total_loss_vector.append(np.mean(total_losses))
        recon_loss_vector.append(np.mean(recon_losses))
        lik_loss_vector.append(np.mean(like_losses))
        print("Epoch " + str(epoch) + " complete, with loss " + str(np.mean(total_losses)))
        if epoch % 20 == 0 or epoch == 199:
            for cls in range(2):
                modbase = modbases[cls%2]
                torch.save(modbase.state_dict(), os.path.join(args.savedir, args.model_type+'_VAEModel_epo{}_{}'.format(epoch, cls)))
            for cls in range(len(args.cls)):
                modFlow = modFlows[cls]
                torch.save(modFlow.state_dict(), os.path.join(args.savedir, args.model_type+'_flowModel_epo{}_{}'.format(epoch, args.cls[cls])))

    print("Total loss: ")
    plt.plot(total_loss_vector)
    plt.show()
    plt.savefig('total_loss.png')
    for i in total_loss_vector:
        print(i)
    print("Recon loss: ")
    plt.plot(recon_loss_vector)
    plt.show()
    plt.savefig('recon_loss.png')
    print("Likelihood loss: ")
    plt.plot(lik_loss_vector)
    plt.show()
    plt.savefig('lik_loss.png')
    print("Entropy loss: ")
    plt.plot(entr_loss_vector)
    plt.show()
    plt.savefig('entr_loss.png')