import argparse
import os

import numpy as np
import torch
import pathlib
import torch.nn as nn
import torchvision.utils as t_utils
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import RNVPCouplingBlock, PermuteRandom
import pytorch_fid
from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_fid.fid_score import get_activations
from pytorch_fid.fid_score import calculate_activation_statistics
from pytorch_fid.inception import InceptionV3

from .utils.dataset import return_data
# from AE.convAE_infoGAN import ConvAE
from .vae.VAE_infoGAN import ConvVAE
from .AE.convAE_infoGAN import ConvAE
import torchvision.transforms as TF

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms


def __makedir__(args):
    if not os.path.exists(args.savedir+args.dataset+'from_dataset'):
        os.makedirs(args.savedir+'/from_dataset')
    if not os.path.exists(args.savedir+'/sample'):
        os.makedirs(args.savedir+'/sample')

def subnet_fc(c_in, c_out):
    args = default_args()
    return nn.Sequential(nn.Linear(c_in, args.fc_dim), nn.ReLU(), nn.BatchNorm1d(args.fc_dim),
                         nn.Linear(args.fc_dim, args.fc_dim), nn.ReLU(), nn.BatchNorm1d(args.fc_dim), nn.Linear(args.fc_dim,  c_out))

def load_check_point(args,subnet_fc,device):
    
    model_ae = ConvVAE(args).to(device)
    
    nodes = [InputNode(args.num_latent,name='input')]
    
    for k in range(args.nbck):
        nodes.append(Node(nodes[-1],
                          RNVPCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':2.0},
                          name=F'coupling_{k}'))
        nodes.append(Node(nodes[-1],
                          PermuteRandom,
                          {'seed':k},
                          name=F'permute_{k}'))
    
    nodes.append(OutputNode(nodes[-1], name='output'))
    
    model_flow = ReversibleGraphNet(nodes, verbose=False)
    model_flow = model_flow.to(device)
    # change the name to the model you want to test
    if(args.split == 'none'):
        flowModel_name = 'flowModel_epo' + str(args.model_epochs) + '_' + str(args.dataset) + '_' + str(args.fc_dim) + '_' + str(args.nbck)
        ae_name = 'VAEModel_epo'  + str(args.model_epochs) + '_' + str(args.dataset) + '_' + str(args.fc_dim) + '_' + str(args.nbck)
    else:
        flowModel_name = 'flowModel_epo'+str(args.model_epochs) + '_' + str(args.split)
        ae_name = 'VAEModel_epo'+  str(args.model_epochs) + '_' + str(args.split)

    print(flowModel_name)
    state_flow = torch.load(os.path.join(args.model_dir, flowModel_name), map_location = device)
    state_ae = torch.load(os.path.join(args.model_dir, ae_name), map_location = device)
    model_ae.load_state_dict(state_ae)
    model_flow.load_state_dict(state_flow)
    return model_ae,model_flow





class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def fid(X, Y):
    print(sum(torch.isnan(X)))
    print(sum(torch.isnan(Y)))
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    muX = np.mean(X, 0).reshape(X.shape[1], 1)
    muY = np.mean(Y, 0).reshape(Y.shape[1], 1)
    #print(sum(sum(np.isnan(X))))
    #print(sum(sum(np.isnan(Y))))
    CX = np.cov(X.transpose(), bias=True)
    CY = np.cov(Y.transpose(), bias=True)
    print(muX)
    print(muY)
    print(CX)
    print(CY)
    E, V = np.linalg.eigh(CX)
    E = np.clip(E, a_min=0, a_max=None)
    CXh = np.matmul(np.matmul(V, np.diag(np.sqrt(E))), V.transpose())
    U = np.matmul(np.matmul(CXh, CY), CXh)
    E, V = np.linalg.eigh(U)
    E = np.clip(E, a_min=0, a_max=None)
    Uh = np.matmul(np.matmul(V, np.diag(np.sqrt(E))), V.transpose())
    fidout = np.sum((muX - muY) * (muX - muY)) + np.sum(np.diag(CX)) + np.sum(np.diag(CY)) - 2 * np.sum(np.diag(Uh))
    print(np.sum((muX - muY) * (muX - muY)))
    print(np.sum(np.diag(CX)))
    print(np.sum(np.diag(CY)))
    print(-2 * np.sum(np.diag(Uh)))
    print('fid', fidout)
    return fidout

def default_args():
    parser = argparse.ArgumentParser(description='Generate samples to test.')    
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    parser.add_argument("--config_file",type=str, default='',help="config file")
    parser.add_argument('--dset_dir', default='./data/',type=str, help='dataset directory')
    parser.add_argument('--dataset', default='mask-label', type=str, help='dataset name')
    parser.add_argument('--image_size', default=28, type=int, help='image size')
    parser.add_argument('--model_epochs', default=200, type=str, help='which version of the model to use (by # of epochs trained), use multiples of 10')
    parser.add_argument('--num_workers', default=8, type=int, help='dataloader num_workers')
    parser.add_argument('--batch_size', default=250, type=int, help='batch size')
    parser.add_argument('--by_occlusion', default=0, type=int, help='sort detections by occlusion or not')
    parser.add_argument('--remove_small_box', default=0.0, type=float, help="remove detections if it's smaller than remove_small_box")
    parser.add_argument('--process_likelihoods', default=0, type=int, help='process likelihoods or not')
    parser.add_argument("--model_dir", dest='model_dir', default="./saved_models/",
                        help="Where to save the trained model.")
    parser.add_argument('--recon', default='yes', type=str, help='get reconstructions or not, may not be enough memory if images large')
    parser.add_argument('--savedir', default='./samples/', type=str, help='save image address')
    parser.add_argument('--split', default='none', type=str, help='label value to train on, default all labels')
    parser.add_argument("--num_latent",  default=64, type=int, #64 #16
                        help="dimension of latent code z")
    
    parser.add_argument("--fc_dim", dest='fc_dim', default=256, type=int,
                        help="Hidden size of inner layers of nonlinearity of flow")
    parser.add_argument("--num_block", dest='nbck', default=1, type=int, #4 #2
                        help="num of flow blocks")
    
    parser.add_argument('--device', dest = 'device',default=0, type=int, 
                        help='Index of device')

    parser.add_argument('--loss_type', default='MSE', type=str, 
                        help='Type of loss',choices = ('MSE','Perceptual','cross_entropy'))

    args = parser.parse_args()
    
    #__makedir__(args)
    args.model_dir += args.dataset
    if args.dataset in ['mask', 'mask-label', 'mask-small']:
        args.image_size = 50
    return args



if __name__ == "__main__":
    args = default_args()
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.device) if use_gpu else "cpu")
    #device = torch.device("mps:{}".format(args.device))
    
    print(args)
    modAE,modFlow = load_check_point(args,subnet_fc,device)
    modAE.train()
    modFlow.train()

    data_loader_path = args.savedir + args.dataset + '/real'  # data as taken from torch, the data that is used to train the model
    samples_path = args.savedir + args.dataset + '/sample'  # generated samples path
    paths = [data_loader_path, samples_path]
    dims = 2048  # ask on this

    """    path = pathlib.Path(data_loader_path)
    files = sorted([file for file in path.glob('*.{}.png')])

    loader = return_data(args)
    real_data = np.empty([0, 3, 32, 32])

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=args.num_workers)
    loader = return_data(args)
    real_data = np.empty([0, 3, 32, 32])
    for batch_idx, (dat,_) in enumerate(loader):
        real_data = dat#np.append(real_data, dat, axis=0)
        break
    #real_data_tensor = torch.from_numpy(real_data)
    real_data = real_data.to(device)
    print(real_data)
    print(type(real_data))"""
    loader = return_data(args)
    #path = pathlib.Path(data_loader_path)
    #files = sorted([file for file in path.glob('*.{}.png')])

    #block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    #model = InceptionV3([block_idx]).to(device)

    #real_data_tensor = get_activations(files, model, args.batch_size, dims, device, args.num_workers)
    data_list = torch.tensor([])
    sample_list = torch.tensor([])

    data_list=data_list.to(device)
    sample_list=sample_list.to(device)

    for batch_idx, dat in enumerate(loader):
        if batch_idx * args.batch_size < 10000:
            realData = dat[0]  # np.append(real_data, dat, axis=0)
            realData = realData.to(device)

            nc, h, w = realData.shape[1:]

            data_batch_to_append = realData
            data_batch_to_append = data_batch_to_append.reshape((args.batch_size, nc * h * w))
            #print(data_batch_to_append.shape)
            # data_list[batch_idx * 250,:] = data_batch_to_append
            data_list = torch.cat((data_list, data_batch_to_append), dim=0)
            #print(data_list.shape)

            if (args.recon == 'yes'):
                if nc == 1:
                    realData = torch.cat((realData, realData, realData), dim=1)
                for j in range(250):
                    realData[j, :, :, :] = (realData[j, :, :, :] - torch.min(realData[j, :, :, :])) / (
                        torch.max(realData[j, :, :, :] - torch.min(realData[j, :, :, :])))
                realData = realData * 255
                for j in range(realData.size(0)):
                    t_utils.save_image(realData.view(250, 3, h, w)[j, :, :, :],
                                       (args.savedir + args.dataset + '/real/{}.png').format(batch_idx * 250 + j),
                                       normalize=True)

                recon, z, mu, logvar = modAE(realData)
                nc, h, w = recon.shape[1:]
                if nc == 1:
                    recon = torch.cat((recon, recon, recon), dim=1)
                for j in range(250):
                    recon[j, :, :, :] = (recon[j, :, :, :] - torch.min(recon[j, :, :, :])) / (
                        torch.max(recon[j, :, :, :] - torch.min(recon[j, :, :, :])))
                recon = recon * 255
                for j in range(recon.size(0)):
                    if batch_idx * 250 + j < 10000:
                        t_utils.save_image(recon.view(250, 3, h, w)[j, :, :, :],
                                           (args.savedir + args.dataset + '/reconstructions/{}.png').format(
                                               batch_idx * 250 + j), normalize=True)
            if batch_idx % 50 == 0 :
                print(batch_idx)


    #print(type(samples_for_reconstruction))
    #print(samples_for_reconstruction.shape)

    for i in range(20):
        noise = torch.randn(500, args.num_latent).to(device)
        zs = modFlow(noise, rev=True)[0]
        sample = modAE.decode(zs)
        nc, h, w = sample.shape[1:]

        sample_batch_to_append = sample
        sample_batch_to_append = sample_batch_to_append.reshape((500, nc * h * w))
        #print(sample_batch_to_append.shape)
        sample_list = torch.cat((sample_list, sample_batch_to_append), dim=0)
        #print(sample_list.shape)

        if nc == 1:
            sample = torch.cat((sample, sample, sample), dim=1)
        for j in range(500):
            sample[j, :, :, :] = (sample[j, :, :, :] - torch.min(sample[j, :, :, :])) / (
                torch.max(sample[j, :, :, :] - torch.min(sample[j, :, :, :])))
        sample = sample * 255
        for j in range(sample.size(0)):
            t_utils.save_image(sample.view(500, 3, h, w)[j, :, :, :],
                               (args.savedir + args.dataset + '/sample/{}.png').format(j + i * 500), normalize=True)
        print(i)



    #batch = sample[0:250,:,:,:]
    #data_tensor = torch.cat(data_list, dim = 1)
    #sample_tensor = torch.cat(sample_list, dim=1)
    print(data_list.shape)
    print(sample_list.shape)


    #sample_for_fid = batch.reshape((250,3072))
    #real_data_tensor_for_fid = real_data_tensor.reshape((250,3072))
    path = pathlib.Path(samples_path)
    files = sorted([file for file in path.glob('*.{}.png')])

    #samples_tensor = get_activations(files, model, args.batch_size, dims, device, args.num_workers)

    fid(data_list, sample_list)
    #fid(sample_for_fid, sample_for_fid)


    fid = calculate_fid_given_paths(paths, args.batch_size, device, dims)
    print("FID: " + str(fid))




    
        
   
