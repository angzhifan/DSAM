import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import math
import random
import time
import datetime
import json
import netw
import sys
import cv2
import numpy as np
import pandas as pd
import warnings
import pycocotools.mask as mask_utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from pprint import pprint

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection import PanopticQuality

from VAE.utils import *
from VAE.GLF_code_new.generate_sample import *
from VAE.GLF_code_new.utils.dataset import adjust_image, inv_adjust_image
from VAE.GLF_code_new.vae.VAE_infoGAN import reparametrize
from netw import initialize_model, network
STEP = 0
warnings.filterwarnings("ignore", category=UserWarning)

model_type = "DPT_Hybrid"  
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
transform_img = T.Resize(48)


def map_label(x):
    return x

def gaussian_nice_loglkhd(h, device):
    return - 0.5 * torch.sum(torch.pow(h, 2), dim=1) - h.size(1) * 0.5 * torch.log(torch.tensor(2 * np.pi).to(device))

def polys_to_mask(polygons, height, width):
	rles = mask_utils.frPyObjects(polygons, height, width)
	rle = mask_utils.merge(rles)
	mask = mask_utils.decode(rle)
	return mask

remove_small_box = 0
mask_thres = 0.5
img_path = '/home/fana/data/KITTI/testing/image_2/'
f = open('/home/fana/data/KITTI_AMODAL_DATASET/grouped_data_test.json')
annotate = json.load(f)
f.close()
print("len(annotate):", len(annotate))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_torch_fn_mp(fn, world_size, args):
    """Run fn, passing it:
    0) rank
    1) world_size
    2) args
    """
    mp.spawn(fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

def update_whole_recon(canvas, single_recon, upperleft):
    canvas_size = canvas.shape
    single_recon_strip = single_recon[max(0, -upperleft[0]):,max(0, -upperleft[1]):,:]
    upperleft_strip = (max(0, upperleft[0]), max(0, upperleft[1]))
    l1, l2, _ = single_recon_strip.shape
    indicator = np.expand_dims(canvas[upperleft_strip[0]:(upperleft_strip[0]+l1), upperleft_strip[1]:(upperleft_strip[1]+l2), :].sum(2), axis=2)
    l1, l2, _ = indicator.shape
    patch = single_recon_strip[:l1,:l2,:]*(indicator==0) + canvas[upperleft_strip[0]:(upperleft_strip[0]+l1), upperleft_strip[1]:(upperleft_strip[1]+l2), :]
    canvas[upperleft_strip[0]:(upperleft_strip[0]+l1), upperleft_strip[1]:(upperleft_strip[1]+l2), :] = patch
    return canvas


def get_depth(input_image):
    height, width, n_channels = input_image.shape
    assert n_channels == 3
    input_batch = transform(input_image)
    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=input_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = depth.cpu().numpy()
    return depth

def get_keep(labels, boxes, areas, small_box_thres):
    keep = []
    for box_index in range(len(labels)):
        #if (boxes[box_index][2]-boxes[box_index][0])*(boxes[box_index][3]-boxes[box_index][1])>=small_box_thres:
        if areas[box_index] >=small_box_thres:
            keep.append(box_index)
    return keep


def find_code_for_target(target, pred_mask_resized, modAE, modFlow, device, args):
    mu_code1 = Variable(torch.zeros(1, modAE.nz)).to(device)
    mu_code1.requires_grad_(True)
    optimizer_mu = torch.optim.Adam([mu_code1], lr=args.mu_lr)
    if args.model_type =='VAEflow': 
        log_var_code1 = Variable(torch.zeros(1, modAE.nz)).to(device)
        log_var_code1.requires_grad_(True)
        optimizer_log_var = torch.optim.Adam([log_var_code1], lr=args.log_sigma_lr)
    for j in range(args.test_latent_iternum):
        optimizer_mu.zero_grad()
        if args.model_type =='VAEflow': 
            optimizer_log_var.zero_grad()        
            z = modAE.reparameterize(mu_code1, log_var_code1)
            zhat, logd = modFlow(z)  
            en_loss = -torch.mean(torch.sum(log_var_code1, 1)).div(2)
            loss_ll = gaussian_nice_loglkhd(zhat, device) + logd
            loss_ll = -loss_ll.mean()
            recon = modAE.decode(z)
            test_minus_mu = (target - recon)* pred_mask_resized
            total_loss = (test_minus_mu*test_minus_mu).sum()/(args.sigma * args.sigma * pred_mask_resized.mean()) + loss_ll + en_loss
            total_loss.backward(retain_graph=True)
            optimizer_mu.step()
            optimizer_log_var.step()
        elif args.model_type =='GLF': 
            z = mu_code1
            zhat, logd = modFlow(z)  
            loss_ll = gaussian_nice_loglkhd(zhat, device) + logd
            loss_ll = -loss_ll.mean()
            recon = modAE.decode(z)
            test_minus_mu = (target - recon)* pred_mask_resized
            total_loss = (test_minus_mu*test_minus_mu).sum()/(args.sigma * args.sigma * pred_mask_resized.mean()) + loss_ll
            total_loss.backward(retain_graph=True)
            optimizer_mu.step()
    if args.model_type =='VAEflow': 
        return mu_code1, log_var_code1
    elif args.model_type =='GLF': 
        return mu_code1


def single_reconstruction(vae, pred_label, target, device, args, pred_mask_resized = None):
    target = torch.tensor(target).squeeze()
    target = torch.moveaxis(target, 2, 0)
    assert target.shape[0]==3
    
    if int(pred_label)>0:
        modAE = vae['modAE_ob'].to(device)
    else:
        modAE = vae['modAE_bg'].to(device)
    modFlow = vae[str(pred_label)].to(device)
    target = target.unsqueeze(0)
    if args.model_type =='VAEflow': 
        if args.latent_opt == 0 or pred_mask_resized is None:
            z, mu, log_var = modAE.encode(target.to(device))
        else: 
            mu, log_var = find_code_for_target(target.to(device), torch.tensor(pred_mask_resized).to(device), modAE, modFlow, device, args)
        recon = modAE.decode(mu)
        latent_ll, logd = 0, 0
        k = 100
        for _ in range(k):
            z1 = reparametrize(mu, log_var).to(device)
            zhat1, logd1 = modFlow(z1)
            latent_ll += gaussian_nice_loglkhd(zhat1, device).cpu()/k
            logd += logd1.cpu()/k
    elif args.model_type =='GLF': 
        if args.latent_opt == 0 or pred_mask_resized is None:
            mu = modAE.encode(target.to(device))
        else:
            mu = find_code_for_target(target.to(device), torch.tensor(pred_mask_resized).to(device), modAE, modFlow, device, args)
        recon = modAE.decode(mu)
        zhat, logd = modFlow(mu)
        latent_ll = gaussian_nice_loglkhd(zhat, device).cpu()
        logd = logd.cpu()
    recon = torch.moveaxis(recon.squeeze().reshape(3, target.shape[2], target.shape[3]), 0, 2)
    if args.model_type =='VAEflow': 
        return recon.cpu().detach().numpy(), mu, log_var, latent_ll, logd
    elif args.model_type =='GLF': 
        return recon.cpu().detach().numpy(), mu, 0.0, latent_ll, logd

def whole_reconstruction(vae, image, I, ReconDict, S, D, B, scores, occlusion_scores, bb, masks, pred_labels, device, args):
    """Given a set of occluded individual component selections and a composite image,
    reconstruct each of these selections into the unoccluded full components.
    :param vae: dictionary of GLF or VAE
    :param image: original image to run the algorithm on
    :param I: original image confined to the union of the predicted object masks
    :param ReconDict: Hashmap individually reconstructed images and mu, log_var, latent_ll, logd
    :param S, D, B: as described in the algorithm
    :param bb: (list[tuple]) bounding boxes of individual reconstructions
    :param occlusion_scores: (list[float]) occlusion scores of individual reconstructions
    :param pred_labels: (list[int]) labels of individual component reconstructions
    :returns: the loss L and the singly reconstructed components ReconDict.
    """
    global STEP
    SnB = S+B
    assert image.shape[2] == 3 

    # occlusion scores of objects selected
    occ_score_selected = [occlusion_scores[i] for i in SnB]
    # get the indices of the sorted occlusion scores in descending order
    sorted_occ_score_locations = np.argsort(occ_score_selected)[::-1] 
    canvas = np.zeros_like(image)
    Loss = 0

    # process all the selected detections
    for occ_score_index in sorted_occ_score_locations: 
        index = SnB[occ_score_index]
        if index in S:
            operation = 'S'
        else:
            operation = 'B'
        bb_i = bb[index]
        pred_mask_all = (masks[index].squeeze() > mask_thres).to(dtype=torch.int32) #binarize the predicted object mask
        target_size = (bb_i[3] - bb_i[1], bb_i[2] - bb_i[0])
        l = max(target_size)
        x1, y1 = (l - target_size[0]) // 2, (l - target_size[1]) // 2
        upperleft = [bb_i[1] - x1, bb_i[0] - y1]  #the upper left corner of the image context
        pad_params = ((x1, l-(x1 + bb_i[3] - bb_i[1])), (y1, l-(y1 + bb_i[2] - bb_i[0])), (0,0))

        # do single reconstruction if it hasn't been done
        if (index, operation) not in ReconDict:
            # Get the single reconstruction target
            target = image[max(upperleft[0], 0):(upperleft[0] + l), max(upperleft[1], 0):(upperleft[1] + l), :]
            pred_mask = pred_mask_all[max(upperleft[0], 0):(upperleft[0] + l), max(upperleft[1], 0):(upperleft[1] + l)].unsqueeze(2)
            if target.shape[0] < l or target.shape[1] < l: #if the image context is not entirely within the image border
                target = np.pad(image[bb_i[1]:bb_i[3], bb_i[0]:bb_i[2], :], pad_params, mode='constant') 
                pred_mask = np.expand_dims(np.pad(pred_mask_all[bb_i[1]:bb_i[3], bb_i[0]:bb_i[2]], (pad_params[0], pad_params[1]), mode='constant').astype('float32'), axis=2)
                if args.latent_opt == 0:
                    # add the single reconstruction to the ReconDict
                    ReconDict[(index, operation)] = (target*pred_mask, torch.zeros(args.latent_space_dimension), torch.zeros(args.latent_space_dimension), torch.tensor([0.0]), torch.tensor([0.0]))
                else:                    
                    target = cv2.resize(target, dsize=(48, 48), interpolation=cv2.INTER_LINEAR)
                    adjust = target.max()
                    target = adjust_image(target, adjust)
                    pred_mask_resized = cv2.resize(pred_mask, dsize=(48, 48), interpolation=cv2.INTER_LINEAR)

                    if operation == 'S':
                        pred_label = pred_labels[index]
                    else:
                        pred_label = str(-pred_labels[index])
                    single_recon, mu, log_var, latent_ll, logd = single_reconstruction( 
                        vae,
                        pred_label,
                        target,
                        device,
                        args,
                        pred_mask_resized
                    )                        
                    single_recon = cv2.resize(single_recon, dsize=(int(l), int(l)), interpolation=cv2.INTER_LINEAR)
                    single_recon = inv_adjust_image(single_recon, adjust)
                    single_recon = single_recon * pred_mask
    
                    ReconDict[(index, operation)] = (single_recon, mu, log_var, latent_ll, logd)
            else:    
                pred_mask = pred_mask.view((l, l, 1))
    
                sr_region = canvas[max(upperleft[0], 0):(upperleft[0] + l), max(upperleft[1], 0):(upperleft[1] + l), :]
                blank = (sr_region.sum(2) <= 0)[:,:,np.newaxis]
                # if we have fewer than args.occlusion_number_of_pixels pixels above the occlusion threshold, no reconstruction
                if ((target*pred_mask.cpu().numpy()*blank).sum(2)>0).sum() < args.occlusion_number_of_pixels:
                    ReconDict[(index, operation)] = (np.zeros((l, l, 3)), torch.zeros(args.latent_space_dimension), torch.zeros(args.latent_space_dimension), torch.tensor([0.0]), torch.tensor([0.0]))
                    if args.draw:
                        print('Occlusion too large for image', index, pred_labels[index], operation,
                              (target.sum(2) > args.occlusion_thresh).sum())
                else:
                    # if args.draw:
                        # print(target.shape)
                        
                        # plt.imshow(target)
                        # plt.show()
                    
                    target = cv2.resize(target, dsize=(48, 48), interpolation=cv2.INTER_LINEAR)
                    adjust = target.max()
                    target = adjust_image(target, adjust)
                    
                    if args.draw:
                        plt.imshow(target.reshape((48, 48, 3)))
                        plt.show()
                        print('single reconstruction:', pred_labels[index], operation)
                        target_copy = target.copy()
                    
                    if operation == 'S':
                        pred_label = pred_labels[index]
                    else:
                        pred_label = str(-pred_labels[index])
                    single_recon, mu, log_var, latent_ll, logd = single_reconstruction(
                        vae,
                        pred_label,
                        target,
                        device,
                        args
                    ) 
                    
                    # if args.draw:
                    #     plt.imshow(single_recon)
                    #     plt.show()
                    
                    single_recon = cv2.resize(single_recon, dsize=(int(l), int(l)), interpolation=cv2.INTER_LINEAR)
                    single_recon = inv_adjust_image(single_recon, adjust)
                    single_recon = single_recon * pred_mask.numpy()
                    if args.draw:
                        plt.imshow(single_recon)
                        plt.show()
                        target_copy = cv2.resize(target_copy, dsize=(int(l), int(l)), interpolation=cv2.INTER_LINEAR)
                        print("L2 in single recon:", (((target_copy-single_recon)*pred_mask.numpy())**2).sum())
    
                    # add the single reconstruction to the ReconDict
                    ReconDict[(index, operation)] = (single_recon, mu, log_var, latent_ll, logd)
        (single_recon, mu, log_var, latent_ll, logd) = ReconDict[(index, operation)]
        loss_ll = - latent_ll
        loss_logd = -logd
        if args.model_type =='VAEflow': 
            loss_en = -((1+math.log(2*math.pi))*torch.ones_like(mu) + log_var).sum().cpu()/2
        elif args.model_type =='GLF': 
            loss_en = 0.0 
        if args.draw:
            print(operation, 'loss_ll, loss_logd, loss_en:', loss_ll, loss_logd, loss_en, 2*args.sigma*args.sigma*(loss_ll+ loss_logd + loss_en))
        Loss += 2*args.sigma*args.sigma*(loss_ll+ loss_logd + loss_en)
        canvas = update_whole_recon(canvas, single_recon, upperleft)
    Loss = Loss+((I-canvas)**2).sum()      
    if args.draw:
        print('L2 improvement:', (I**2).sum()-((I-canvas)**2).sum())
        fig, ax = plt.subplots(figsize=(37.5, 124.2))
        ax.imshow(canvas)
        plt.show()
        plt.close()
    return Loss, ReconDict


def detections_selection(vae, image, I, scores, bb, occlusion_scores, pred_labels, masks, device, args):
    """Main entry point for the detections selection algorithm.
    :param vae: (torch.nn.module) VAE decoder used to train latent codes
    :param image: original image to run the algorithm on
    :param I: original image confined to the union of the predicted object masks
    :param scores: (list[float]) detection confidence scores
    :param bb: (list[tuple[int]]) estimated bounding boxes
    :param occlusion_scores: (list[float]) occlusion scores of detections
    :param pred_labels: (list[int]) predicted labels of detections
    :param device: (str) execution device
    :param args: (namespace)
    :param image_index: (int) index of image
    :return: (list[int]) selected objects, lowest_loss
    """
    S, D, B = [], [], []
    ReconDict = {}
    lowest_loss = np.sum((I - np.zeros_like(I)) * (I - np.zeros_like(I)))
    if args.draw:
        print('loss before DSA', lowest_loss.item())

    # for each prediction, process the detections by objectness score...
    for box_index in range(len(pred_labels)):
        # if args.draw:
        #     print('No.', box_index, bb[box_index], pred_labels[box_index])
        if args.process_likelihoods:
            # run whole reconstruction algorithm on image if the new box is selected
            temp_S, temp_D, temp_B = S.copy(), D.copy(), B.copy()
            temp_S.append(box_index)
            L1, ReconDict = whole_reconstruction(
                vae,
                image, I,
                ReconDict,
                temp_S, temp_D, temp_B,
                scores, occlusion_scores, bb, masks, pred_labels,
                device,
                args
            )

            # run whole reconstruction algorithm on image if the new box is chosen to be background
            temp_S, temp_D, temp_B = S.copy(), D.copy(), B.copy()
            temp_B.append(box_index)
            L2, ReconDict = whole_reconstruction(
                vae,
                image, I,
                ReconDict,
                temp_S, temp_D, temp_B,
                scores, occlusion_scores, bb, masks, pred_labels,
                device,
                args
            )

            # compare the losses from different selection configurations and choose the configuration which minimizes the loss
            if lowest_loss <= min(L1, L2):
                pass
            elif L1 <= L2:
                S.append(box_index)
                lowest_loss = L1
            else:
                B.append(box_index)
                lowest_loss = L2
            if args.draw:
                print('****selected', S, '*****labels', [pred_labels[s] for s in S], 'with object',
                      L1.item(), 'with bg', L2.item(), 'lowest', lowest_loss.item())
        else:
            S += [box_index]
    return S, D, B


def soft_nms_Gaussian_penalty(iou1, sigma=0.5):
    return np.exp(-(iou1 ** 2) / sigma)


def soft_nms(labels, scores, bb, occlusion_scores):
    l_pred = len(labels)
    visited = set()
    for i in range(l_pred - 1):
        # find the one with max score
        max_ind, max_score = -1, -1
        for j in range(l_pred):
            if j not in visited and scores[j] > max_score:
                max_ind, max_score = j, scores[j]
        visited.add(max_ind)
        # update the scores
        for j in range(l_pred):
            if j not in visited:
                iou_j = iou(bb[max_ind], bb[j])
                Gaussian_penalty = soft_nms_Gaussian_penalty(iou_j)
                scores[j] = scores[j] * Gaussian_penalty
    ii = np.flip(np.argsort(scores))
    return labels[ii], scores[ii], bb[ii], occlusion_scores[ii]

def get_pano(height, width, labels, masks, pano_height = 376, pano_width = 1242):
    pano = np.zeros((height, width, 2))
    for k in range(len(labels)):
        mask = masks[k]*(pano[:,:,0]==0)
        pano[:,:,0] = mask*labels[k] + pano[:,:,0]
        pano[:,:,1] = mask*k + pano[:,:,1]
    tmp_height, tmp_width = pano.shape[0], pano.shape[1]
    pad_hw = (((pano_height-tmp_height)//2, (pano_height-tmp_height+1)//2), ((pano_width-tmp_width)//2 , (pano_width-tmp_width+1)//2), (0,0))
    padded_pano = np.pad(pano, pad_hw)
    assert (padded_pano.shape[0], padded_pano.shape[1]) == (pano_height, pano_width)
    return padded_pano


def get_truth_detections(maskrcnn_model, img_name, device, args):
    input_image = mpimg.imread(img_path+img_name)
    height, width, n_channels = input_image.shape
    assert n_channels==3
    fig, ax = plt.subplots(figsize=(37.5, 124.2))
    depth = get_depth(input_image)
    plt.imshow(depth)
    plt.close()
    
    # get the ground truth
    annotation = annotate[img_name]
    area = torch.as_tensor(annotation['i_area'], dtype=torch.float32)
    labels = annotation['category_id']
    boxes = annotation['i_bbox']
    masks_area = []
    for poly in annotation["i_segm"]:
        mask = polys_to_mask(poly, height, width)
        masks_area.append(mask.sum())
    
    # remove small objects if necessary
    keep = get_keep(labels, boxes, masks_area, args.remove_small_box)
    labels = [map_label(labels[box_index]) for box_index in keep]
    boxes = [boxes[box_index] for box_index in keep]
    masks = [annotation["i_segm"][box_index] for box_index in keep]

    # get the predictions
    input_image_torch = torch.tensor(input_image.transpose((2, 0, 1)), dtype=torch.float32)
    results = maskrcnn_model([input_image_torch.to(device)])
    scores = results[0]['scores'].cpu()
    bb = results[0]['boxes'].to(torch.int32).cpu()
    pred_labels = [map_label(int(label)) for label in results[0]['labels'].cpu()]
    pred_masks = list(results[0]['masks'].cpu().detach())
    occlusion_scores = []
    predicted_masks_area = []
    for box_index, pred_mask in enumerate(results[0]['masks'].cpu().detach().numpy()):
        pred_mask = (pred_mask.squeeze() > mask_thres).astype(np.int32)
        avg_depth = (pred_mask * depth).sum() / max(1, pred_mask.sum())
        occlusion_scores.append(avg_depth)
        predicted_masks_area.append(pred_mask.sum())
        
    # remove small boxes if necessary
    keep = get_keep(pred_labels, bb, predicted_masks_area, args.remove_small_box)
    keep = np.array(keep)
    scores = scores[keep]
    occlusion_scores = np.array([occlusion_scores[box_index] for box_index in keep])
    bb = bb[keep,:]
    pred_labels = [pred_labels[box_index] for box_index in keep]
    pred_masks = [pred_masks[box_index] for box_index in keep]

    # predicted and ground truth masks
    all_masks = np.zeros((height, width, 3))
    for poly in masks:
        mask = polys_to_mask(poly, height, width)
        all_masks[:,:,0] += mask
    for i,pred_mask in enumerate(pred_masks):
        pred_mask = (pred_mask.squeeze()>mask_thres).detach().cpu().numpy().astype(np.int32)
        all_masks[:,:,2] += pred_mask
    all_masks = np.minimum(1, all_masks)
    
    return depth, labels, boxes, masks, scores, occlusion_scores, bb, pred_labels, pred_masks, all_masks 


def draw_image(maskrcnn_model, img_name, device, args):
    """Runs the input image through the RCNN model, show the number of predictions,
    the prediction labels, and the occlusion scores.
    :param maskrcnn_model: Mask rcnn model
    :param img_name: img_name
    """
    input_image = mpimg.imread(img_path+img_name)
    height, width, n_channels = input_image.shape
    assert n_channels==3
    # fig, ax = plt.subplots(figsize=(37.5,124.2))
    # ax.imshow(input_image)
    # plt.show()
    # plt.close()
    
    
    depth, labels, boxes, masks, scores, occlusion_scores, bb, pred_labels, pred_masks, all_masks = get_truth_detections(maskrcnn_model, img_name, device, args)
    print('ground truth labels:', labels)
    print('pred_labels, scores, occlusion_scores:', pred_labels, scores, occlusion_scores)
    
    # plot predicted and ground truth boxes
    fig, ax = plt.subplots(figsize=(37.5,124.2))
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    for i,box in enumerate(bb):
        if scores[i]>args.detect_thresh:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=3, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
    ax.imshow(input_image)
    plt.show()
    plt.close()

    # plot predicted and ground truth masks
    fig, ax = plt.subplots(figsize=(37.5,124.2))
    for i,pred_mask in enumerate(pred_masks):
        avg_depth = (pred_mask*depth).sum()/pred_mask.sum()
        box = bb[i]
        if scores[i]>args.detect_thresh:
            ax.text(box[0], box[1], str(np.round(avg_depth,2)), color='white', fontsize='xx-large')
    ax.imshow(all_masks)
    plt.show()
    plt.close()

    # the original image confined within the union of the predicted masks
    # I = input_image*np.expand_dims(all_masks[:,:,2], axis=2)
    # fig, ax = plt.subplots(figsize=(37.5,124.2))
    # ax.imshow(I)
    # plt.show()
    # plt.close()
    

def test_example(img_name, maskrcnn_model, vae, device, args):
    """Run the detection selection algorithm on the test image with img_name.
    :param img_name: (str) name of test image
    :param maskrcnn_model: RCNN model used for detection
    :param vae: The VAE of GLF used.
    :param args: (namespace)
    :return: (list[int]) predicted object indices, in order
    """
    # fetch image
    input_image = mpimg.imread(img_path+img_name)
    height, width = input_image.shape[0], input_image.shape[1]

    if args.draw:
        draw_image(maskrcnn_model, img_name, device, args)

    depth, labels, boxes, masks, scores, occlusion_scores, bb, pred_labels, pred_masks, all_masks = get_truth_detections(maskrcnn_model, img_name, device, args)
    I = input_image*np.expand_dims(all_masks[:,:,2], axis=2)
    
    if args.soft_nms:
        pred_labels, scores, bb, occlusion_scores = soft_nms(np.array(pred_labels), scores.cpu().detach().numpy(),
                                                             bb.cpu().detach().numpy(), np.array(occlusion_scores))

    # run detections selection algorithm on image
    S, D, B = detections_selection(
        vae,
        input_image, I,
        scores,
        bb,
        occlusion_scores,
        pred_labels, pred_masks,
        device,
        args
    )

    # get panoptic segmentation
    pano_target = get_pano(height, width, labels, [polys_to_mask(mask, height, width) for mask in masks])
    labels = torch.as_tensor(labels, dtype=torch.int64)
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    targets_dict = {'boxes': boxes , 'labels': labels, 'pano': torch.tensor(pano_target)}
    
    S = np.array(S)
    if args.by_occlusion:
        assert len(occlusion_scores) == len(pred_labels)
        if len(S)>0:
            ii = np.flip(np.argsort(occlusion_scores[S]))
            S_sorted = S[ii]
        else:
            S_sorted = [] 
        pano_pred = get_pano(height, width, [pred_labels[box_index] for box_index in 
                    S_sorted], [(pred_masks[box_index].squeeze().cpu().detach().numpy()>0.5).astype(np.int32) for box_index in S_sorted])
    else:
        pano_pred = get_pano(height, width, [pred_labels[box_index] for box_index in 
                    S],[(pred_masks[box_index].squeeze().cpu().detach().numpy()>0.5).astype(np.int32) for box_index in S])
    assert pano_pred.ndim==3 and pano_pred.shape[2]==2
    
    if len(S) < len(pred_labels):
        DSA_matters = len(pred_labels)-len(S)
    else:
        DSA_matters = 0 
    print(img_name, "DSA_matters:", len(pred_labels)-len(S))
    if args.draw:
        fig, ax = plt.subplots(figsize=(37.5, 124.2))
        pano = np.concatenate(((pano_pred/pano_pred.max())**(1/2), np.zeros((pano_pred.shape[0],pano_pred.shape[1],1))), axis=2)
        ax.imshow(pano)
        plt.show()
        plt.close()
        fig, ax = plt.subplots(figsize=(37.5, 124.2))
        pano = np.concatenate(((pano_target/pano_target.max())**(1/2), np.zeros((pano_target.shape[0],pano_target.shape[1],1))), axis=2)
        ax.imshow(pano)
        plt.show()
        plt.close()
        print(len(np.unique(pano_pred[:,:,1])), len(np.unique(pano_target[:,:,1])))
    if len(S)>0:
        preds_dict = {'boxes': torch.tensor(bb[S,:]) , 'labels': torch.tensor(np.array(pred_labels)[
                    S]), 'scores': torch.tensor(scores[S]), 'pano': torch.tensor(pano_pred)}
    else:
        preds_dict = {'boxes': torch.tensor([]) , 'labels': torch.tensor([]), 'scores': torch.tensor([]), 'pano': torch.tensor(pano_pred)}
    myPQ = PanopticQuality(things = {1,2,4,5,6,7,8}, stuffs = {0})
    PQ_i = myPQ(preds_dict['pano'], targets_dict['pano'])
    if args.draw:
        print('PQ score:', PQ_i)
    return preds_dict, targets_dict, DSA_matters, PQ_i


def get_flow(args):
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
    return model_flow
    

def read_dgm(rank, args, device):
    if args.run_gpu>=0 or args.run_parallel:
        map_location = f"cuda:{rank}"
    else:
        map_location = f"cpu"

    args_glf = default_args()
    args_glf.model_dir='./VAE/GLF_code_new/saved_models/mask-all'
    print(args_glf)

    if args.model_type =='VAEflow': 
        modAE_ob = ConvVAE(args_glf)
        modAE_bg = ConvVAE(args_glf)
    elif args.model_type =='GLF': 
        modAE_ob = ConvAE(args_glf)
        modAE_bg = ConvAE(args_glf)
    ae_name = args.model_type + '_VAEModel_epo'+  str(args_glf.model_epochs)
    state_ae_ob = torch.load(os.path.join(args_glf.model_dir, ae_name+'_0'))
    state_ae_bg = torch.load(os.path.join(args_glf.model_dir, ae_name+'_1'))
    modAE_ob.load_state_dict(state_ae_ob)
    modAE_ob.eval()
    modAE_bg.load_state_dict(state_ae_bg)
    modAE_bg.eval()
    net = {'modAE_ob':modAE_ob, 'modAE_bg':modAE_bg}
    for cls in ['1','-1','2','-2','4','-4','5','-5','6','-6','7','-7','8','-8']:
        modFlow = get_flow(args_glf)
        flowModel_name = args.model_type + '_flowModel_epo'+str(args_glf.model_epochs) + '_' + cls
        state_flow = torch.load(os.path.join(args_glf.model_dir, flowModel_name))
        modFlow.load_state_dict(state_flow)
        modFlow.eval()
        net[cls] = modFlow
    return net


def read_fasterrcnn(rank, args):
    """Reads in fasterrcnn/maskrcnn model."""
    num_classes = 8 + 1
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, box_score_thresh=args.detect_thresh)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    if args.run_gpu>=0:
        map_location = rank
    elif args.run_parallel:
        map_location=f"cuda:{rank}"
    else:
        map_location = f"cpu"

    checkpoint = torch.load(os.path.join(args.predir,args.faster_rcnn), map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    if args.soft_nms:
        model.roi_heads.nms_thresh = 1.0
    model.eval()
    return model.to(rank)


def my_main(rank, world_size, args):
    """
    Run the detection selection algorithm on the test images
    """
    if args.run_parallel:
        setup(rank, world_size)
    torch.manual_seed(42)

    # ========================================================================
    # read decoder into memory in process executing on rank
    if args.run_parallel:
        vae = read_dgm(rank, args, rank)
    else:
        vae = read_dgm(args.run_gpu, args, device)
    # read fasterrcnn into memory in process executing on rank
    maskrcnn_model = read_fasterrcnn(rank, args)

    start_time = time.time()
    # separate output file for algorithm running on each device
    output_filename = os.path.join(args.dn,'output','output_process_'+str(rank))
    output_file = open(output_filename, "w")

    img_names = os.listdir(img_path)
    args.num_test_images = len(img_names)
    
    # each of the four devices gets a quarter of the images
    n_images_device = int(args.num_test_images / world_size)
    if type(rank) is int:
        img_start = rank * n_images_device
        img_end = n_images_device + rank * n_images_device
    elif args.image_to_process<0:
        img_start = 0
        img_end = n_images_device
    else:
        img_start=args.image_to_process
        img_end=img_start+1
        args.num_test_images=1

    # the results
    PQs = []
    img_ids = []
    DSA_matters_total = 0
    cnt = 0
    for i in range(img_start, img_end):
        print(f"analyzing image {i} out of {args.num_test_images}...")
        if img_names[i] not in annotate:
            continue

        # run detection selection algorithm on image
        preds_dict, targets_dict, DSA_matters, PQ_i = test_example(
            img_names[i],
            maskrcnn_model,
            vae,
            rank,
            args
        )
        
        DSA_matters_total += int(DSA_matters>0)
        PQs.append(PQ_i.numpy())
        img_ids.append(i)
        cnt += 1
        get_errors(output_file, targets_dict['labels'], preds_dict['labels'], i, DSA_matters, PQ_i.numpy())

    # log total runtime
    time_elapsed = time.time() - start_time
    output_file.write("DSA_matters? {} \n".format(DSA_matters_total/cnt))
    # write average PQ to file
    if cnt>0:
        output_file.write(
            'avg PQ of rank {} is {} \n'.format(rank, np.array(PQs).mean())
        )
    output_file.write('{:.0f}s'.format(time_elapsed))
    output_file.close()
    np.save('./temp_output/PQs_'+str(rank)+'.npy', np.array(PQs))
    np.save('./temp_output/img_ids_'+str(rank)+'.npy', np.array(img_ids))




if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    args=get_args()
    
    # get directory of script
    print("args.model_type:", args.model_type, "args.remove_small_box:", args.remove_small_box, "args.by_occlusion:", args.by_occlusion, "args.process_likelihoods:", args.process_likelihoods)

    # parse command line overrides to config file args
    if args.trans_type == 'None':
        args.trans_space_dimension=0

    t1=time.time()
    os.system('rm output/output_process*')
    
    # run on GPU or CPU
    device=None
    if args.run_parallel:
        n_gpus = torch.cuda.device_count()
        print("n_gpus:", n_gpus, "args.run_parallel:", args.run_parallel)
        world_size=np.minimum(n_gpus, args.run_parallel)
        print(f"executing on {world_size} gpus...")
        # run the my_main routine, distributing it across GPUs
        run_torch_fn_mp(
            my_main, world_size, args
        )
    elif args.run_gpu>=0 and torch.cuda.device_count()>0:
        world_size = 1
        device = torch.device("cuda:"+str(args.run_gpu))
        print("executing on gpu",device)
        my_main(device, 1, args)
    else:
        world_size = 1
        device = torch.device("cpu")
        print("executing on cpu...")
        my_main(device, 1, args)
    print('Time',time.time()-t1)

    all_PQs = []
    for rank in range(world_size):
        PQs = np.load('./temp_output/PQs_'+str(rank)+'.npy')
        img_ids = np.load('./temp_output/img_ids_'+str(rank)+'.npy')
        print(rank, PQs.shape, img_ids.shape)
        all_PQs.append(pd.DataFrame({'img_id': img_ids, 'PQ': PQs}))
    all_PQs = pd.concat(all_PQs)
    print(all_PQs['PQ'].describe())
    
    process_results(world_size, all_PQs, args, device)


