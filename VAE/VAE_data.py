import json
import pycocotools.mask as mask_utils
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
phase = 'test'
annpath = '/home/fana/data/KITTI_AMODAL_DATASET/update_'+phase+'_2020.json'
imagepath = '/home/fana/data/KITTI/'+phase+'ing/image_2/'
output_path = '/home/fana/DSA_mask/VAE/'
VAE_d = 48

def polys_to_mask(polygons, height, width):
	rles = mask_utils.frPyObjects(polygons, height, width)
	rle = mask_utils.merge(rles)
	mask = mask_utils.decode(rle)
	return mask

def get_VAE_data(annpath = annpath, imagepath = imagepath):
    f = open(annpath)
    data = json.load(f)
    images = []
    masks = []
    labels = []

    images_id = {}
    for image in data['images']:
        images_id[image['id']] = image['file_name']

    cnt = 0
    start = time.time()
    for annotation in data['annotations']:
        if min(annotation['i_bbox']) < 0: continue
        key = images_id[annotation['image_id']]
        im_data = mpimg.imread(imagepath + key)
        height, width, n_channels = im_data.shape
        assert n_channels == 3
        box = annotation['i_bbox']
        old_size = max(box[2], box[3])
        if old_size < 30: continue
        my_image = np.zeros((old_size, old_size, n_channels))
        x1, y1 = (old_size - box[3]) // 2, (old_size - box[2]) // 2
        if box[1]-x1<0 or box[0]-y1<0 or box[1]+box[3]+x1>=height or box[0]+box[2]+y1>=width: continue
        my_image = im_data[(box[1]-x1):(box[1]-x1+old_size), (box[0]-y1):(box[0]-y1+old_size), :]
        my_image = cv2.resize(my_image, dsize=(VAE_d, VAE_d), interpolation=cv2.INTER_LINEAR)
        images.append(my_image)

        mask = [polys_to_mask(annotation["i_segm"], height, width)]
        mask = np.array(mask).squeeze()
        my_mask = mask[(box[1]-x1):(box[1]-x1+old_size), (box[0]-y1):(box[0]-y1+old_size)]
        my_mask = (cv2.resize(my_mask, dsize=(VAE_d, VAE_d)) > 0.5).astype(np.int8)
        masks.append(my_mask)
        labels.append(annotation['category_id'])

        cnt += 1
        if cnt%1000==0: 
            print(cnt)

    images = np.array(images)
    masks = np.array(masks)
    labels = np.array(labels)
    np.save(output_path+'VAE_'+phase+'_images'+str(cnt)+'.npy', images)
    np.save(output_path+'VAE_'+phase+'_masks'+str(cnt)+'.npy', masks)
    np.save(output_path + 'VAE_'+phase+'_labels'+str(cnt)+'.npy', labels)

    print('time cost:', time.time()-start)
    print('num objects:', cnt, images.shape, masks.shape, labels.shape)

get_VAE_data()

