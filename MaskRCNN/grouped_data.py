import json
from collections import defaultdict
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import pdb

path = '/home/fana/data/KITTI_AMODAL_DATASET/'

def polys_to_mask(polygons, height, width):
	rles = mask_utils.frPyObjects(polygons, height, width)
	rle = mask_utils.merge(rles)
	mask = mask_utils.decode(rle)
	return mask

def group_data(phase):
    filepath = path + 'update_' + phase + '_2020.json'
    f = open(filepath)
    data = json.load(f)

    images_id = {}
    grouped_data ={}
    for image in data['images']:
        images_id[image['id']] = image
        grouped_data[image['file_name']] = {'category_id':[], 'a_segm':[], 'i_segm':[], 'a_bbox':[], 'i_bbox':[], 'a_area':[],
                                 'i_area':[], 'oco_id':[], 'ico_id':[], 'height':image['height'], 'width':image['width']}
    print("len images_id: ", len(images_id))

    for annotation in data['annotations']:
        if min(annotation['i_bbox']) < 0: continue
        image = images_id[annotation['image_id']]
        for key in grouped_data[image['file_name']]:
            if key in ['height', 'width']: continue
            # if key == "i_segm":
            #     grouped_data[image['file_name']][key].append(polys_to_mask(annotation[key], image[
            #         'height'], image['width']).tolist())
            elif key in ['a_bbox', 'i_bbox']:
                box = annotation[key]
                grouped_data[image['file_name']][key].append([box[0], box[1], box[0]+box[2], box[1]+box[3]])
            else:
                grouped_data[image['file_name']][key].append(annotation[key])

    print("len grouped_data: ", len(grouped_data))

    to_delete = []
    for image_name in grouped_data:
        l = len(grouped_data[image_name]['i_bbox'])
        if l == 0:  to_delete.append(image_name)

    for image_name in to_delete:
        del grouped_data[image_name]

    print("len grouped_data nonempty: ", len(grouped_data))

    with open(path + "grouped_data_" + phase + ".json", "w") as outfile:
        json.dump(grouped_data, outfile)


group_data("train")
print("finish training set")
group_data("test")
print("finish testing set")
