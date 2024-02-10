import collections
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import pdb
import copy
import torch
import time
import numpy as np
import random
from datetime import date
import json
#from utils import get_args
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.image as mpimg
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
model_type = "DPT_Hybrid" # "MiDaS_small" #
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
iou_thres = 0.75

def polys_to_mask(polygons, height, width):
	rles = mask_utils.frPyObjects(polygons, height, width)
	rle = mask_utils.merge(rles)
	mask = mask_utils.decode(rle)
	return mask

def map_labels(label):
    if label in [4,5,6,7]:
        return 4
    elif label in [1, 8]:
        return 1
    elif label in [2, 3]:
        return 2

def iou(array1, array2):
    x1, y1, x2, y2 = array1[0],array1[1],array1[2],array1[3]
    x1p, y1p, x2p, y2p = array2[0],array2[1],array2[2],array2[3]
    if not all([x2 >= x1, y2 >= y1, x2p >= x1p, y2p >= y1p]):
        return 0
    far_x = np.min([x2, x2p])
    near_x = np.max([x1, x1p])
    far_y = np.min([y2, y2p])
    near_y = np.max([y1, y1p])
    if not all([far_x >= near_x, far_y >= near_y]):
        return 0
    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    pred_box_area = (x2p - x1p + 1) * (y2p - y1p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)#weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

# We assume the training data has pairs of objects
class Mask_Rcnn_Dataset_KINS(Dataset):
    def __init__(self, keys, path_images, anotate):
        super().__init__()
        self.keys = keys
        self.path_images = path_images
        self.anotate = anotate

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        im_data = mpimg.imread(self.path_images+key)
        height, width, n_channels = im_data.shape
        assert n_channels==3
        annotation = self.anotate[key]

        area = annotation['i_area']
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.as_tensor([map_labels(label) for label in annotation['category_id']], dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(annotation['category_id']),), dtype=torch.int64)

        target = {}
        boxes = annotation['i_bbox']
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        masks = [polys_to_mask(poly, height, width) for poly in annotation["i_segm"]]
        masks = np.array(masks)

        image_id = torch.tensor([idx])

        target['boxes'] = boxes
        target['labels'] = labels
        target["masks"] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        return im_data.astype(np.float32).transpose((2, 0, 1)), target, idx

def evaluation(model,dataloader):
    valid_loss = []
    for images, targets, image_ids in dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v[i].to(device) for k, v in targets.items()} for i in range(batch_size)]
        #tmp_start = time.time()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        valid_loss.append(losses.item())
    return np.mean(valid_loss)


def preprocess_anotate_KINS(dataset_dict):
    return dataset_dict

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    batch_size = 1
    padding = False #True if batch_size>1 else False
    total_train_loss = []
    patience = 5
    trigger_times = 0


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    dataset = 'KINS'
    num_epochs = 50
    cont_training = False
    output_path = '/home/fana/DSA_mask/MaskRCNN/'
    phase = 'train' #'test' #

    if dataset == 'KINS':
        num_classes = 8 + 1
        path = '/home/fana/data/KITTI_AMODAL_DATASET/'
        train_annotate = json.load(open(path + "grouped_data_train.json"))
        test_anotate = json.load(open(path + "grouped_data_test.json"))
        path_train_images = '/home/fana/data/KITTI/training/image_2/'
        path_test_images = '/home/fana/data/KITTI/testing/image_2/'

        if phase=='train':
            train_annotate = preprocess_anotate_KINS(train_annotate)
            train_num, valid_num = int(len(train_annotate) * 0.8), len(train_annotate) - int(len(train_annotate) * 0.8)
            #train_num, valid_num = 800, 200
            train_keys = list(train_annotate.keys())
            random.shuffle(train_keys)
            train_keys, valid_keys = train_keys[:train_num], train_keys[train_num:(train_num+valid_num)]
            print('train, valid data:', train_num, valid_num)
            print('finish train, valid annotation')
            traindata = Mask_Rcnn_Dataset_KINS(train_keys, path_train_images, train_annotate)
            trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
            validdata = Mask_Rcnn_Dataset_KINS(valid_keys, path_train_images, train_annotate)
            validloader = torch.utils.data.DataLoader(validdata, batch_size=batch_size, shuffle=False)
        elif phase=='test':
            test_anotate = preprocess_anotate_KINS(test_anotate)
            test_keys = list(test_anotate.keys())
            print('test data:', len(test_keys))
            testdata = Mask_Rcnn_Dataset_KINS(test_keys, path_test_images, test_anotate)
            testloader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)

    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    start_time = time.time()

    if phase == 'train':
        checkpoint_path = os.path.join(output_path, 'maskrcnn' + dataset + '_' + str(
            train_num + valid_num)+ '_' + str(date.today()))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        if cont_training:
            print('Continuing to train',checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path,map_location=device)['state_dict'])
        current_valid_loss = evaluation(model, validloader)
        print('validation set current_valid_loss:', current_valid_loss)
        best_valid_loss = current_valid_loss

        for epoch in range(num_epochs):
            print(f'Epoch :{epoch + 1}')
            start_time = time.time()
            train_loss = []
            model.train()
            cnt = 0
            for images, targets, image_ids in trainloader:

                images = list(image.to(device) for image in images)
                targets = [{k: v[i].to(device) for k, v in targets.items()} for i in range(batch_size)]

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                train_loss.append(losses.item())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                cnt += 1
                if cnt%4000==0: print("cnt=", cnt)
                if cnt%4000==0:
                    print(losses, targets[0]['image_id'].cpu().numpy().astype(int)[0])
                    print(train_keys[targets[0]['image_id'].cpu().numpy().astype(int)[0]], loss_dict)
            epoch_train_loss = np.mean(train_loss)
            total_train_loss.append(epoch_train_loss)
            print(f'Epoch train loss is {epoch_train_loss}')

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'train_loss_min': epoch_train_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            time_elapsed = time.time() - start_time
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # early stopping
            current_valid_loss = evaluation(model, validloader)
            print('validation set current_valid_loss:', current_valid_loss)
            if current_valid_loss > best_valid_loss:
                trigger_times += 1
                print('Trigger Times:', trigger_times)
                if trigger_times >= patience:
                    print('Early stopping')
                    break
            else:
                best_valid_loss = current_valid_loss
                print('trigger times: 0')
                trigger_times = 0
                # save checkpoint
                torch.save(checkpoint, checkpoint_path)
    elif phase=='test':
        checkpoint_path = os.path.join(output_path, 'maskrcnn' + dataset + '_' + str(7474)+ '_2023-04-09')
        print('Begin to test', checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])

        test_loss = []
        model.eval()
        metric = MeanAveragePrecision()
        total_pairs, correct_pairs = 0, 0
        cnt = 0
        for image, target, image_id in testloader:
            input_image_torch = torch.tensor(torch.squeeze(image,0), dtype=torch.float32).to(device)
            temp = model([input_image_torch])

            input_image = mpimg.imread(path_test_images+test_keys[image_id[0]])
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

            gt = test_anotate[test_keys[image_id[0]]]
            l0, l1 = len(gt['category_id']), temp[0]['labels'].cpu().detach().shape[0]
            if cnt%500==0: print('num true boxes, num predictions', l0, l1)
            bbs = temp[0]['boxes'].cpu().detach().numpy()
            pred_labels = temp[0]['labels'].cpu().numpy()
            scores = np.round(temp[0]['scores'].cpu().detach().numpy(), 4)
            preds = [dict(boxes = temp[0]['boxes'].cpu().detach(),
                          scores = temp[0]['scores'].cpu().detach(),labels = temp[0]['labels'].cpu().detach(),)]
            target = [dict(boxes = torch.tensor(target['boxes']).squeeze(0),labels = target['labels'].view(-1),)]
            metric.update(preds, target)

            # calculate the acc of depth
            depths = collections.defaultdict(list)
            for i in range(l1):
                for j in range(l0):
                    if iou(bbs[i], gt['i_bbox'][j])>iou_thres and pred_labels[i]==gt['category_id'][j]:
                        pred_mask = (temp[0]['masks'].detach().cpu().numpy()[i].squeeze() > 0.5).astype(np.int32)
                        avg_depth = (pred_mask * depth).sum() / pred_mask.sum()
                        # print('avg_depth', avg_depth, pred_mask.max(), pred_mask.min(), pred_mask.sum())
                        depths[gt['oco_id'][j]].append((gt['ico_id'][j], avg_depth))
            for oco_id in depths:
                depth_cluster = sorted(depths[oco_id])
                l = len(depth_cluster)
                for i in range(l):
                    for j in range(l):
                        if depth_cluster[i][0]!=depth_cluster[j][0]:
                            total_pairs += 1
                            if (depth_cluster[i][0]-depth_cluster[j][0])*(depth_cluster[i][1]-depth_cluster[j][1])<0:
                                correct_pairs += 1
            cnt += 1
        print('MeanAveragePrecision', metric.compute())
        print('occlusion accuracy:', correct_pairs/total_pairs, 'correct_pairs:', correct_pairs, 'total_pairs:', total_pairs)

    time_elapsed = time.time() - start_time
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


