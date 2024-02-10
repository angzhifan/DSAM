import numpy as np
import argparse
import os
import sys
from configparser import ExtendedInterpolation
import configparser
import json
import cv2
from collections import Counter
import datetime

def parse_args(config_parser):

    """Parses args, using values from the config file as defaults."""
    parser = argparse.ArgumentParser(description='Process some integers.')


    parser.add_argument('--dec_layers_top', nargs="*",
                        type=lambda s: [item for item in s.split('\n') if item != ''],
                        default=config_parser["training"]["dec_layers_top"],
                        help='Top decoder layers - separate for each class')

    parser.add_argument('--dec_layers_bot', nargs="*",
                        type=lambda s: [item for item in s.split('\n') if item != ''],
                        default=config_parser["training"]["dec_layers_bot"],
                        help='bottom decoder layers same for all classes')

    parser.add_argument(
        "--bn",
        type=str,
        default=config_parser["training"]["bn"],
        help="type of batch normalization"
    )
    parser.add_argument(
        "--predir",
        type=str,
        default=config_parser["testing"]["predir"],
        help="folder for data and models"
    )

    parser.add_argument(
        "--by_occlusion",
        type=int,
        default=config_parser["testing"]["by_occlusion"],
        help="sort detections by occlusion"
    )
    parser.add_argument(
        "--remove_small_box",
        type=float,
        default=config_parser["testing"]["remove_small_box"],
        help="remove detections if it's smaller than remove_small_box"
    )
    parser.add_argument(
        "--train_data_suffix",
        type=str,
        default=config_parser["training"]["train_data_suffix"],
        help="suffix for train data images",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=config_parser["training"]["config_file"],
        help="config file"
    )
    parser.add_argument(
        "--trans_type",
        type=str,
        default=config_parser["training"]["trans_type"],
        help="type of transformations for decoder."
    )
    parser.add_argument(
        "--latent_space_dimension",
        type=int,
        default=int(config_parser["training"]["latent_space_dimension"]),
        help="dimension of the latent space"
    )
    parser.add_argument(
        "--cont_training",
        type=int,
        default=int(config_parser["training"]["cont_training"]),
        help="Continue training"
    )
    parser.add_argument(
        "--hidden_layer_size",
        type=int,
        default=int(config_parser["training"]["hidden_layer_size"]),
        help="size of hidden layer in fully connected decoder"
    )
    parser.add_argument(
        "--silent",
        type=int,
        default=int(config_parser["training"]["silent"]),
        help="Show images"
    )
    parser.add_argument(
        "--continue_training",
        type=int,
        default=int(config_parser["training"]["continue_training"]),
        help="Whether to continue training an existing VAE"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=int(config_parser["training"]["gpu"]),
        help="Which gpu to use"
    )
    parser.add_argument(
        "--train",
        type=int,
        default=int(config_parser["training"]["train"]),
        help="Train or test"
    )
    parser.add_argument(
        "--clutter",
        type=int,
        default=int(config_parser["training"]["clutter"]),
        help="Use clutter to train vae"
    )
    parser.add_argument(
        "--n_training_images",
        type=int,
        default=int(config_parser["training"]["n_training_images"]),
        help="number of training images"
    )
    parser.add_argument(
        "--n_testing_images",
        type=int,
        default=int(config_parser["training"]["n_testing_images"]),
        help="number of testing images"
    )

    parser.add_argument(
        "--cnn_num_train",
        type=int,
        default=int(config_parser["training"]["cnn_num_train"]),
        help="number of training data for faster r-cnn"
    )
    parser.add_argument(
        "--cnn_num_valid",
        type=int,
        default=int(config_parser["training"]["cnn_num_valid"]),
        help="number of validation data for faster r-cnn"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=int(config_parser["training"]["n_epochs"]),
        help="number of training epochs"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=int(config_parser["training"]["image_Size"]),
        help="size of training images"
    )
    parser.add_argument(
        "--image_size_unflattened",
        type=int,
        default=int(config_parser["training"]["image_size_unflattened"]),
        help="size of training images"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=float(config_parser["training"]["sigma"]),
        help="square root of c parameter"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(config_parser["training"]["batch_size"]),
        help="training batch size"
    )
    parser.add_argument(
        "--lr_latent_codes",
        type=float,
        default=float(config_parser["training"]["lr_latent_codes"]),
        help="learning rate for optimization of latent codes"
    )
    parser.add_argument(
        "--lr_decoder",
        type=float,
        default=float(config_parser["training"]["lr_decoder"]),
        help="learning rate for optimization of decoder parameters"
    )
    parser.add_argument(
        "--latent_opt",
        type=float,
        default=float(config_parser["training"]["latent_opt"]),
        help="whether to do latent code optimizations"
    )
    parser.add_argument(
        "--latent_iternum",
        type=int,
        default=int(config_parser["training"]["latent_iternum"]),
        help="number of latent code iterations"
    )
    parser.add_argument(
        "--test_latent_iternum",
        type=int,
        default=int(config_parser["training"]["test_latent_iternum"]),
        help="number of latent code iterations in test"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=config_parser["training"]["model_name"],
        help="trained model name"
    )
    parser.add_argument(
        "--cnn_type",
        type=str,
        default=config_parser["training"]["cnn_type"],
        help="type of faster rcnn"
    )
    parser.add_argument(
        "--num_class",
        type=int,
        default=int(config_parser["training"]["num_class"]),
        help="number of classes for faster rcnn"
    )
    parser.add_argument(
        "--mu_lr",
        type=float,
        default=float(config_parser["testing"]["mu_lr"]),
        help="learning rate for latent mean optimization"
    )
    parser.add_argument(
        "--test_data_suffix",
        type=str,
        default=config_parser["testing"]["test_data_suffix"],
        help="suffix for test data images",
    )
    parser.add_argument(
        "--faster_rcnn",
        type=str,
        default=config_parser["testing"]["faster_rcnn"],
        help="name of pretrained faster RCNN model",
    )
    parser.add_argument(
        "--vae_decoder",
        type=str,
        default=config_parser["testing"]["vae_decoder"],
        help="name of the pretrained VAE decoder"
    )
    parser.add_argument(
        "--log_sigma_lr",
        type=float,
        default=float(config_parser["testing"]["log_sigma_lr"]),
        help="learning rate for latent variance optimization"
    )
    parser.add_argument(
        "--shift_params_lr",
        type=float,
        default=float(config_parser["testing"]["shift_params_lr"]),
        help="learning rate for shift parameter optimization"
    )
    parser.add_argument(
        "--image_to_process",
        type=int,
        default=int(config_parser["testing"]["image_to_process"]),
        help="Which image to process"
    )

    def process_list_type(ar):
        ll=[int(a) for a in ar.split(',')]
        return ll

    parser.add_argument('--recheck', nargs="*",
                        type=process_list_type,
                        #type=lambda s: [int(item) for item in s.split(',')],
                        default=config_parser["testing"]["recheck"],
                        help='pair of classes to recheck')
    parser.add_argument(
        "--dim",
        type=int,
        default=int(config_parser["testing"]["dim"]),
        help="size of input images to vae"
    )
    parser.add_argument(
        "--cover_thresh",
        type=float,
        default=float(config_parser["testing"]["cover_thresh"]),
        help="coverage threshold"
    )
    parser.add_argument(
        "--visible_thresh",
        type=float,
        default=float(config_parser["testing"]["visible_thresh"]),
        help="visible image threshold"
    )
    parser.add_argument(
        "--validate",
        type=int,
        default=int(config_parser["testing"]["validate"]),
        help="use validation set"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=str(config_parser["testing"]["model_type"]),
        help="model type, VAEflow or GLF"
    )
    parser.add_argument(
        "--model_size",
        type=int,
        default=int(config_parser["testing"]["model_size"]),
        help="H,W dimension of decoder output"
    )
    parser.add_argument(
        "--test_image_size",
        type=int,
        default=int(config_parser["testing"]["test_image_size"]),
        help="Test image dimension"
    )
    parser.add_argument(
        "--big_dim",
        type=int,
        default=int(config_parser["testing"]["big_dim"]),
        help="Bigger image dimension for down sizes testimage"
    )
    parser.add_argument(
        "--angle",
        type=int,
        default=int(config_parser["testing"]["angle"]),
        help="Angle to rotate image"
    )
    parser.add_argument(
        "--detect_thresh",
        type=float,
        default=float(config_parser["testing"]["detect_thresh"]),
        help="image detection threshold"
    )
    parser.add_argument(
        "--soft_nms",
        type=int,
        default=float(config_parser["testing"]["soft_nms"]),
        help="Apply soft non maximum suppresion"
    )
    parser.add_argument(
        "--process_likelihoods",
        type=int,
        default=float(config_parser["testing"]["process_likelihoods"]),
        help="Run the likelihood processing"
    )
    parser.add_argument(
        "--lamb",
        type=int,
        default=int(config_parser["testing"]["lamb"]),
        help="TODO what is this"
    )
    parser.add_argument(
        "--occlusion_thresh",
        type=float,
        default=float(config_parser["testing"]["occlusion_thresh"]),
        help="Minimum value at pixel to belong to occluding mask"
    )
    parser.add_argument(
        "--occlusion_number_of_pixels",
        type=float,
        default=float(config_parser["testing"]["occlusion_number_of_pixels"]),
        help="Minimum number of pixels on object"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=float(config_parser["testing"]["noise"]),
        help="noise to add to the image"
    )
    parser.add_argument(
        "--num_test_images",
        type=int,
        default=int(config_parser["testing"]["num_test_images"]),
        help="number of test images to use"
    )
    parser.add_argument(
        "--draw",
        type=int,
        default=int(config_parser["testing"]["draw"]),
        help="whether to run in verbose mode"
    )
    parser.add_argument(
        "--blur",
        type=float,
        default=int(config_parser["testing"]["blur"]),
        help="Gaussian blurry parameter"
    )
    parser.add_argument(
        "--run_parallel",
        type=int,
        default=int(config_parser["testing"]["run_parallel"]),
        help="whether to split execution across GPUs"
    )
    parser.add_argument(
        "--run_gpu",
        type=int,
        default=int(config_parser["testing"]["run_gpu"]),
        help="Specific gpu to run on"
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        default=config_parser["testing"]["vae_type"],
        help="Type of VAE"
    )
    parser.add_argument(
        "--check_vae",
        type=int,
        default=int(config_parser["testing"]["check_vae"]),
        help="whether to check vae without embedding"
    )
    parser.add_argument(
        "--det_background",
        type=int,
        default=int(config_parser["testing"]["det_background"]),
        help="create background model from existing detections"
    )
    parser.add_argument(
        "--adjust_scale",
        type=int,
        default=int(config_parser["testing"]["adjust_scale"]),
        help="add scaling to transformation parameters"
    )
    parser.add_argument(
        "--adjust_rotate",
        type=int,
        default=int(config_parser["testing"]["adjust_rotate"]),
        help="add rotation to transformation parameters"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=int(config_parser["testing"]["sample"]),
        help="whether to sample in the decoder or just use mean"
    )

    parsed_args = parser.parse_args()
    if len(parsed_args.recheck)==1:
        parsed_args.recheck=parsed_args.recheck[0]
    return parsed_args

def find_bb(img, size=(28,28), thres=0):
  img = img.reshape(size[0],size[1],3).sum(2)
  temp1 = np.nonzero(np.sum(img, 0)>thres)
  temp2 = np.nonzero(np.sum(img, 1)>thres)
  return [np.min(temp1), np.min(temp2), np.max(temp1), np.max(temp2)]


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
    #print([far_x , near_x, far_y , near_y])
    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    pred_box_area = (x2p - x1p + 1) * (y2p - y1p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_args():
    dn = os.path.dirname(os.path.realpath(__file__))
    if len(sys.argv) == 1:
        print('You need to provide a parameter file')
        exit()
    else:
        if 'config_file' in sys.argv[1]:
            arg_file = sys.argv[1].strip('-').split('=')[1]
            full_file = os.path.join(dn, 'ARGS', arg_file)
            if not os.path.exists(full_file):
                print('config file: ' + full_file + ' Doesnt exist')
                exit(0)

        else:
            print('First argument must be configuration file')
            exit()

    # parse args from config file
    config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    config.read([os.path.join(dn, 'ARGS', 'args_default.conf'), os.path.join(dn, 'ARGS', arg_file)])
    
    args = parse_args(config)

    if args.predir=="":
        args.predir=os.path.join(dn[0:dn.find('research')+len('research')],'shared')
    args.dn = dn
    return args

def add_clutter(image,radius,num_clutter):


    ii=np.argmax(np.sum(image,axis=2))
    xc=ii//image.shape[1]
    yc=ii%image.shape[1]
    color=image[xc,yc]
    dim0=image.shape[0]-radius
    dim1=image.shape[1]-radius
    qq0=np.random.randint(radius,dim1,2*num_clutter)
    qq0=qq0.reshape(num_clutter,2)
    for k in  qq0:
            x=k[0]
            y=k[1]
            image=cv2.circle(image,(x,y),radius,color,-1)

    return image

def resize_objects(args,image):

    if args.big_dim>args.test_image_size:
        big_image = np.zeros((args.big_dim, args.big_dim, 3))
        hh = (args.big_dim - args.test_image_size) // 2
        big_image[hh:hh + args.test_image_size, hh:hh + args.test_image_size] = image
        image = np.array(Image.fromarray(np.uint8(big_image*255)).resize((args.test_image_size, args.test_image_size)))/255.

        image = cv2.resize(big_image, (args.test_image_size, args.test_image_size))
    else:
        crop_size = args.big_dim
        bb = find_bb(image, (args.test_image_size, args.test_image_size))
        target_size_j = (bb[3] - bb[1], bb[2] - bb[0])
        upperleft = [(crop_size - target_size_j[0]) // 2, (crop_size - target_size_j[1]) // 2]
        if (upperleft[0]<0 or upperleft[1]<0):
            print('big_dim too small')
            exit(0)
        bg = np.zeros((crop_size, crop_size, 3))
        bg[upperleft[0]:(upperleft[0] + target_size_j[0]), upperleft[1]:(upperleft[1] + target_size_j[1]
                                                                                 ), :] = image[bb[1]:bb[3], bb[0]:bb[2],                                                                     :]
        image = cv2.resize(bg, (args.test_image_size, args.test_image_size))

    return image


def read_correct_vals_off_file(file):
    """Reads the correct number of boxes and correct labels values off the
    end of each process file."""
    with open(file) as f:
        for line in f:
            pass
        last_line = line
    return last_line.split(" ")

def process_results(world_size, all_PQs, args, device):
    run_timestamp = datetime.datetime.strftime(datetime.datetime.now(), format="%Y-%m-%d_%H_%M")
    output_file_template = os.path.join(args.dn, 'output', "output_process_")
    combined_output_filename = os.path.join(args.dn, 'output', 'results_' + run_timestamp + '.txt')
    all_PQs.to_csv(args.dn+'/output/'+'/PQ_' + run_timestamp + '.csv', index=False)
    max_time_execute = 0
    ranks=list(range(world_size))+([device] if device is not None else [])
    for rank in ranks:
        process_output_filename = output_file_template+str(rank)
        if os.path.exists(process_output_filename):
            res = read_correct_vals_off_file(process_output_filename)
            time_exec_proc = round(float(res[0].replace('s', '')))
        # execution time is determined by last process to finish
            max_time_execute = max(
                max_time_execute, time_exec_proc
        )

    # write arguments to output file
    print(f"writing results to {combined_output_filename}")
    with open(combined_output_filename, "w") as final_output_file:

        final_output_file.write("runtime arguments:")
        final_output_file.write("\n")
        json.dump(args.__dict__, final_output_file, indent=2,sort_keys=True)
        final_output_file.write("\n")

        # write combined results to file
        final_output_file.write(
            f"execution time: {max_time_execute} seconds\n")
        for rank in ranks:
            process_output_filename = output_file_template + str(rank)
            if os.path.exists(process_output_filename):
                with open(process_output_filename,"r") as of:
                    for ll in of:
                        if 'predicted' in ll or 'error' or 'Missing' in ll:
                            final_output_file.write(ll)
        final_output_file.write(
            f"\n all_PQs.describe(): {all_PQs['PQ'].describe()} ")




def get_data(args):
    if args.check_vae == 1:
        # Running vae on test images sampled from same population as training. Boxes and labels are given
        bg_test = np.load(os.path.join(args.predir, 'VAE_train.npy')) / 255
        num = bg_test.shape[0]
        test_boxes_gt = np.tile(np.array((0, 0, args.model_size, args.model_size)), (num, 1)).reshape((num, 1, 4))
        train_gt = np.load(os.path.join(args.predir, 'train_gt.npy'))[0:num // 2, :, :]
        test_gt = train_gt[:, :, 4].reshape((-1, 1))
        test_gt = np.concatenate((np.ones((num, 1), dtype=np.int32), test_gt), axis=1)
    else:
        print('Loading ',os.path.join(args.predir, 'test_data' + args.test_data_suffix + '.npy'))
        bg_test = np.load(os.path.join(args.predir, 'test_data' + args.test_data_suffix + '.npy')) / 255
        test_gt = np.load(os.path.join(args.predir, 'test_gt' + args.test_data_suffix + '.npy'))
        test_boxes_gt = np.load(os.path.join(args.predir, 'test_boxes_gt' + args.test_data_suffix + '.npy'))

    return bg_test,test_boxes_gt,test_gt

def get_errors(output_file, gt, pred, i, DSA_matters, PQ_i):
        output_file.write(f"{i} DSA_matters? {DSA_matters} predicted {pred} ground truth {str(gt)} PQ score {PQ_i}\n")
        print(f"{i} predicted {pred} ground truth {str(gt)}\n")
        # cdiff = Counter(gt.numpy().tolist()) - Counter(pred.numpy().tolist())
        # if len(cdiff.items()) > 0:
        #     output_file.write(f"Missing detections: {(dict(cdiff))}\n")
        output_file.flush()
        sys.stdout.flush()
