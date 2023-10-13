import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable

from PIL import Image
import numpy as np
import os
import time

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss

def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    return tuning_step


def neural_style_transfer(config):
    content_img_path = config['content_img_path']
    style_img_path = config['style_img_path']

    # Directly use the output_img_dir without creating a new sub-directory for each image.
    dump_path = config['output_img_dir']
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, config['height'], device)
    style_img = utils.prepare_img(style_img_path, config['height'], device)

    if config['init_method'] == 'random':
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')
    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]


    #
    # Start of optimization procedure
    #

    content_basename = os.path.splitext(os.path.basename(config['content_img_path']))[0]
    image_filename = f"NST_{content_basename}.png"

    # never used it really
    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)   
        for cnt in range(config['num_of_iterations'][config['optimizer']]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, image_filename, config, cnt, config['num_of_iterations'][config['optimizer']], should_display=False)

    # lbfgs
    elif config['optimizer'] == 'lbfgs':
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS((optimizing_img,), max_iter=config['num_of_iterations']['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, config['num_of_iterations'][config['optimizer']], should_display=False)

            cnt += 1
            return total_loss

        optimizer.step(closure)

    return dump_path

# function that will commit style transfer on an entire image folder
def process_directory(content_images_dir, style_image_path, output_img_dir, labels_dir, optimization_config):
    for content_image_file in os.listdir(content_images_dir):
        content_image_path = os.path.join(content_images_dir, content_image_file)
        # skip files that already exist
        if "NST_" + content_image_file in os.listdir(output_img_dir):
            continue
        if os.path.isfile(content_image_path):
            # Update the content image path in the config
            optimization_config['content_img_name'] = os.path.basename(content_image_path)
            optimization_config['content_img_path'] = content_image_path

            start_time_ = time.time()

            # Perform style transfer
            neural_style_transfer(optimization_config)

            end_time_ = time.time()
            duration_ = end_time_ - start_time_
            print(f"Neural style transfer for image {content_image_file} took {duration_:.2f} seconds.")

            # Map and copy the label
            label_file = os.path.splitext(content_image_file)[0] + ".txt" 
            original_label_path = os.path.join(labels_dir, label_file)
            new_label_path = os.path.join(output_img_dir, "NST" + label_file)
            if os.path.isfile(original_label_path):
                # Copy the original label to the output directory with the new naming scheme
                import shutil
                shutil.copy(original_label_path, new_label_path)



if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    root_dir = 'D:/DomainAdap.Neural'
    # content_images_dir = os.path.join(root_dir, 'Data', 'VKitti', 'VKITTI', 'clone', 'Scene01_clone_Camera0_00188.png')
    #D:\DomainAdap.Neural\Data\VKitti\VKITTI\clone
    content_images_dir = os.path.join(root_dir, 'Data', 'VKitti', 'VKITTI', 'clone') # for process_directory
    style_image_dir = os.path.join(root_dir, 'pytorch-neural-style-transfer', 'style_images', 'twilight3.png')
    output_img_dir = os.path.join(root_dir, 'Data', 'VKitti', 'VKITTI', 'stylized clone')
    labels_dir = os.path.join(root_dir, 'Data', 'Vkitti', 'VKITTI', 'clone labels')
    
    
    optimization_config = {
        'content_img_name': os.path.basename(content_images_dir),
        'style_img_name': os.path.basename(style_image_dir),
        'content_img_path': content_images_dir,
        'style_img_path': style_image_dir,
        'height': (640, 640),
        'num_of_iterations': {
        "lbfgs": 800,
        "adam": 3000,},
        'content_weight': 4e5,
        'style_weight': 3e2,
        'tv_weight': 1e1,
        'optimizer': 'lbfgs',
        'model': 'vgg19',
        'init_method': 'content',
        'saving_freq': -1,
        'output_img_dir': output_img_dir,
    }

    # 
    ## recommended hyperparameter settings
    #
    # lbfgs, content init -> (cw, sw, tv) = (1e5, 3e4, 1e0)
    # lbfgs, style   init -> (cw, sw, tv) = (1e5, 1e1, 1e-1)
    # lbfgs, random  init -> (cw, sw, tv) = (1e5, 1e3, 1e0)

    # adam, content init -> (cw, sw, tv, lr) = (1e5, 1e5, 1e-1, 1e1)
    # adam, style   init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)
    # adam, random  init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)

    # process entire directory
    start_time = time.time()

    process_directory(content_images_dir, style_image_dir, output_img_dir, labels_dir, optimization_config)

    end_time = time.time()
    duration = end_time - start_time
    print(f"The process_directory function took {duration:.2f} seconds to complete.")

    """
    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    results_path = neural_style_transfer(optimization_config)

    print(f"The neural_style_transfer function took {duration:.2f} seconds to complete.")
    """

    
    
    
