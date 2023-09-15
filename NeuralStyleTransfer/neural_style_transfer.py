import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse
import time

start_time = time.time()

# Hyperparameters
height = 400
content_weight = 1e5
style_weight = 3e4
tv_weight = 1e0
optimizer = 'lbfgs'
model = 'vgg19'
init_method = 'content'
saving_freq = -1


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
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    return tuning_step


def neural_style_transfer(config):
    content_img_path = os.path.join(config['content_images_dir'])
    style_img_path = os.path.join(config['style_images_dir'])

    os.makedirs(config['output_img_dir'], exist_ok=True)

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

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = {
        "lbfgs": 1000,
    }

    cnt = 0
    total_loss, content_loss, style_loss, tv_loss = 0, 0, 0, 0

    # Start of optimization procedure
    optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
    cnt = 0

    def closure():
        nonlocal cnt, total_loss, content_loss, style_loss, tv_loss
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        if total_loss.requires_grad:
            total_loss.backward()
    
        with torch.no_grad():
            print(f'LBFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
            print(f"Saving image for LBFGS after all iterations")  # Added print statement
            utils.save_and_maybe_display(optimizing_img, config['output_img_dir'], config, cnt, 1000, should_display=False)
        
        cnt += 1
        return total_loss

    optimizer.step(closure)

    return config['output_img_dir']

if __name__ == "__main__":
    start_time = time.time()

    ROOT_DIR = 'D:/DomainAdap.Neural'
    content_images_dir = os.path.join(ROOT_DIR, 'kitti', 'data_object_image_2', 'training', 'image_2', '000031.png')
    style_images_dir = os.path.join(ROOT_DIR, 'pytorch-neural-style-transfer', 'style_images', 'rainyday.jpg')
    output_img_dir = os.path.join(ROOT_DIR, 'pytorch-neural-style-transfer', 'output-dir')
    img_format = (6, '.jpg')

    if not os.path.exists(content_images_dir):
        raise ValueError(f"Content image not found at {content_images_dir}")

    if not os.path.exists(style_images_dir):
        raise ValueError(f"Style image not found at {style_images_dir}")

    # Consolidated configuration dictionary
    config = {
        'height': 400,
        'content_weight': 1e5,
        'style_weight': 3e4,
        'tv_weight': 1e0,
        'optimizer': 'lbfgs',
        'model': 'vgg19',
        'init_method': 'content',
        'saving_freq': -1,
        'content_img_name': os.path.basename(content_images_dir),
        'style_img_name': os.path.basename(style_images_dir),
        'content_images_dir': content_images_dir,
        'style_images_dir': style_images_dir,
        'output_img_dir': output_img_dir,
        'img_format': img_format
    }

    results_path = neural_style_transfer(config)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for the neural style transfer process: {elapsed_time:.2f} seconds")
