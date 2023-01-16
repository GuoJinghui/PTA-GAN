"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    # x = x.view(x.size(0), 1, 256, 256)
    x = x.view(x.size(0), 1, 256, 256)
    return x


def to_img2(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    # x = x.view(x.size(0), 1, 256, 256)
    x = x.view(x.size(0), 1, 512, 512)
    return x


def L1L2(tensor, mask):
    # print(tensor.dtype)
    # print(mask.dtype)

    L1 = torch.mul(tensor, mask.float()).float().cuda()
    L2 = torch.mul(tensor, 1.0 - mask.float()).float().cuda()
    return L1, L2


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def tensor_to_np(tensor):
    # img = tensor.mul(255).byte()
    img = tensor[0].cpu().float().numpy()
    return img


def show_from_cv(img, title=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cm_hot = mpl.cm.get_cmap('hot')
    im = cm_hot(img)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    plt.figure()
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def tensor2im(input_image, image_path, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            # _, cols, rows = image_numpy.shape
            # image_numpy = cv2.resize(image_numpy, (rows, cols))
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255  # post-processing: tranpose and scaling
        # image_numpy = np.array(image_numpy * 255, dtype=np.uint8)
        # image_numpy = image_numpy.reshape(image_numpy.shape + (1,))
        # image_numpy = image_numpy[:, :, ::-1].copy()
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    # image_numpy = image_numpy.astype(imtype)
    # im_color = cv2.applyColorMap(image_numpy, cv2.COLORMAP_TURBO)
    # cv2.imwrite(image_path, im_color)
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_csv(tensor, csv_path, MIN=0.0, MAX=180000.0):
    if not isinstance(tensor, np.ndarray):
        if isinstance(tensor, torch.Tensor):  # get the data from a variable
            csv_tensor = tensor.data
        else:
            return tensor
        csv_numpy = csv_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if csv_numpy.shape[0] == 1:  # drop channel
            # _, rows, cols = csv_numpy.shape
            csv_numpy = (csv_numpy + 1) / 2.0
            # csv_numpy = np.transpose(csv_numpy, (1, 2, 0))
            if csv_numpy.size == 262144:
                csv_numpy = csv_numpy.reshape(512, 512)
            else:
                csv_numpy = csv_numpy.reshape(256, 256)
            # image_numpy = cv2.resize(image_numpy, (rows, cols))
        csv_numpy = np.array((csv_numpy * (MAX - MIN) + MIN), dtype='f') # post-processing: tranpose and scaling
        # image_numpy = np.array(image_numpy * 255, dtype=np.uint8)
        # image_numpy = image_numpy.reshape(image_numpy.shape + (1,))
        # image_numpy = image_numpy[:, :, ::-1].copy()
    else:  # if it is a numpy array, do nothing
        csv_numpy = tensor
    # image_numpy = image_numpy.astype(imtype)
    # im_color = cv2.applyColorMap(image_numpy, cv2.COLORMAP_TURBO)
    # cv2.imwrite(image_path, im_color)
    np.savetxt(csv_path, csv_numpy, delimiter=",")


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)

    # Create heatmap image in red channel
    # heatmap = torch.empty(1, 256, 256).uniform_(0, 1)
    # heatmap = torch.cat((heatmap, torch.zeros(2, 256, 256)))
    # h_img = TF.to_pil_image(heatmap)
    #
    # res = Image.blend(image_pil, h_img, 0.5)
    image_pil.save(image_path)



def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
