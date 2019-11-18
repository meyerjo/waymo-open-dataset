import os
import re
import tensorflow as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_camera_image(frame, camera_image, camera_labels, layout, cmap=None):
    """Show a camera image and the given camera labels."""

    ax = plt.subplot(*layout)

    # Draw the camera labels.
    for camera_labels in frame.camera_labels:
        # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_image.name:
            continue

        # Iterate over the individual labels.
        for label in camera_labels.labels:
            # Draw the object bounding box.
            ax.add_patch(patches.Rectangle(
                xy=(label.box.center_x - 0.5 * label.box.length,
                    label.box.center_y - 0.5 * label.box.width),
                width=label.box.length,
                height=label.box.width,
                linewidth=1,
                edgecolor='red',
                facecolor='none'))

    # Show the camera image.
    plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
    plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
    plt.grid(False)
    plt.axis('off')

def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
    """Plots range image.

    Args:
      data: range image data
      name: the image title
      layout: plt layout
      vmin: minimum value of the passed data
      vmax: maximum value of the passed data
      cmap: color map
    """
    plt.subplot(*layout)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(name)
    plt.grid(False)
    plt.axis('off')

def get_range_image(range_images, laser_name, return_index):
    """Returns range image given a laser name and its return index."""
    return range_images[laser_name][return_index]

def show_range_image(range_image, layout_index_start = 1):
    """Shows range image.

    Args:
      range_image: the range image data from a given lidar of type MatrixFloat.
      layout_index_start: layout offset
    """
    range_image_tensor = tf.convert_to_tensor(range_image.data)
    range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
    lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
    range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                  tf.ones_like(range_image_tensor) * 1e10)
    range_image_range = range_image_tensor[...,0]
    range_image_intensity = range_image_tensor[...,1]
    range_image_elongation = range_image_tensor[...,2]
    plot_range_image_helper(range_image_range.numpy(), 'range',
                            [8, 1, layout_index_start], vmax=75, cmap='gray')
    plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                            [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
    plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                            [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')

def projected_points_to_image(im_shape, points, color_func=None):
    from PIL import Image
    assert(color_func is not None)
    #
    im = np.zeros(shape=im_shape, dtype=np.float32)
    y = points[:, 0].astype(np.int32)
    x = points[:, 1].astype(np.int32)
    d = points[:, 2].tolist()
    d_col = np.array([color_func(_) for _ in d])

    im[x, y, :] = d_col[:, 0:3] * 255

    return Image.fromarray(im.astype(np.uint8))

def rgba(r):
    """Generates a color based on range.

    Args:
      r: the range value of a given point.
    Returns:
      The color for a given range
    """
    c = plt.get_cmap('jet')((r % 20.0) / 20.0)
    c = list(c)
    c[-1] = 0.5  # alpha
    return c

def plot_image(camera_image, create_fig=True):
    """Plot a cmaera image."""
    if create_fig:
        plt.figure(figsize=(20, 12))
    plt.imshow(tf.image.decode_jpeg(camera_image.image))
    plt.grid("off")

def plot_points_on_image(
        projected_points, depth_image, camera_image, rgba_func,
                         point_size=5.0):
    """Plots points on a camera image.

    Args:
      projected_points: [N, 3] numpy array. The inner dims are
        [camera_x, camera_y, range].
      camera_image: jpeg encoded camera image.
      rgba_func: a function that generates a color from a range value.
      point_size: the point size.

    """

    xs = []
    ys = []
    colors = []

    for point in projected_points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba_func(point[2]))

    plt.subplot(131)
    plot_image(camera_image, create_fig=False)
    plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
    #
    plt.subplot(132)
    plot_image(camera_image, create_fig=False)
    #
    plt.subplot(133)
    plt.imshow(depth_image)
    plt.grid("off")

import numpy as np
def asStride(arr,sub_shape,stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1=arr.strides[:2]
    m1,n1=arr.shape[:2]
    m2,n2=sub_shape
    view_shape=(1+(m1-m2)//stride[0],1+(n1-n2)//stride[1],m2,n2)+arr.shape[2:]
    strides=(stride[0]*s0,stride[1]*s1,s0,s1)+arr.strides[2:]
    subs=np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs

def poolingOverlap(mat,ksize,stride=None,method='max',pad=False):
    '''Overlapping pooling on 2D or 3D data.
    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).
    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize
    if stride is None:
        stride=(ky,kx)
    sy,sx=stride

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,sy)
        nx=_ceil(n,sx)
        size=((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        mat_pad=mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view=asStride(mat_pad,ksize,stride)

    if method=='max':
        result=np.nanmax(view,axis=(2,3))
    else:
        result=np.nanmean(view,axis=(2,3))

    return result