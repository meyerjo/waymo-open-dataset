import argparse
import re
import threading
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

if hasattr(tf, 'enable_eager_execution'):
    tf.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from custom_utils import poolingOverlap, projected_points_to_image, turbo_rgba
import matplotlib.pyplot as plt
from custom_utils import colorize_np_arr

import errno
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

executor = None

camera_names = {
    'front': open_dataset.CameraName.FRONT,
    # 'front_left': open_dataset.CameraName.FRONT_LEFT,
    # 'front_right': open_dataset.CameraName.FRONT_RIGHT,
    # 'side_left': open_dataset.CameraName.SIDE_LEFT,
    # 'side_right': open_dataset.CameraName.SIDE_RIGHT
}


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get(object_list, name):
    """ Search for an object by name in an object list. """

    object_list = [obj for obj in object_list if obj.name == name]
    return object_list[0]


def depth_map_by_camera_name(frame, camera, points_all, cp_points_all):
    """

    :param frame: frame camera
    :param camera: camera name
    :param points_all:  ndarray
    :param cp_points_all: ndarray
    :return:
    """
    camera_image = get(frame.images, camera)
    # get the image
    img_np = np.zeros(shape=(1280, 1920))
    # img_tensor = tf.image.decode_jpeg(camera_image.image)
    #
    # img_np = np.zeros_like(img_tensor.numpy())[:, :, 0].astype(np.float32)
    # img_np[:] = np.nan

    # The distance between lidar points and vehicle frame origin.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

    mask = tf.equal(cp_points_all_tensor[..., 0], camera_image.name)
    cp_points_all_tensor = tf.cast(
        tf.gather_nd(
            cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

    projected_points_all_from_raw_data = tf.concat(
        [cp_points_all_tensor[..., 1:3], points_all_tensor],
        axis=-1).numpy()

    coordinates = projected_points_all_from_raw_data[:, 0:2]. \
        astype(np.int32)

    img_np[coordinates[:, 1], coordinates[:, 0]] = \
        projected_points_all_from_raw_data[:, 2]

    dense_matrix = \
        poolingOverlap(img_np, ksize=(15, 10), stride=(1, 1), method=None,
                       pad=True)

    return dense_matrix

def pool_points_project_to_image(dense_matrix, img_np):

    # TODO: this is super costly
    (width, height) = img_np.shape
    dense_matrix = np.nanmean(dense_matrix, axis=(2, 3))

    xv, yv = np.meshgrid(range(0, width), range(0, height), indexing='ij')
    xv, yv = xv.reshape(-1, ), yv.reshape(-1, )
    dense_vector = dense_matrix.reshape(-1, )
    valid_coordinates = dense_vector > 0.
    xv, yv = xv[valid_coordinates], yv[valid_coordinates]
    dense_vector = dense_vector[valid_coordinates]
    projected_matrix = np.vstack([yv, xv, dense_vector]).transpose()

    # project the points to an depth_np_arr
    depth_np_arr = projected_points_to_image(
        img_np.shape, projected_matrix
    )

    return depth_np_arr

def project_image_lidar_to_image_plane(frame, camera_names):
    (range_images, camera_projections, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    # # convert the points to a point cloud from the first laser return
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose,
        ri_index=0)
    # ... and the second return
    points_second, cp_points_second = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose,
        ri_index=1)
    #
    points_all = np.concatenate(points, axis=0)
    cp_points_all = np.concatenate(cp_points, axis=0)
    points_second_all = np.concatenate(points_second, axis=0)
    cp_points_second_all = np.concatenate(cp_points_second, axis=0)
    #
    points_all = np.concatenate([points_all, points_second_all])
    cp_points_all = np.concatenate([cp_points_all, cp_points_second_all])

    # go through all the camera images
    _image_results = {}
    for camera_name, camera_dataset in camera_names.items():
        # img_np = np.zeros(shape=(1280, 1920))
        depth_np_matrix = depth_map_by_camera_name(
            frame, camera_dataset, points_all, cp_points_all
        )

        if camera_name not in _image_results:
            _image_results[camera_name] = depth_np_matrix
        else:
            raise BaseException('Camera-Name appear twice {}'.format(camera_name))

    return _image_results

def colorize_depth_map_save_img(depth_map, output_path, frame_id):
    print('[{}] {} Start colorizing: {}'.format(
        datetime.now(), threading.get_ident(), os.path.join(
            output_path, 'frame_{:04d}'.format(frame_id))
    ))
    img_np = np.zeros(shape=(1280, 1920))
    depth_np_arr = pool_points_project_to_image(depth_map, img_np)
    print(np.max(depth_np_arr))
    colorized_depth_image = colorize_np_arr(
        depth_np_arr, turbo_rgba
    )
    _save_path = os.path.join(output_path,
                              'frame_{:05d}.png'.format(frame_id))
    colorized_depth_image.save(_save_path)



def handle_file(input_filename, output_dir):
    assert executor is not None, 'executor has to be set to None'
    input_path, input_fname = os.path.split(input_filename)
    segment_m = re.search(
        '(.+_with_camera_labels).tfrecord$', input_fname
    )
    if segment_m is None:
        return
    # create the output folder
    if not os.path.exists(os.path.join(output_dir, 'lidar_rgb')):
        os.mkdir(os.path.join(output_dir, 'lidar_rgb'))

    output_data_folder = os.path.join(output_dir, 'lidar_rgb',
                                      segment_m.group(1))
    if not os.path.exists(output_data_folder):
        os.mkdir(output_data_folder)

    try:
        dataset = tf.data.TFRecordDataset(input_filename, compression_type='')
    except BaseException as e:
        print('Error while reading file: {}'.format(e))
        return

    read_frames = []
    print('[{}] Reading all frames'.format(datetime.now()))
    _s = datetime.now()
    for data in dataset:
        frame = open_dataset.Frame()
        try:
            frame.ParseFromString(bytearray(data.numpy()))
        except BaseException as e:
            print('\tError: {}'.format(e))
        read_frames.append(frame)
    print('[{}] Read all frames. Took {}'.format(
        datetime.now(), datetime.now() - _s))

    # # TODO: dropping frames for development purposes
    # read_frames = read_frames[:5]

    print('[{}] Extract dense matrices for all frames'.format(datetime.now()))
    _s = datetime.now()
    image_matrices = []
    for i, frame in enumerate(read_frames):
        if i % 10 == 0:
            print('Frame {}/{}'.format(i, len(read_frames)))
            continue
        image_matrices.append(project_image_lidar_to_image_plane(frame, camera_names))
    print('[{}] Extracted all dense-matrices for all frames ({} / sample)'.format(
        datetime.now() - _s, (datetime.now() - _s) / len(read_frames)))

    print('[{}] Colorize all frames'.format(datetime.now()))
    futures = []
    for frame_id, image_dict in enumerate(image_matrices):
        for camera_name, depth_np_matrix in image_dict.items():
            _fpath = os.path.join(output_data_folder, camera_name)
            if not os.path.exists(_fpath):
                mkdir_p(_fpath)

            # colorize_depth_map_save_img(depth_np_matrix, _fpath, frame_id)
            future = executor.submit(
                colorize_depth_map_save_img, depth_np_matrix, _fpath, frame_id
            )
            futures.append(future)

            # img_np = np.zeros(shape=(1280, 1920))
            # depth_np_arr = pool_points_project_to_image(depth_np_matrix, img_np)
            # print(np.max(depth_np_arr))
            # colorized_depth_image = colorize_np_arr(
            #     depth_np_arr, turbo_rgba
            # )
            # image_dict[camera_name] = colorized_depth_image
    print('[{}] Colorized all frames'.format(datetime.now() - _s))

    return futures


if __name__ == '__main__':
    parser = argparse.ArgumentParser('export_data')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory',
                        default='./output')
    parser.add_argument('--max_workers', type=int, default=1)
    parser.add_argument('--files_to_handle', type=int, default=-1)
    args = parser.parse_args()

    max_workers = 1
    if args.max_workers != -1:
        max_workers = args.max_workers
    executor = ThreadPoolExecutor(max_workers=max_workers)

    from turbo_map import RGBToPyCmap, turbo_colormap_data

    mpl_data = RGBToPyCmap(turbo_colormap_data)
    plt.register_cmap(name='turbo', data=mpl_data,
                      lut=turbo_colormap_data.shape[0])

    mpl_data_r = RGBToPyCmap(turbo_colormap_data[::-1, :])
    plt.register_cmap(name='turbo_r', data=mpl_data_r,
                      lut=turbo_colormap_data.shape[0])

    if not os.path.exists(args.output_dir):
        print('Output directory does not exist')
        exit(1)

    input_directory = os.path.join(
        os.path.expanduser('~'), 'Downloads', 'waymodataset'
    ) if args.input_dir is None else args.input_dir
    if not os.path.exists(input_directory):
        print('Input directory does not exist')
        exit(1)

    # get the files in the directory
    files = sorted(os.listdir(input_directory))
    if args.files_to_handle > 0:
        files = files[:args.files_to_handle]

    num_files = len(files)
    print(f'Reading from {input_directory} found {num_files} files')

    # go through them
    collected_futures = {}
    for i, f in enumerate(files):
        print(f'({i}/{num_files}) Extracting lidar data from {f}')
        _input_filename = os.path.join(input_directory, f)
        futures = handle_file(_input_filename, args.output_dir)
        collected_futures[f] = futures

    for key, futures in collected_futures.items():
        print(key, sum([f.done() for f in futures]))

    executor.shutdown(wait=True)
