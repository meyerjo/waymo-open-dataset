import argparse
import re
import time

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from custom_utils import poolingOverlap, projected_points_to_image, turbo_rgba
import matplotlib.pyplot as plt
from custom_utils import colorize_np_arr

import errno
import os

from concurrent.futures import ThreadPoolExecutor

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
    img_tensor = tf.image.decode_jpeg(camera_image.image)


    img_np = np.zeros_like(img_tensor.numpy())[:, :, 0].astype(np.float32)
    img_np[:] = np.nan

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

    # TODO: this seems to be wrong
    (width, height) = img_np.shape

    coordinates = projected_points_all_from_raw_data[:, 0:2].\
        astype(np.int32)

    img_np[coordinates[:, 1], coordinates[:, 0]] = \
        projected_points_all_from_raw_data[:, 2]

    dense_matrix = \
        poolingOverlap(img_np, ksize=(15, 10), stride=(1, 1), method='mean',
                       pad=True)

    xv, yv = np.meshgrid(range(0, width), range(0, height), indexing='ij')
    xv, yv = xv.reshape(-1,), yv.reshape(-1,)
    dense_vector = dense_matrix.reshape(-1, )
    valid_coordinates = dense_vector > 0.
    xv, yv = xv[valid_coordinates], yv[valid_coordinates]
    dense_vector = dense_vector[valid_coordinates]
    projected_matrix = np.vstack([yv, xv, dense_vector]).transpose()

    # project the points to an depth_np_arr
    depth_np_arr = projected_points_to_image(
        img_tensor.numpy().shape, projected_matrix
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
        depth_np_arr = depth_map_by_camera_name(
            frame, camera_dataset, points_all, cp_points_all
        )
        colorized_depth_image = colorize_np_arr(
            depth_np_arr, turbo_rgba
        )
        if camera_name not in _image_results:
            _image_results[camera_name] = []

        _image_results[camera_name].append(
            colorized_depth_image
        )
    return _image_results


def handle_file(input_filename, output_dir):
    input_path, input_fname = os.path.split(input_filename)
    segment_m = re.search(
        '(.+_with_camera_labels).tfrecord$', input_fname
    )
    if segment_m is None:
        return
    # create the output folder


    if not os.path.exists(os.path.join(output_dir, 'lidar_rgb')):
        os.mkdir(os.path.join(output_dir, 'lidar_rgb'))

    output_data_folder = os.path.join(output_dir, 'lidar_rgb', segment_m.group(1))
    if not os.path.exists(output_data_folder):
        os.mkdir(output_data_folder)

    try:
        dataset = tf.data.TFRecordDataset(input_filename, compression_type='')
    except BaseException as e:
        print('Error while reading file: {}'.format(e))
        return

    frame_id = 0
    for data in dataset:
        frame_start_time = time.time()
        frame = open_dataset.Frame()
        try:
            frame.ParseFromString(bytearray(data.numpy()))
        except BaseException as e:
            print('\tError: {}'.format(e))

        # rgb projection of images
        camera_names = {
            'front': open_dataset.CameraName.FRONT,
            # 'front_left': open_dataset.CameraName.FRONT_LEFT,
            # 'front_right': open_dataset.CameraName.FRONT_RIGHT,
            # 'side_left': open_dataset.CameraName.SIDE_LEFT,
            # 'side_right': open_dataset.CameraName.SIDE_RIGHT
        }

        _image_results = project_image_lidar_to_image_plane(frame, camera_names)
        for camera_name, images in _image_results.items():
            _fpath = os.path.join(
                output_data_folder, camera_name)
            if not os.path.exists(_fpath):
                mkdir_p(_fpath)
            for i, col_img in enumerate(images):
                _save_path = os.path.join(_fpath, 'frame_{:05d}.png'.format(i))
                col_img.save(_save_path)

        frame_id += 1
        print('\t Frame {:04d} handled took: {} sec'.format(
            frame_id-1, time.time() - frame_start_time
        ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('export_data')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory',
                        default='./output')
    parser.add_argument('--max_workers', type=int, default=4)
    args = parser.parse_args()

    executor = ThreadPoolExecutor(max_workers=args.max_workers)

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
    files = os.listdir(input_directory)
    num_files = len(files)
    print(f'Reading from {input_directory} found {num_files} files')

    # go through them
    for i, f in enumerate(files):
        print(f'({i}/{num_files}) Extracting lidar data from {f}')
        # print('{}/{} files parsed. Current file: {}'.format(
        #     i, len(files), f))
        # go through all the files
        _input_filename = os.path.join(input_directory, f)
        # handle_file(_input_filename, args.output_dir)
        executor.submit(handle_file, _input_filename, args.output_dir)

