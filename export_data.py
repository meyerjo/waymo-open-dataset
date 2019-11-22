import argparse
import json
import os
import re

import numpy as np
import tensorflow as tf
from PIL import Image

tf.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt

from custom_utils import get_range_image, show_camera_image, show_range_image, \
    plot_points_on_image, rgba, poolingOverlap, projected_points_to_image


def project_image_lidar_to_image_plane(frame):
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

    _images = []
    # go through all the camera images
    for i, current_image in enumerate(frame.images):
        img_tensor = tf.image.decode_jpeg(current_image.image)
        img_np = np.zeros_like(img_tensor.numpy())[:, :, 0].astype(np.float32)
        img_np[:] = np.nan

        # The distance between lidar points and vehicle frame origin.
        points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        mask = tf.equal(cp_points_all_tensor[..., 0], current_image.name)

        cp_points_all_tensor = tf.cast(
            tf.gather_nd(
                cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

        projected_points_all_from_raw_data = tf.concat(
            [cp_points_all_tensor[..., 1:3], points_all_tensor],
            axis=-1).numpy()

        # TODO: this seems to be wrong
        (width, height) = img_np.shape

        coordinates = projected_points_all_from_raw_data[:, 0:2].astype(
            np.int32)

        img_np[coordinates[:, 1], coordinates[:, 0]] = \
            projected_points_all_from_raw_data[:, 2]

        dense_matrix = \
            poolingOverlap(img_np, ksize=(15, 10), stride=(1, 1), method='mean', pad=True)

        xv, yv = np.meshgrid(range(0, width), range(0, height), indexing='ij')

        xv = xv.reshape(-1, )
        yv = yv.reshape(-1, )
        dense_vector = dense_matrix.reshape(-1, )
        valid_coordinates = dense_vector > 0.
        xv = xv[valid_coordinates]
        yv = yv[valid_coordinates]
        dense_vector = dense_vector[valid_coordinates]

        projected_matrix = np.vstack([yv, xv, dense_vector]).transpose()

        colorized_depth_image = projected_points_to_image(
            img_tensor.numpy().shape, projected_matrix, rgba
        )

        _images.append(colorized_depth_image)
    return _images


def handle_file(DIR, fname, OUTPUT_DIR):
    segment_m = re.search(
        '^(.+)_with_camera_labels.tfrecord$', fname
    )
    # create the output folder
    output_data_folder = os.path.join(OUTPUT_DIR, segment_m.group(1))
    if not os.path.exists(output_data_folder):
        os.mkdir(output_data_folder)

    if not os.path.exists(os.path.join(output_data_folder, 'rgb')):
        os.mkdir(os.path.join(output_data_folder, 'rgb'))

    FILENAME = os.path.join(DIR, fname)
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

    frame_id = 0
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # frame.images are given
        for index, image in enumerate(frame.images):
            _save_path = os.path.join(
                output_data_folder, 'rgb', 'frame_{}_cam_{}.png'.format(
                    frame_id, index
                )
            )
            image_decode_jpg = tf.image.decode_jpeg(image.image)
            im = Image.fromarray(image_decode_jpg.numpy())
            im.save(_save_path)

        _save_path = os.path.join(
            output_data_folder, 'info'
        )
        if not os.path.exists(_save_path):
            os.mkdir(_save_path)

        data = []
        for i, img in enumerate(frame.images):
            velocities = {x[0].name: x[1] for x in img.velocity.ListFields()}
            pose_data = {x[0].name: x[1] for x in img.pose.ListFields()}
            data.append({
                'id': i,
                'name': img.name,
                'velocities': velocities,
                'pose': pose_data,
                'shutter': img.shutter,
                'pose_timestamp': img.pose_timestamp,
                'camera_trigger_time': img.camera_trigger_time,
                'camera_readout_done_time': img.camera_readout_time,
            })
        with open(os.path.join(
                _save_path, 'frame_{}.json'.format(frame_id)), 'w') as f:
            f.write(json.dumps(data))


        if not os.path.exists(os.path.join(output_data_folder, 'lidar_rgb')):
            os.mkdir(os.path.join(output_data_folder, 'lidar_rgb'))
        # rgb projection of images
        _images = project_image_lidar_to_image_plane(frame)
        for i, proj_img in enumerate(_images):
            _save_path = os.path.join(
                output_data_folder, 'lidar_rgb',
                'frame_{}_cam_{}.png'.format(
                    frame_id, i
                )
            )
            proj_img.save(_save_path)

        # write the summary files

        # output bounding boxes

        frame_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('export_data')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory',
                        default='./output')
    args = parser.parse_args()

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

    # go through them
    for i, f in enumerate(files):
        if (i % 10 == 0):
            print('{}/{} files parsed. Current file: {}'.format(
                i, len(files), f))
        # go through all the files
        handle_file(input_directory, f, args.output_dir)

        if i >= 1:
            break
