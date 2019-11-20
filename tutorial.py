import os
import re

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt

from custom_utils import get_range_image, show_camera_image, show_range_image, \
    plot_points_on_image, rgba, poolingOverlap, projected_points_to_image

# FILENAME = '/home/meyerjo/code/waymo-od/tutorial/frames'
OUTPUT_DIR = os.path.join(
    os.path.expanduser('~'), 'Downloads', 'waymodataset'
)
files = os.listdir(OUTPUT_DIR)


def camera_response_times(dataset):
    frame_triggered = []
    frame_readout_time = []
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        frame_readout_time.append(frame.images[open_dataset.CameraName.FRONT].camera_readout_done_time)
        frame_triggered.append(frame.images[open_dataset.CameraName.FRONT].camera_trigger_time)
    earliest_trigger_time = min(frame_triggered)
    frame_triggered = [f - earliest_trigger_time for f in frame_triggered]
    # still using the trigger time as 'anchor' point in order to see the offset
    frame_readout_time = [f - earliest_trigger_time for f in frame_readout_time]

    plt.plot(frame_triggered, np.ones_like(frame_triggered), 'x')
    plt.plot(frame_readout_time, np.ones_like(frame_readout_time)*2, 'x')
    plt.xlabel('time')
    plt.ylabel('type')
    plt.yticks([1, 2], ['trigger', 'readout'])
    plt.show()

    time_trigger = [frame_triggered[::-1][i] - frame_triggered[::-1][i + 1] for i in
                    range(len(frame_triggered[::-1]) - 1)]

    print('trigger_time: {} +- {}; min trigger time: {}, max trigger time: {}'.format(
        np.mean(time_trigger),
        np.std(time_trigger),
        np.min(time_trigger),
        np.max(time_trigger)
    ))


    return frame_triggered, frame_readout_time


def handle_file(DIR, fname, OUTPUT_DIR):
    segment_m = re.search(
        '^(.+)_with_camera_labels.tfrecord$', fname
    )
    plt_fig_folder = os.path.join(
        OUTPUT_DIR, segment_m.group(1)
    )
    if not os.path.exists(plt_fig_folder):
        os.mkdir(plt_fig_folder)

    FILENAME = os.path.join(DIR, fname)
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    i = 0

    camera_response_times(dataset)

    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        print(
            'frame.context: {} ({}, {}, {})'.format(
                frame.context.name,
                frame.context.stats.location,
                frame.context.stats.time_of_day,
                frame.context.stats.weather,
            )
        )

        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        plt.figure(figsize=(25, 13))
        for index, image in enumerate(frame.images):
            show_camera_image(
                frame, image, frame.camera_labels, [3, 3, index + 1]
            )
        plt.show()
        i += 1

        names = ['TOP', 'FRONT', 'REAR', 'SIDE_LEFT', 'SIDE_RIGHT']
        for _laser_name in names:
            _laser_name_id = getattr(open_dataset.LaserName, _laser_name, None)
            if _laser_name_id is None:
                continue

            range_image_top_return_0 = get_range_image(range_images, _laser_name_id, 0)
            range_image_top_return_1 = get_range_image(range_images, _laser_name_id, 1)
            # 1,4 correspond to the start index
            plt.figure(figsize=(32, 20))
            show_range_image(range_image_top_return_0, 1)
            show_range_image(range_image_top_return_1, 4)
            ax = plt.subplot(*[8, 1, 8])
            ax.text(0.5, 0.5, 'Sensor data from lidar: {}'.format(_laser_name), clip_on=False, fontsize=32)
            plt.grid(False)
            plt.axis('off')
            plt.show()

        # convert the points to a point cloud from the first laser return
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=0)

        points_second, cp_points_second = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)
        points_second_all = np.concatenate(points_second, axis=0)
        cp_points_second_all = np.concatenate(cp_points_second, axis=0)

        points_all = np.concatenate([points_all, points_second_all])
        cp_points_all = np.concatenate([cp_points_all, cp_points_second_all])

        for i, current_image in enumerate(frame.images):
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

            img_tensor = tf.image.decode_jpeg(current_image.image)
            img_np = np.zeros_like(img_tensor.numpy())[:, :, 0].astype(np.float32)
            img_np[:] = np.nan

            # TODO: this seems to be wrong
            (width, height) = img_np.shape

            coordinates = projected_points_all_from_raw_data[:, 0:2].astype(np.int32)

            img_np[coordinates[:, 1], coordinates[:, 0]] = projected_points_all_from_raw_data[:, 2]

            dense_matrix = poolingOverlap(img_np, ksize=(15, 10),
                                          stride=(1, 1), method='mean', pad=True)

            xv, yv = np.meshgrid(
                range(0, width), range(0, height), indexing='ij'
            )


            xv = xv.reshape(-1,)
            yv = yv.reshape(-1,)
            dense_vector = dense_matrix.reshape(-1,)
            valid_coordinates = dense_vector > 0.
            xv = xv[valid_coordinates]
            yv = yv[valid_coordinates]
            dense_vector = dense_vector[valid_coordinates]

            projected_matrix = np.vstack([yv, xv, dense_vector]).transpose()

            colorized_depth_image = projected_points_to_image(
                tf.image.decode_jpeg(current_image.image).numpy().shape,
                projected_matrix, rgba
            )

            plt.figure()
            # plt.scatter(yv[:-1], xv[:-1], c=colorized_m)
            plt.imshow(colorized_depth_image)
            # plt.gca().invert_yaxis()
            plt.title('XYZ')
            plt.show()



            plt.figure(figsize=(64, 20))
            plt.tight_layout()
            plot_points_on_image(
                projected_matrix, colorized_depth_image,
                current_image, rgba, point_size=5.0)
            plt.show()
            # plt.show()
            plt.savefig('./camera_{}.png'.format(i))


        break




for i, f in enumerate(files):
    handle_file(OUTPUT_DIR, f, './output/')

    if i >= 1:
        break
