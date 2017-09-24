import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from skimage.transform import resize
from parameters import params
from utils import preprocess_images, get_filename
import matplotlib.image as mpimg

# Map of the FS → Load data.


def load_driving_log_csv(params):
    """params → symbolic_points ≡ [[center_img_path, left_img_path, right_img_path, angle, …], …]"""

    symbolic_points = None

    def helper():
        nonlocal symbolic_points
        if symbolic_points is None:
            symbolic_points = []
            for data_path in params['data']:
                csv_path = data_path + '/driving_log.csv'
                with open(csv_path) as csvfile:
                    reader = csv.reader(csvfile)
                    for symbolic_point in reader:
                        for idx in range(3):
                            symbolic_point[idx] = data_path + '/IMG/' + get_filename(symbolic_point[idx])
                        symbolic_point[3] = float(symbolic_point[3])
                        symbolic_points.append(symbolic_point)
        return symbolic_points
    return helper
load_driving_log_csv = load_driving_log_csv(params)



def generator(params, symbolic_points):
    while True:
        symbolic_points = shuffle(symbolic_points)

        for offset in range(0, len(symbolic_points), params['batch_size']):
            batch_symbolic_points = symbolic_points[offset:offset+params['batch_size']]
            images = []
            angles = []
            for symbolic_point in batch_symbolic_points:
                [center_img_path, left_img_path, right_img_path, angle, *rest] = symbolic_point
                angle = float(angle)
                jitter_x = 0
                jitter_angle = 0.
                if -params['angle_interval'] <= angle or angle <= params['angle_interval']:
                    jitter_x = int(np.random.uniform()*params['trans_interval'] - params['trans_interval']/2)
                    jitter_angle = jitter_x*params['angle_per_px']
                angle = angle + jitter_angle
                point_images_angles = [angle + correction for correction in [0.,params['correction'], -params['correction']]]

                point_images = [cv2.imread(path) for path in [center_img_path, left_img_path, right_img_path]]
                for n,image in enumerate(point_images):
                    left_limit = params['trans_interval']/2
                    right_limit = left_limit + 320 - params['trans_interval']
                    point_images[n] = resize(image[:,int(left_limit + jitter_x):int(right_limit + jitter_x)], (160,320,3))

                images.extend(point_images)
                angles.extend(point_images_angles)

                images.extend([np.fliplr(img) for img in point_images])
                angles.extend([-angle for angle in point_images_angles])

            yield shuffle(np.array(images), np.array(angles))
