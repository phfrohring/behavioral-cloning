root_dir = '/Users/nomos/Documents/udacity/self_driving_car/CarND-Behavioral-Cloning-P3'
correction = 0.15
params = {
    'correction': correction,
    'crop_y_top': 62, # px
    'crop_y_bottom': 50, # px
    'angle_interval': correction + 0.05,
    'trans_interval': 2*60, # px
    'angle_per_px': -0.008,
    'data_generation_factor': 6,
    'resize_output': (66, 200),
    'learning_rate': 0.0001,
    'nb_epoch': 9,
    'batch_size': 32,
    'data': [
        './data/udacity',
        './data/left_turn_without_border_1',
        './data/left_turn_without_border_2',
        './data/left_turn_without_border_3',
        './data/left_turn_without_border_4',
        './data/sharp_turn_right_1',
        './data/recovery_bridge_1',
        './data/recovery_left_turn_without_border_1',
        './data/recovery_left_turn_without_border_2'
    ],
    'model_path': 'model.h5'
}
