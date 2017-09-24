from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from keras.optimizers import Adam
from parameters import params
from utils import preprocess_images
from sklearn.model_selection import train_test_split
from load_data import load_driving_log_csv, generator



# Define model ≡ NVIDIA model
#
# See: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

model = Sequential()

model.add(Lambda(preprocess_images, input_shape=(160,320,3), output_shape=(66,200,1)))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation='elu'))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation='elu'))

model.add(Convolution2D(48,5,5,subsample=(2,2),activation='elu'))

model.add(Convolution2D(64,3,3,activation='elu'))

model.add(Convolution2D(64,3,3,activation='elu'))

model.add(Flatten())

model.add(Dense(100))

model.add(Dense(50))

model.add(Dense(10))

model.add(Dense(1))

adam = Adam(lr=params['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam)



# Load CSV file which lines point to sampled data.
symbolic_data = load_driving_log_csv()

# Split the sampled data into training (80%) and validation sets (20%).
symbolic_train_data, symbolic_valid_data = train_test_split(symbolic_data, test_size=0.2)

# Take into account data augmentation when counting the samples.
nb_train_samples = params['data_generation_factor']*len(symbolic_train_data)
nb_valid_samples = params['data_generation_factor']*len(symbolic_valid_data)

# Build the generators so that the data set does not need to be entirely loaded
# in memory at once.
train_generator = generator(params, symbolic_train_data)
valid_generator = generator(params, symbolic_valid_data)

# Train the model.
model.fit_generator(
    train_generator,
    samples_per_epoch = nb_train_samples,
    validation_data = valid_generator,
    nb_val_samples = nb_valid_samples,
    nb_epoch=params['nb_epoch']
)



# Save Model To FS.
model.save(params['model_path'])
