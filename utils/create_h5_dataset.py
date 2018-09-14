from tflearn.data_utils import build_hdf5_image_dataset
import h5py

new_train = "train_test/new_train.txt"
new_val = "train_test/new_val.txt"
new_test = "train_test/new_test.txt"

# image_shape option can be set to different values to create images of different sizes
build_hdf5_image_dataset(new_val, image_shape=(448, 448), mode='file', output_path='new_val.h5', categorical_labels=True, normalize=False)
print('Done creating new_val.h5')
build_hdf5_image_dataset(new_test, image_shape=(448, 448), mode='file', output_path='new_test.h5', categorical_labels=True, normalize=False)
print ('Done creating new_test.h5')
build_hdf5_image_dataset(new_train, image_shape=(448, 448), mode='file', output_path='new_train.h5', categorical_labels=True, normalize=False)
print ('Done creating new_train_448.h5')
