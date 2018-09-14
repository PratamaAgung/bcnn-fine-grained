import glob
import os

def write_append_to_file(file_path, payload):
    if os.path.exists(file_path):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    with open(file_path, append_write) as handle:
        handle.write(payload)


all_images_folder = glob.glob("data_gemastik/*")
train_txt = "train_test/new_train.txt"
val_txt = "train_test/new_val.txt"
test_txt = 'train_test/new_test.txt'

index = 0
for folder in all_images_folder:
    images_in_folder = glob.glob(folder + "/*")
    train_data = images_in_folder[: int(0.8 * len(images_in_folder))]
    val_data = train_data[ : int(0.1 * len(images_in_folder))]
    test_data = images_in_folder[int(0.8 * len(images_in_folder)) :]

    for train in train_data:
        write_append_to_file(train_txt, train + " " + str(index) + "\n")

    for test in test_data:
        write_append_to_file(test_txt, test + " " + str(index) + "\n")

    for val in val_data:
        write_append_to_file(val_txt, val + " " + str(index) + "\n")
    index += 1
