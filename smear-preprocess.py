import random
import os
import shutil
import pathlib

def split_folder(data_dir, output_dir, ratio):
    # for each type of cell 
    for cell in classes:
        # locate cells path
        cell_path = os.path.join(data_dir, cell)
        files = os.listdir(cell_path)

        images = []
        labels = []

        for f in files:
            if '-d' in f:
                labels.append(os.path.join(cell_path, f))
            else:
                images.append(os.path.join(cell_path, f))


        # Make sure to always shuffle with a fixed seed so that the split is reproducible
        images.sort()
        labels.sort()
        random.seed(230)
        random.shuffle(images)
        random.seed(230)
        random.shuffle(labels)

        # ratio for train and test
        split_train = int(ratio[0] * len(images))

        # split files 
        images_train = images[:split_train]
        images_test = images[split_train:]

        label_train = labels[:split_train]
        label_test = labels[split_train:]


        files_type = [(images_train, "train/images"), (images_test, "test/images"), (label_train, "train/labels"), (label_test, "test/labels")]

        # copy files into output directory
        for (files, folder_type) in files_type:
            full_path = os.path.join(output_dir, folder_type)
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, full_path)


data_dir = "smear"
classes = ["carcinoma_in_situ", "light_dysplastic", "moderate_dysplastic", "normal_columnar", "normal_intermediate", "normal_superficiel", "severe_dysplastic"]
output_dir = "smear-segmentation"
# train:test
ratio = [0.8, 0.2]
split_folder(data_dir, output_dir, ratio)