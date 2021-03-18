import random
import os
import shutil
import pathlib

data_dir = "smear"
classes = ["carcinoma_in_situ", "light_dysplastic", "moderate_dysplastic", "normal_columnar", "normal_intermediate", "normal_superficiel", "severe_dysplastic"]
output_dir = "separated-smear"

# train:test
ratio = [0.2, 0.8]

def split_folder(data_dir, output_dir, ratio):
    # for each type of cell 
    for cell in classes:
        # locate cells path
        cell_path = os.path.join(data_dir, cell)
        files = os.listdir(cell_path)
        files = [os.path.join(cell_path, f) for f in files if not ("-d") in f]

        # Make sure to always shuffle with a fixed seed so that the split is reproducible
        random.seed(230)
        files.sort()
        random.shuffle(files)

        # ratio for train and test
        split_train = int(ratio[0] * len(files))

        # split files 
        files_train = files[:split_train]
        files_test = files[split_train:]
        files_type = [(files_train, "train"), (files_test, "test")]

        # copy files into output directory
        for (files, folder_type) in files_type:
            full_path = os.path.join(output_dir, folder_type)
            full_path = os.path.join(full_path, cell)
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, full_path)

split_folder(data_dir, output_dir, ratio)