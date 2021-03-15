import random
import os
import shutil
import pathlib

# the original dataset downloaded from http://www.cs.uoi.gr/~marina/sipakmed.html
data_dir = "SPIKaMeD"
classes = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]
output_dir = "separated-data"

# train:val:test
ratio = [0.7, 0.2, 0.1]

def split_folder(data_dir, output_dir, ratio):
    # for each type of cell 
    for cell in classes:
        # locate cells path
        cell_path = os.path.join(data_dir, cell)
        files = os.listdir(cell_path)
        files = [os.path.join(cell_path, f) for f in files if f.endswith('.bmp')]

        # Make sure to always shuffle with a fixed seed so that the split is reproducible
        random.seed(230)
        files.sort()
        random.shuffle(files)

        # ratio for train, validation and test
        split_train = int(ratio[0] * len(files))
        split_val = split_train + int(ratio[1] * len(files))

        # split files 
        files_train = files[:split_train]
        files_val = files[split_train:split_val] 
        files_test = files[split_val:]
        files_type = [(files_train, "train"), (files_val, "val"), (files_test, "test")]

        # copy files into output directory
        for (files, folder_type) in files_type:
            full_path = os.path.join(output_dir, folder_type)
            full_path = os.path.join(full_path, cell)
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, full_path)

split_folder(data_dir, output_dir, ratio)