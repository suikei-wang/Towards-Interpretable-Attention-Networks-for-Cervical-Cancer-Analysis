import cv2
import numpy as np
import os
import pathlib

if __name__ == '__main__':
    data_dir = 'separated-data'
    classes = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]
    type_dir = ['train', 'val', 'test']
    save_dir = 'masked-separated-data'
    save_seg_dir = 'seg-separated-data'
    for t in type_dir:
        for cell in classes:
            cell_path = os.path.join(data_dir, t, cell)
            pathlib.Path(os.path.join(save_dir, t, cell)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.join(save_seg_dir, t, cell)).mkdir(parents=True, exist_ok=True)
            files = os.listdir(cell_path)
            files = [os.path.join(cell_path, f) for f in files if f.endswith('.bmp')]

            for f in files:
                img = cv2.imread(f)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                ret, thresh = cv2.threshold(gray, 0 , 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                f = f.replace('separated-data/', '')
                f = f.replace('bmp', 'png')

                # cv2.imwrite(os.path.join(save_dir, f), thresh)

                img[gray >= thresh] = 0
                cv2.imwrite(os.path.join(save_seg_dir, f), img)