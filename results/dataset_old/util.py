import os
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def print_some_errors(brand_err, base_path, num = 5):
    brand_err = shuffle(brand_err)
    
    fig = plt.figure(figsize=(20, 10))
    
    paths = []
    
    for i in range(num):
        row = brand_err.iloc[i]
        img_path = os.path.join(base_path, row['folder'], 'shot.png')
        paths.append(img_path)
        with Image.open(img_path) as img:
            ax1 = fig.add_subplot(1,num,i+1)
            ax1.title.set_text(f"{row['true_brand']} -> p:{row['pred_brand']}")
            ax1.imshow(img)
            
    return paths

def precision_sensitivity(confusion_matrix, confusion_matrix_rec):
    tp, tp_rec = confusion_matrix[1][1], confusion_matrix_rec[1][1]
    fp, fp_rec = confusion_matrix[1][0], confusion_matrix_rec[1][0]
    fn, fn_rec = confusion_matrix[0][1], confusion_matrix_rec[0][1]
    ppv = tp / (tp + fp)
    ppv_rec = tp_rec / (tp_rec + fp_rec)
    print(f"Precision: {ppv}, Precision brand: {ppv_rec}")
    tpr = tp / (tp + fn)
    tpr_rec = tp_rec / (tp_rec + fn_rec)
    print(f"Sensitivity: {tpr}, Sensitivity brand: {tpr_rec}")