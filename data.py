import os
import cv2
import numpy as np

class EuroSATDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def load_data(self):
        images = []
        labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            
            for img_name in os.listdir(cls_dir):
                if not img_name.endswith(('.jpg', '.png', '.jpeg')): 
                    continue
                
                img_path = os.path.join(cls_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img_flatten = img.flatten()
                
                images.append(img_flatten)
                labels.append(self.class_to_idx[cls_name])
                
        #转换为 Numpy 数组并归一化到 0-1
        X = np.array(images, dtype=np.float32) / 255.0
        Y = np.array(labels, dtype=np.int64)        
        return X, Y, self.classes