'''
得到旋转图片
'''
from keras.models import Model
from train_regression_resnet import create_regression_model
from process_anno_angle import rotate,rotate_by_vec
import os
import cv2
import numpy as np
import math
import scipy.misc
from yolo import YOLO
from math import *

if __name__ =='__main__':
    input_shape=(224,224)
    model = create_regression_model(input_shape,load_pretrained=True,freeze_body=0, weights_path='logs/resnet_angel/trained_regression_weights_resnet_final.h5')
    yolo = YOLO()
    test_folder = os.path.join(os.path.abspath('.'),os.path.join('data','test_images'))
    new_folder = os.path.join(os.path.abspath('.'),os.path.join('data','new'))
    images = os.listdir(test_folder)
    
    for img in images:
        image = cv2.imread(os.path.join(test_folder,img))
        cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
        boxes = yolo.detect_result(image)
        
        if len(boxes)>0:
            box = boxes[0]
            #crop
            image = image[box[1]:box[3],box[0]:box[2],:]

            image = cv2.resize(image,input_shape)
            data = np.expand_dims(image,0)
            
            result = model.predict(data)

            pt_sin_ori=float(result[0])
            pt_cos_ori=float(result[1])

            r =sqrt(pt_sin_ori* pt_sin_ori + pt_cos_ori* pt_cos_ori)

            #print('pt sin,cos:',[pt_sin_ori,pt_cos_ori,r])

            pt_sin = pt_sin_ori / r
            pt_cos = pt_cos_ori / r

            #print('pt sin,cos:',[pt_sin,pt_cos])
                

            a1 = asin(pt_sin)/math.pi*180
            a2 = acos(pt_cos)/math.pi*180

            #print('gt sin,cos:',[sin(pi*gt),cos(pi*gt)])
            
                
            #print('pt angle:')
            #print('use sin:',a1,' or ',180.0-a1)
            #print('use cos:',a2,' or ',a2+180)
            #print('gt angle:',gt*180)
            #print('\n')

            #image = image.astype(np.float32)
            #print(image.shape,image.dtype)

            rot_image = cv2.resize(rotate_by_vec(image,pt_sin,pt_cos),(600,600))
            cv2.imwrite(os.path.join(new_folder,img),rot_image)
