from keras.models import Model
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, regression_body
from train_regression import create_regression_model
from yolo3.utils import crop
from process_anno_angle import rotate,rotate_by_vec
import cv2
import numpy as np
import math
import scipy.misc
from math import *

if __name__ =='__main__':
    input_shape=(416,416)
    model = create_regression_model(input_shape, freeze_body=3, weights_path='logs/111/ep108-loss0.399-val_loss0.384.h5')
    with open('train_regression.txt','r') as file:
        while True:
            line = file.readline().split(' ')
            name = line[0]
            image = cv2.imread(name)
            cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
            aug_angle = float(line[1])
            box = tuple([int(x) for x in line[2].split(',')])
            org_angle = float(line[3].strip())

            #crop
            image = rotate(image,aug_angle*180)
            image = image[box[1]:box[3],box[0]:box[2],:]
            image = cv2.resize(image,(416,416))

            gt = (aug_angle + org_angle)
            #print()
           
            
            if True:
                data = np.expand_dims(image,0)
                result = model.predict(data)
                #print(result)

                theta_sin=float(result[0])
                theta_cos=float(result[1])
                
                #print(rotate_m.shape)

                a1 = asin(theta_sin)/math.pi*180
                a2 = acos(theta_cos)/math.pi*180

                print('gt sin,cos:',[sin(pi*gt),cos(pi*gt)])
                print('pt sin,cos:',[theta_sin,theta_cos])
                #print(0.509*180)
                
                #print([theta_sin,theta_cos],[a1,a2])
                print('pt angle:')
                print('use sin:',a1,' or ',180.0-a1)
                print('use cos:',a2,' or ',a2+180)
                print('----------------')
                print('gt angle:',gt*180)
                print('\n')

                #image = image.astype(np.float32)
                #print(image.shape,image.dtype)

                '''
                degree = 60
                theta = degree/180*pi
                print(cv2.getRotationMatrix2D((416/2,416/2),degree,1))
                a = np.array([
                    [1,0,0],
                    [0,1,0],
                    [-208,-208,1]])
                b = np.array([
                    [cos(theta),-sin(theta),0],
                    [sin(theta),cos(theta),0],
                    [0,0,1]])
                c = np.array([
                    [1,0,0],
                    [0,1,0],
                    [208,208,1]])
                rot_m = np.dot(a,np.dot(b,c))
                print(rot_m.T)
                '''

                rot_image = cv2.resize(rotate_by_vec(image,theta_sin,theta_cos),input_shape)
                image = np.hstack((image,rot_image))
            
            cv2.imshow('input_image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
