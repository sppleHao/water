'''
角度预处理
'''
import numpy as np
import cv2
import os
import sys
import math
from math import *
import scipy.misc
from yolo import YOLO

#恢复原图框
def scale_box(box,scale):
    new_box = (box[0]/scale[0],box[1]/scale[1],box[2]/scale[0],box[3]/scale[1])
    return [int(x) for x in new_box]

#恢复原图点
def scale_point(point,scale):
    new_p = (point[0]/scale[0],point[1]/scale[1])
    return [int(x) for x in new_p]

def box_to_str(box):
    return str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3])

#逆时针旋转图片而不裁剪
def rotate(img,degree):
    height,width=img.shape[:2]
    heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
    widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))

    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)

    matRotation[0,2] +=(widthNew-width)/2 
    matRotation[1,2] +=(heightNew-height)/2 

    imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))

    return imgRotation

def rotate_by_vec(img,sin_theta,cos_theta):
    height,width=img.shape[:2]
    #print(img.shape)
    heightNew=int(width*fabs(sin_theta)+height*fabs(cos_theta))
    widthNew=int(height*fabs(sin_theta)+width*fabs(cos_theta))

    a=np.array([
                [1,0,0],
                [0,1,0],
                [-width/2,-height/2,1]])
    b=np.array([
                [cos_theta,sin_theta,0],
                [-sin_theta,cos_theta,0],
                [0,0,1]],dtype='float32'
                )
    c=np.array([
                [1,0,0],
                [0,1,0],
                [width/2,height/2,1]])
    
    matRotation = np.dot(a,np.dot(b,c))
    matRotation = matRotation.T
    matRotation = matRotation[:2,:]
    #print(matRotation)

    matRotation[0,2] +=(widthNew-width)/2 
    matRotation[1,2] +=(heightNew-height)/2 

    imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))

    return imgRotation        

#旋转角度(顺时针)
def compute_angel(point1,point2):
    """
    计算需要顺时针旋转才能转正的角度
    输出[0-2],为弧度值除以pi
    """
    p1 = np.array(point1,dtype=float)
    p2 = np.array(point2,dtype=float)

    #表盘左侧到右侧的向量
    v = p2-p1

    #x轴的向量
    up_vector = np.array([1,0],dtype=float)
    v_norm = np.linalg.norm(v)

    #夹角的弧度值
    cos_theta = np.arccos(np.dot(v,up_vector)/v_norm)

    #left y > right y , 夹角为顺时针
    if(point1[1]>point2[1]):
        cos_theta = cos_theta/math.pi
    else:
        cos_theta = 2-cos_theta/math.pi
    return round(float(cos_theta),3)

if __name__ == '__main__':
    resize_shape = (1200,800)
    anno_dir = os.path.join(os.path.abspath('.'),os.path.join('data','images','annotation'))
    txts = os.listdir(anno_dir)

    yolo = YOLO()

    meta = []

    for txt in txts:
        #print(txt)
        if (str(txt).endswith('.txt')):
            with open(os.path.join(anno_dir,txt),'r') as f:
                img = cv2.imread('data/images/'+str(txt[:-4])+'.jpg')
                shape = img.shape
                scale_x = resize_shape[0]/shape[1]
                scale_y = resize_shape[1]/shape[0]
                scale = (scale_x,scale_y)

                #无用-----------------------
                xmin = f.readline().strip()
                ymin = f.readline().strip()
                xmax = f.readline().strip()
                ymax = f.readline().strip()
                #box = [int(x) for x in (xmin,ymin,xmax,ymax)]
                #box = scale_box(box,scale)
                #print([shape,box])
                xlist = [int(x) for x in f.readline().strip().split('\t')]
                ylist = [int(x) for x in f.readline().strip().split('\t')]

                long_number_points = []
                angle_points = []
                small_meter_points = []
                for i in range(len(xlist)):
                    point = (xlist[i],ylist[i])
                    if i<4:
                        long_number_points.append(scale_point(point,scale))
                    elif i<6:
                        angle_points.append(point)
                    else:
                        small_meter_points.append(scale_point(point,scale))

                angle = compute_angel(angle_points[0],angle_points[1])
                
                #anno['long_number_points'] = long_number_points
                #anno['small_meter_points']=small_meter_points

                per_angle = 10
                                 
                for i in range(int(360/per_angle)):
                    image = rotate(img,i*per_angle) #逆时针旋转30、60、。。。
                    boxes = yolo.detect_result(image)
                    if len(boxes)>0:
                        print(boxes[0])
                        anno = dict()
                        anno['name'] = str(txt[:-4])+'.jpg'
                        anno['meter_box'] = boxes[0] #(left,top,right,bottom)
                        anno['aug_angle'] = i*per_angle/360
                        anno['ori_angle']=angle
                        meta.append(anno)
                    else:
                        print('cannot detect')

    with open('train_regression.txt','w+') as file:
        for m in meta:
            file.write('data/images/'+m['name'])
            file.write(' ')
            file.write(str(m['aug_angle']))
            file.write(' ')                
            file.write(box_to_str(m['meter_box']))
            file.write(' ')
            file.write(str(m['ori_angle']))
            file.write('\n')





                
    