'''
大盘处理
'''
import numpy as np
import cv2
import os
import sys
import math

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
    anno_dir = os.path.join(os.path.abspath('.'),os.path.join('imges','annotation'))
    txts = os.listdir(anno_dir)

    meta = []

    for txt in txts:
        #print(txt)
        if (str(txt).endswith('.txt')):
            anno = dict()
            with open(os.path.join(anno_dir,txt),'r') as f:
                anno['name'] = str(txt[:-4])+'.jpg'

                img = cv2.imread('imges/'+anno['name'])
                shape = img.shape
                scale_x = resize_shape[0]/shape[1]
                scale_y = resize_shape[1]/shape[0]
                scale = (scale_x,scale_y)

                xmin = f.readline().strip()
                ymin = f.readline().strip()
                xmax = f.readline().strip()
                ymax = f.readline().strip()
                box = [int(x) for x in (xmin,ymin,xmax,ymax)]
                box = scale_box(box,scale)
                #print([shape,box])

                anno['meter_box'] = box

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
                anno['angle']=angle
                print(angle)
                anno['long_number_points'] = long_number_points
                anno['small_meter_points']=small_meter_points
                
                meta.append(anno)

    if True:
        with open('train.txt','w+') as file:
            for m in meta:
                file.write('data/images/'+m['name'])
                file.write(' ')
                file.write(box_to_str(m['meter_box']))
                file.write(',')
                file.write('0')
                file.write('\n')
    if False:
        with open('keras-yolo3/train_regression.txt','w+') as file:
            for m in meta:
                file.write('data/images/'+m['name'])
                file.write(' ')
                file.write(box_to_str(m['meter_box']))
                file.write(' ')
                file.write(str(m['angle']))
                file.write('\n')





                
    