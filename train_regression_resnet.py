"""
Retrain the YOLO model for your own dataset.
训练旋转角度回归模型
"""
import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import regression_body_resnet
from yolo3.utils import get_random_data_angle

def _main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    # TensorFlow wizardry
    config = tf.ConfigProto()
    
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    K.tensorflow_backend.set_session(tf.Session(config=config))

    annotation_path = 'train_regression.txt'
    log_dir = 'logs/000/'

    input_shape = (224,224) # multiple of 32, hw

    #model = create_regression_model(input_shape, freeze_body=6, weights_path='model_data/trained_weights_final.h5') # make sure you know what you freeze
    model = create_regression_model(input_shape,load_pretrained=True,freeze_body=0, weights_path='logs/000/ep069-loss0.014-val_loss0.004.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    '''
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss=mean_squared_error)

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape,train=True),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape,train=False),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_regression_weights_resnet_stage1.h5')
    '''
    #Train with all layers
    
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-5), loss=mean_squared_error)

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape,train=True),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape,train=False),
                validation_steps=max(1, num_val//batch_size),
                initial_epoch=69,
                epochs=150,
                callbacks=[logging, checkpoint,early_stopping,reduce_lr])
        model.save_weights(log_dir + 'trained_regression_weights_resnet_final.h5')

def create_regression_model(input_shape,load_pretrained=True, freeze_body=0,
            weights_path=None):
    '''create the training model'''
    K.clear_session() # get a new session
    h, w = input_shape    

    model_body = regression_body_resnet()
    print('Create Regression Model body.')

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        # Freeze darknet53 body or freeze all but n last layers.
    if freeze_body>0:
        num = len(model_body.layers)-freeze_body
        for i in range(num): 
            model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    else:
        print('total {} layers.'.format(len(model_body.layers)))

    model = model_body
    return model

def data_generator(annotation_lines, batch_size, input_shape,train=False):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        sin_data = []
        cos_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, sin_angle, cos_angle = get_random_data_angle(annotation_lines[i], input_shape, train)
            #print(image.shape)
            image_data.append(image)
            sin_data.append(sin_angle)
            cos_data.append(cos_angle)
            i = (i+1) % n
        image_data = np.array(image_data)
        sin_data = np.array(sin_data)
        cos_data = np.array(cos_data)
        #print(angle_data)
        yield image_data, [sin_data , cos_data]

def data_generator_wrapper(annotation_lines, batch_size, input_shape,train=False):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape,train)

if __name__ == '__main__':
    _main()
