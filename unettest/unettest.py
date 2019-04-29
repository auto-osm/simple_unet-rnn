import numpy as np 
import os
import scipy.io as sio
import skimage.io as io
import skimage.transform as trans
from PIL import Image
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.utils.vis_utils import plot_model
from keras.layers.convolutional_recurrent import ConvLSTM2D
base_dir1='D://py//unettest//data//membrane//train//image'
base_dir2='D://py//unettest//data//membrane//train//label'
def read_image(imageName):
    im = Image.open(imageName).convert('L')
    data = np.array(im)
    return data
text1=os.listdir(base_dir1)
text2=os.listdir(base_dir2)
images=[]
labels=[]
x_times=[]
y_times=[]
inputs=[]
x1=[]
x2=[]
x3=[]
y1=[]
for textPath in text1:
    for fn in os.listdir(os.path.join('D://py//unettest//data//membrane//train//image', textPath)):
        if fn.endswith('.png'):
            fd = os.path.join(base_dir1, textPath, fn)
            images.append(read_image(fd))
            #labels.append(textPath)
for textPath in text2:
    for fn in os.listdir(os.path.join('D://py//unettest//data//membrane//train//label', textPath)):
        if fn.endswith('.png'):
            fd = os.path.join(base_dir2, textPath, fn)
            labels.append(read_image(fd))
x=np.array(images)
y=np.array(labels)
'''
for i in range(10):
	x1.append(np.reshape(x[i]/255,[256,256,1]))
for i in range(10):
	x2.append(np.reshape(x[i+10]/255,[256,256,1]))
for i in range(10):
	x3.append(np.reshape(x[i+20]/255,[256,256,1]))
for i in range(10):
	y1.append(np.reshape(y[i+10]/255,[256,256,1]))
'''
for i in range(2,60):
	x1.append(np.reshape(x[i-2]/255,[256,256,1]))
	x2.append(np.reshape(x[i-1]/255,[256,256,1]))
	x3.append(np.reshape(x[i]/255,[256,256,1]))
	y1.append(np.reshape(y[i-2]/255,[256,256,1]))
#print(x)
#print('************')
#print(y)
'''
for i in range(0,30,5):
	x_time=np.concatenate([x[i],x[i+1]],axis=0)
	x_time=np.concatenate([x_time,x[i+2]],axis=0)
	x_time=np.concatenate([x_time,x[i+3]],axis=0)
	x_time=np.concatenate([x_time,x[i+4]],axis=0)
	x_times.append(np.reshape(x_time,[5,256,256,1]))
#print(x_times[0])
#fp=open('test.txt','w')
#fp.write(str(x_times[0]))
#fp.close()
#print(x_times[0].shape)
for i in range(0,30,5):
	y_time=np.concatenate([y[i],y[i+1]],axis=0)
	y_time=np.concatenate([y_time,y[i+2]],axis=0)
	y_time=np.concatenate([y_time,y[i+3]],axis=0)
	y_time=np.concatenate([y_time,y[i+4]],axis=0)
	y_times.append(np.reshape(y_time,[5,256,256,1]))
for i in range(6):
	z_time=np.concatenate([x_times[i],y_times[i]],axis=0)
	inputs.append(np.reshape(z_time,[2,5,256,256,1]))
#print(inputs[0].shape)
'''
'''
model=Sequential()

model.add(TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')))
model.add(TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')))
model.add(TimeDistributed(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')))
model.add(TimeDistributed(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')))
  
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')))
model.add(TimeDistributed(Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(UpSampling2D(size=(2, 2))))
model.add(ConvLSTM2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True))


  

model.add(ConvLSTM2D(256, 3, padding='same', return_sequences=True))
model.add(ConvLSTM2D(256, 3, padding='same', return_sequences=True))
model.add(TimeDistributed(UpSampling2D(size=(2, 2))))
model.add(ConvLSTM2D(128, 2, padding='same',return_sequences=True))


model.add(ConvLSTM2D(128, 3, padding='same', return_sequences=True))
model.add(ConvLSTM2D(128, 3, padding='same', return_sequences=True))
model.add(TimeDistributed(UpSampling2D(size=(2, 2))))
model.add(ConvLSTM2D(64, 2, padding='same', return_sequences=True))


model.add(ConvLSTM2D(64, 3, padding='same', return_sequences=True))
model.add(ConvLSTM2D(64, 3, padding='same', return_sequences=True))
model.add(TimeDistributed(Conv2D(2, 3, activation='relu', padding='same')))
 
model.add(TimeDistributed(Conv2D(1, 1,activation='softmax', padding='same')))

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
   
model.fit(inputs,inputs,batch_size=32)
'''
'''
def unet1(pretrained_weights = None,input_size = (None,256,256,1)):
    inputs = Input(input_size)
    conv1 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(inputs)
    conv1 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool1)
    conv2 = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    conv3 = TimeDistributed(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool2)
    conv3 = TimeDistributed(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv3)
    # pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    # conv4 = TimeDistributed(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool3)
    # conv4 = TimeDistributed(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4)
    drop4 = TimeDistributed(Dropout(0.5))(conv3)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(drop4)
    conv5 = TimeDistributed(Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool4)
    conv5 = TimeDistributed(Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv5)
    drop5 = TimeDistributed(Dropout(0.5))(conv5)
    up6 = ConvLSTM2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(
    TimeDistributed(UpSampling2D(size=(2, 2)))(drop5))
    merge6 = concatenate([drop4, up6], axis=4)
    # conv6 = ConvLSTM2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(merge6)
    # conv6 = ConvLSTM2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(conv6)
    # up7 = ConvLSTM2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(
    # TimeDistributed(UpSampling2D(size=(2, 2)))(conv6))
    merge7 = concatenate([conv3, up6], axis=4)
    conv7 = ConvLSTM2D(256, 3, padding='same', return_sequences=True)(merge7)
    conv7 = ConvLSTM2D(256, 3, padding='same', return_sequences=True)(conv7)
    up8 = ConvLSTM2D(128, 2, padding='same',return_sequences=True)(
    TimeDistributed(UpSampling2D(size=(2, 2)))(conv7))
    merge8 = concatenate([conv2, up8], axis=4)
    conv8 = ConvLSTM2D(128, 3, padding='same', return_sequences=True)(merge8)
    conv8 = ConvLSTM2D(128, 3, padding='same', return_sequences=True)(conv8)
    up9 = ConvLSTM2D(64, 2, padding='same', return_sequences=True)(
    TimeDistributed(UpSampling2D(size=(2, 2)))(conv8))
    merge9 = concatenate([conv1, up9], axis=4)
    conv9 = ConvLSTM2D(64, 3, padding='same', return_sequences=True)(merge9)
    conv9 = ConvLSTM2D(64, 3, padding='same', return_sequences=True)(conv9)
    conv9 = TimeDistributed(Conv2D(2, 3, activation='relu', padding='same'))(conv9)
    # conv9 = ConvLSTM2D(2, 3, padding='same', return_sequences=True)(conv9)
    # conv10 = ConvLSTM2D(3, 1, activation='softmax', return_sequences=True)(conv9)
    conv10 = TimeDistributed(Conv2D(1, 1,activation='softmax', padding='same'))(conv9)
    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='MRI_brain_seg_UNet3D.png', show_shapes=True)
    model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights) 
    return model
   '''
def unet(pretrained_weights = None,input_size1 = (256,256,1),input_size2=(256,256,1),input_size3=(256,256,1)):
    inputs1 = Input(input_size1)
    inputs2 = Input(input_size2)
    inputs3 = Input(input_size3)
    conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs2)
    conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
    conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1_1)
    conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
    conv3_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2_1)
    conv3_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_1)
    pool3_1 = MaxPooling2D(pool_size=(2, 2))(conv3_1)
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3_1)
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_1)
    drop4_1 = Dropout(0.5)(conv4_1)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(drop4_1)

    conv5_1 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4_1)
    conv5_1 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_1)
    drop5_1 = Dropout(0.5)(conv5_1)
    
    conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs1)
    #conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1_2)
    #conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_2)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2_2)
    #conv3_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_2)
    pool3_2 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3_2)
    #conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    drop4_2 = Dropout(0.5)(conv4_2)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(drop4_2)

    conv5_2 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4_2)
    #conv5_2 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_2)
    drop5_2 = Dropout(0.5)(conv5_2)
    
    conv1_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs3)
    #conv1_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_3)
    pool1_3 = MaxPooling2D(pool_size=(2, 2))(conv1_3)
    conv2_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1_3)
    #conv2_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_3)
    pool2_3 = MaxPooling2D(pool_size=(2, 2))(conv2_3)
    conv3_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2_3)
    #conv3_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_3)
    pool3_3 = MaxPooling2D(pool_size=(2, 2))(conv3_3)
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3_3)
    #conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
    drop4_3 = Dropout(0.5)(conv4_3)
    pool4_3 = MaxPooling2D(pool_size=(2, 2))(drop4_3)

    conv5_3 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4_3)
    #conv5_3 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_3)
    drop5_3 = Dropout(0.5)(conv5_3)
    drop5=concatenate([drop5_1,drop5_2], axis = -1)
    drop5=concatenate([drop5,drop5_3], axis = -1)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4_1,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3_1,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2_1,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1_1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = [inputs1,inputs2,inputs3], output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
model = unet()
model.fit([np.reshape(x1[0:49],[49,256,256,1]),np.reshape(x2[0:49],[49,256,256,1]),np.reshape(x3[0:49],[49,256,256,1])],np.reshape(y1[0:49],[49,256,256,1]),verbose=2,epochs=20,batch_size=2)
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(inputs,callbacks=[model_checkpoint])
a=model.predict([np.reshape(x[70],[1,256,256,1]),np.reshape(x[70],[1,256,256,1]),np.reshape(x[70],[1,256,256,1])])
#print(a)
sio.savemat('result.mat',{"foo":a})
