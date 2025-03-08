import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,GlobalAvgPool2D
from keras.applications.resnet import ResNet50
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import string
import random
from captcha.image import ImageCaptcha

# 字符包含所有数字和所有大小写英文字母，一共 62 个
characters = string.digits + string.ascii_letters
characters_set=string.digits + string.ascii_letters+string.digits+string.digits
#产生一个字符的image作为一个样本
def create_dataset(sample_num=10000, width=50, height=50,channel=3):
    x = np.zeros((sample_num+sample_num//10, height, width,channel), dtype=np.float32)
    y = np.zeros((sample_num+sample_num//10, len(characters)+1), dtype=np.float32)
    for i in range(sample_num):
        # 随机生成一个字符
        captcha_text = random.choice(characters_set)
        x[i]=np.array(ImageCaptcha(width,height).generate_image(captcha_text))/255.0
        # 找到字符对应的编号
        index = characters.index(captcha_text)
        # 将编号转换为 one-hot 编码
        y[i, index] = 1

    #负样本
    for i in range(sample_num//10):
        captcha_text = ' '
        x[i+sample_num]=np.array(ImageCaptcha(width,height).generate_image(captcha_text))/255.0
        y[i+sample_num, -1] = 1
    return x, y



def create_model(input_shape):
    model = Sequential()
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model.add(resnet50)
    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Dense(len(characters)+1, activation='softmax'))
    return model

if __name__ == '__main__':
    x_train, y_train = create_dataset()
    x_test, y_test = create_dataset(1000)
    height=x_test[0].shape[0]
    width=x_test[0].shape[1]
    # 创建数据集
    
    # 创建模型
    input_shape = (height, width,3)  # 根据你的数据集调整这些值
    model = create_model(input_shape)

    # 编译模型
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=6, verbose=1),
                 CSVLogger('Captcha.csv'),
                 ModelCheckpoint('Best_Captcha.h5', monitor='val_loss', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)]
    # 训练模型
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test),batch_size=64,callbacks=callbacks)
