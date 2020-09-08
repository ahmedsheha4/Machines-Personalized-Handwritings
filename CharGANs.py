import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from tensorflow.keras.layers import Dense,Reshape,Flatten,Dropout,LeakyReLU,BatchNormalization,Conv2D,Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
import cv2
from PIL import Image
import os
import  random


def Model(letter,num_of_epochs,BatchSize):
    data = []
    images = os.listdir("Dataset/Chars/"+letter+"/")
    print(images)
    for i in images:
        print(i)
        image=cv2.imread("Dataset/Chars/"+letter+"/"+i)

        image_from_array = Image.fromarray(image)
        image_from_array = image_from_array.convert('L')
        size_image = image_from_array.resize((28,28))
        data.append(np.array(size_image))
    print(len(data))
    print('***********************')
    plt.imshow(data[0])
    plt.show()
    plt.imshow(data[1])
    plt.show()
    mydata = np.asarray(data)
    mydata = mydata / 255
    mydata = mydata.reshape(-1,28,28,1) * 2. -1.
    print(mydata.shape)
    discriminator = Sequential()

    discriminator.add(Conv2D(64 , kernel_size = (5,5) , strides =(2,2) , padding = "same" ,activation =LeakyReLU(0.3),
                             input_shape =[28,28,1]))

    discriminator.add(Dropout(0.5))

    discriminator.add(Conv2D(128 , kernel_size = (5,5) , strides =(2,2) , padding = "same" ,activation =LeakyReLU(0.3)
                            ))

    discriminator.add(Dropout(0.5))

    discriminator.add(Flatten())

    discriminator.add(Dense(1,activation='sigmoid'))


    input_to_generator = 100
    generator = Sequential()

    generator.add(Dense(7 * 7 * 128,  input_shape=[input_to_generator]))

    generator.add(Reshape([7,7,128]))

    generator.add(BatchNormalization())

    generator.add(Conv2DTranspose(64,kernel_size=(5,5) , strides=(2,2) , padding ="same" , activation = "relu"))

    generator.add(BatchNormalization())

    generator.add(Conv2DTranspose(1,kernel_size=(5,5) , strides=(2,2) , padding ="same" , activation = "tanh"))

    GAN = Sequential([generator,discriminator])
    discriminator.compile(loss='binary_crossentropy' , optimizer='adam')
    discriminator.trainable = False
    GAN.compile(loss='binary_crossentropy',optimizer='adam')

    batch_size = BatchSize
    my_data = mydata
    dataset=tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size = 1000)

    dataset = dataset.batch(batch_size, drop_remainder = True).prefetch(1)
    epochs = num_of_epochs
    generator , discriminator = GAN.layers

    for epoch in range(epochs):
        print(f"Epoch #{epoch+1}")
        i = 0
        for x_batch in dataset:
            i = i +1
            if i%50 == 0 :
                print(f'\t Batch #{i}')
            noisy_image = tf.random.normal(shape=[batch_size,input_to_generator])

            gen_images = generator(noisy_image)
            input_to_discriminator = tf.concat([gen_images,tf.dtypes.cast(x_batch,tf.float32)],axis=0)
            Labels=tf.constant([[0.0]]*batch_size + [[1.0]]*batch_size)
            discriminator.trainable = True

            discriminator.train_on_batch(input_to_discriminator,Labels)
            noisy_image = tf.random.normal(shape=[batch_size,input_to_generator])
            FakeLabels=tf.constant([[1.0]]*batch_size)
            discriminator.trainable = False

            GAN.train_on_batch(noisy_image,FakeLabels)
    noise_images = tf.random.normal(shape=[10,input_to_generator])
    images = generator(noise_images)
    for i in images :
        plt.imshow(i.numpy().reshape(28,28))
        plt.show()

    model_json = generator.to_json()
    with open("Trained Model/model"+letters[letter]+".json", "w") as json_file:
        json_file.write(model_json)

    generator.save_weights("Trained Model/model"+letters[letter]+".h5")
    print("Saved model to disk")

Model('A',200,32)

