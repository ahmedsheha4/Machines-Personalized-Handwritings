import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import cv2
from PIL import Image
import os
import math

letters = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":
           15,"Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,"Y":24,"Z":25," ":26 , ",":27 , ".":28}
def Generate_Handwritten_Words(WordsPerSentence=6):

    print('Enter a Word/Sentence:')
    userinput = input()
    userinput = userinput.upper()
    spaces = 0
    spacesindices = []
    for  index , i in enumerate(list(userinput)) :
        if i == " ":
            spaces += 1
            spacesindices.append(index)
        if i not in letters:
            print('Input should contain only characters from A to Z')
            print('\t ************************************')
            return

    myword = list(userinput)
    print(myword)


    numoflines = math.ceil(spaces/WordsPerSentence)
    print(numoflines)
    if spaces > WordsPerSentence :
        while spaces < WordsPerSentence * numoflines + 1:
            spacesindices.append(spacesindices[-1]+WordsPerSentence)
            spaces +=1

    print(spacesindices)
    numoflines = math.ceil(spaces/WordsPerSentence)
    print('Number of lines: ', numoflines)
    spacesindices2 = []

    for k , n in enumerate(spacesindices) :
        if k % WordsPerSentence == 0 and k != 0:
            print('true')
            spacesindices2.append(n)

    images = []
    row=[]
    m = -1
    if numoflines > 1:
        for j in range(0 , numoflines-1):
            m +=1
            images = []
            for k , i in enumerate(myword):
                print(letters[i])

                if k < spacesindices2[m] and (k > spacesindices2[m - 1] or m - 1 == -1):

                    if letters[i] < 26 :

                        json_file = open('Trained Model/model' + str(letters[i]) + '.json', 'r')
                        loaded_model_json = json_file.read()
                        json_file.close()
                        loaded_model = model_from_json(loaded_model_json)

                        loaded_model.load_weights("Trained Model/model" + str(letters[i]) + ".h5")
                        print("Loaded model from disk")

                        noise = tf.random.normal(shape=[1, 100])
                        image = loaded_model(noise)
                        image = image.numpy().reshape(28, 28)
                        image = cv2.resize(image, (56, 56))
                        images.append(image)


                    else:

                        x = np.empty([56, random.randint(50,60)], dtype=np.float32)
                        x.fill(-1)
                        images.append(x)

            print(images)
            x = np.concatenate([i for i in images], 1)
            row.append(x)

        max_width = 0
        for i in row :
            if i.shape[1] > max_width:
                max_width = i.shape[1]
        print(max_width)
        words=[]
        for i in row:
            print('**Prev Shape**',i.shape)
            if i.shape[1] < 1400 :
                x = np.empty([56, max_width-i.shape[1]], dtype=np.float32)
                x.fill(-1)
                words.append(np.concatenate([i,x],1))
            else :
                x = cv2.resize(i, dsize=(max_width, 56),interpolation=cv2.INTER_CUBIC)
                words.append(x)
        x = np.concatenate([i for i in words], 0)
        x = (255 - x) #to invert image colors
        x=cv2.medianBlur(x, 3) #Median filter to remove the noise
        x=cv2.GaussianBlur(x,(3,3),1)#Gaussian filter to smooth the image
        plt.imshow(x,cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        for i in myword:
            if letters[i] < 26 :
                json_file = open('Trained Model/model' + str(letters[i]) + '.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)

                loaded_model.load_weights("Trained Model/model" + str(letters[i]) + ".h5")
                print("Loaded model from disk")

                noise = tf.random.normal(shape=[1, 100])
                image = loaded_model(noise)
                image = image.numpy().reshape(28, 28)
                image = cv2.resize(image, (56, 56))
                images.append(image)

            else:

                x = np.empty([56, random.randint(45,55)], dtype=np.float32)
                x.fill(-1)
                images.append(x)

            print(images)
        x = np.concatenate([i for i in images], 1)
        x = (255 - x) #to invert image colors
        x=cv2.medianBlur(x, 3) #Median filter to remove the noise
        x=cv2.GaussianBlur(x,(3,3),1) #Gaussian filter to smooth the image
        plt.imshow(x,cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()

Generate_Handwritten_Words(6)

