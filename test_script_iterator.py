#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from my_utils import MixDirectoryIterator
from keras.preprocessing.mimi_custom_image import ImageDataGenerator

data_dir = '/home/mimisvm/Documents/Datasets/cats_dogs/'
image_mean = np.array([123.68, 116.779, 103.939])

inference_datagen = ImageDataGenerator(rescale=1/255, 
                            preprocessing_function=lambda x: x - image_mean)

val_gen = inference_datagen.flow_from_directory(data_dir+'/datamix/'+str(0)+'/validation/', 
                                                target_size=(224,224), batch_size=2, 
                                                shuffle=False)

train_gen = inference_datagen.flow_from_directory(data_dir+'/datamix/'+str(0)+'/validation/', 
                                                target_size=(224,224), batch_size=2, 
                                                shuffle=False)


crazy = inference_datagen.flow_from_directory(data_dir+'/datamix/'+str(0)+'/validation/', 
                                                target_size=(224,224), batch_size=2, 
                                            shuffle=False, class_mode='mimistyle',custom_labels=np.array([[5,6],[7,8],[5,6],[7,8],[5,6],
                                                                                                        [7,8],[5,6],[7,8],[5,6],[7,8]]))

#a = MixDirectoryIterator([val_gen, train_gen])
#print(next(MixDirectoryIterator([val_gen, train_gen]))[0].shape)
#print(next(MixDirectoryIterator([val_gen, train_gen]))[1].shape)

