import matplotlib.pyplot as plt
import keras
import numpy as np
from skimage.io import imread_collection

def plot_specific_images_from_dir(path, idx, cls, dim=(3,5), color=True, dset='train/'):
    """
    plots images in a path defined by idx
    path: string
    idx: np.array with indices of the wanted images
    """
    # reading images
    if color:
        fig = plt.figure(figsize=(15,10))
        
        #cats==1, dogs==0
        data = ['dogs/dog.', 'cats/cat.'][cls]
        
        for i in range(dim[0]*dim[1]):
            ax = fig.add_subplot(*dim,1+i)
            img = skimage.io.imread(data_dir+dset+data+str(idx[i])+'.jpg')
            ax.imshow(img)
            ax.axis('off')

def plot_random_images_from_dir(path, dim=(3,5), color=True, dset='train/'):
    """
    plots random images in a path
    path: string
    """
    # reading images
    if color:
        fig = plt.figure(figsize=(15,10))
        
        
        for i in range(dim[0]*dim[1]):
            ax = fig.add_subplot(*dim,1+i)
            img = skimage.io.imread(data_dir+dset+random.choice(['cats/cat.', 'dogs/dog.'])+str(random.randint(0,12500))+'.jpg')
            ax.imshow(img)
            ax.axis('off')

def plot_random_images_from_array(array, dim=(3,5)):
    """
    plots random images in an array
    array: np.array of images
    """
    # reading images
    fig = plt.figure(figsize=(15,10))
    
    for i in range(dim[0]*dim[1]):
        ax = fig.add_subplot(*dim,1+i)
        ax.imshow(array[random.randint(0,len(array))])
        ax.axis('off')

def get_filenames(path, include_subdirs=True):
    """
    Returns a list of absolute paths for every file in path
    """
    return [os.path.join(path,filename) for filename in os.listdir(path)]


def plot_images_from_array(array, dim=(3,5)):
    """
    plots images of an array
    array: np.array of images
    """
    
    
    assert array.shape[0] >= dim[0]*dim[1], "batch size not big enough for image dimensions"
    
    # reading images
    fig = plt.figure(figsize=(15,10))
    
    for i in range(dim[0]*dim[1]):
        ax = fig.add_subplot(*dim,1+i)
        ax.imshow(array[i])
        ax.axis('off')


def plot_gen_images(generator, dim=(3,5)):
    """
    plots images of an array
    generator: keras ImageDataGenerator.flow_from_directory()
    """
    


    assert type(generator) == keras.preprocessing.image.DirectoryIterator,\
    "generator is not a keras generator object"
    assert generator.batch_size >= dim[0]*dim[1], \
    "batch size not big enough for image dimensions"
    
    if generator.class_mode != None:
        "if class_mode is not None the gen will yield a tuple(images, classes)"
        array = next(generator)[0]
    else:
        array = next(generator)

    # getting the filenames of the current batch
    idx = (generator.batch_index - 1) * generator.batch_size
    batch_filenames = generator.filenames[idx : idx + generator.batch_size]
    
    fig = plt.figure(figsize=(15,10))
    
    for i in range(dim[0]*dim[1]):
        ax = fig.add_subplot(*dim,1+i)

        ax.imshow(array[i])
        ax.axis('off')
        ax.title.set_text(batch_filenames[i])


def get_generators(img_mean=[], plotting=False):
    """
    creates 4 datagenerators and returns them
    img_means: list with means for each channel
    returns: train_gen, train_aug_gen, val_gen, test_gen
    """
    if img_mean==[]:
        image_mean = np.array([123.68, 116.779, 103.939])
    else:
        image_mean = np.array(img_mean)

    
    if plotting:
        # no mean-subtract for plotting
        plot_datagen = ImageDataGenerator(rescale=1/255)

        train_gen_plot =  plot_datagen.flow_from_directory(data_dir+'train_resized/', 
                    target_size=(224,224), batch_size=50, 
                    shuffle=False, class_mode=None)

        val_gen_plot = plot_datagen.flow_from_directory(data_dir+'validation_resized/', 
                                                    target_size=(224,224), batch_size=50, 
                                                    shuffle=False, class_mode=None)

        test_gen_plot = plot_datagen.flow_from_directory(data_dir+'test/', 
                                                    target_size=(224,224), batch_size=50, 
                                                    shuffle=False, class_mode=None)
        return val_gen_plot, test_gen_plot

    else:
        # training without data_augmentation
        train_datagen = ImageDataGenerator(rescale=1/255, 
                                    preprocessing_function=lambda x: x - image_mean)

        # training with data_augmentation
        train_aug_datagen = ImageDataGenerator(rescale=1/255, 
                                    preprocessing_function=lambda x: x - image_mean,
                                            rotation_range=30, 
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                channel_shift_range=0.05,
                                                fill_mode='nearest',
                                                cval=0.,
                                                horizontal_flip=True,
                                                vertical_flip=False,
                                                )

        # inference
        inference_datagen = ImageDataGenerator(rescale=1/255, 
                                    preprocessing_function=lambda x: x - image_mean)

        

        train_gen = train_datagen.flow_from_directory(data_dir+'train_resized/', target_size=(224,224), 
                                                        batch_size=32, shuffle=True)

        train_aug_gen = train_datagen.flow_from_directory(data_dir+'train_resized/', target_size=(224,224), 
                                                        batch_size=32, shuffle=True)

        val_gen = inference_datagen.flow_from_directory(data_dir+'validation_resized/', 
                                                        target_size=(224,224), batch_size=50, 
                                                        shuffle=False, class_mode=None)

        test_gen = inference_datagen.flow_from_directory(data_dir+'test/', 
                                                        target_size=(224,224), batch_size=50, 
                                                        shuffle=False, class_mode=None)


        return train_gen, train_aug_gen, val_gen, test_gen



def make_prediction_and_submission(model, generator, filename='submission.txt', clip_at=0.99):
    """
    model: Keras model
    generator: DirectoryIterator

    returns: predictions as np.array
    """
    assert type(generator) == keras.preprocessing.image.DirectoryIterator,\
    "generator is not a keras generator object"

    predictions = model.predict_generator(generator, steps=generator.n/generator.batch_size, verbose=1)
    ids = np.array([int(i[5:-4]) for i in generator.filenames])
    submission = np.stack([ids, np.clip(predictions[:,1], 1 - clip_at, clip_at)]).T

    np.savetxt(filename, submission, fmt='%d,%.7f', header='id,label', comments='')

    return predictions

def make_submission(predictions, generator, filename='submission.txt', clip_at=0.99):
    """
    predictions: np array with predictions
    generator: DirectoryIterator

    returns: Nothing
    """
    #assert type(generator) == keras.preprocessing.image.DirectoryIterator,\
    #"generator is not a keras generator object"

    
    ids = np.array([int(i[5:-4]) for i in generator.filenames])
    submission = np.stack([ids, np.clip(predictions[:,1], 1 - clip_at, clip_at)]).T

    np.savetxt(filename, submission, fmt='%d,%.7f', header='id,label', comments='')


class MixIterator(object):
    """
    Class to combine a list of keras generators in order to create mixed batches,
    e.g. for pseudo labeling
    J Howard implementation
    """
    def __init__(self, iters):
        """
        iters: list of generators
        """
        self.iters = iters
        self.multi = type(iters) is list
        if self.multi:
            self.N = sum([it[0].N for it in self.iters])
        else:
            self.N = sum([it.N for it in self.iters])

    def reset(self):
        for it in self.iters: 
            it.reset()

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        if self.multi:
            nexts = [[next(it) for it in o] for o in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)
        else:
            nexts = [next(it) for it in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)


class MixDirectoryIterator(object):
    """
    Class to combine a list of keras generators in order to create mixed batches,
    e.g. for pseudo labeling from directories
    """
    def __init__(self, iters):
        """
        iters: list of generators
        """
        
        
        assert len(iters) > 1, "provide at least 2 iterators"

        # for it in iters:
        #     assert type(it) == keras.preprocessing.image.DirectoryIterator or \
        #     type(it) == keras.preprocessing.mimi_custom_image.DirectoryIterator,\
        #     "generator is not a keras generator object"

        self.iters = iters
        self.n = sum([it.n for it in self.iters])
        self.batch_size = sum([it.batch_size for it in self.iters])
        #assumed color images
        self.image_size = self.iters[0].target_size + (3,)
        self.class_indices = self.iters[0].class_indices

      
    def reset(self):
        for it in self.iters: 
            it.reset()

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        
        
        nexts = [next(it) for it in self.iters]
        next_image_batch = np.vstack([img_batch[0] for img_batch in nexts])
        next_label_batch = np.vstack([img_batch[1] for img_batch in nexts])

        return (next_image_batch, next_label_batch)
        