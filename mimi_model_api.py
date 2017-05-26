from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

class Mimi:
    """
    Abstract base class. Wraps keras models in a simpler API.
    """


    def __init__(self,  workdir=os.getcwd(), train_generator=None, val_generator=None, test_generator=None):
        
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.workdir = workdir
        
    def fit_generator(self, epochs, validate=True, 
                        train_generator=None, val_generator=None, 
                        divide_steps_per_epoch_by=1, save_model=True):
        """
        Trains the model using the provided generators. If no generators are provided, it will use the generators provided
        at instanciation.
        """
        if validate:
            if train_generator == None:
                if self.train_generator == None:
                    raise ValueError("train_generator was not defined at the beginning and none was passed")
                else:
                    train_generator = self.train_generator

            if val_generator == None:
                if self.val_generator == None:
                    raise ValueError("val_generator was not defined at the beginning and none was passed")
                else:
                    val_generator = self.val_generator

            if save_model:
                if not os.path.isdir(self.workdir+'/models/'):
                    os.mkdir(self.workdir+'/models/')

                model_saver = ModelCheckpoint(self.workdir+'/models/'+self.name+'.hdf5',save_best_only=True)
                self.history = self.model.fit_generator(train_generator, 
                                        steps_per_epoch=train_generator.n / train_generator.batch_size / divide_steps_per_epoch_by,
                                        epochs=epochs, validation_data=val_generator, 
                                        validation_steps=val_generator.n / val_generator.batch_size,
                                        callbacks=[model_saver])
            else:
                self.history = self.model.fit_generator(train_generator, 
                                        steps_per_epoch=train_generator.n / train_generator.batch_size / divide_steps_per_epoch_by,
                                        epochs=epochs, validation_data=val_generator, 
                                        validation_steps=val_generator.n / val_generator.batch_size,
                                        callbacks=[])

        # if validate was set to false, train without using a validation set
        else:
            if save_model:
                if not os.path.isdir(self.workdir+'/models/'):
                    os.mkdir(self.workdir+'/models/')

                model_saver = ModelCheckpoint(self.workdir+'/models/'+self.name+'.hdf5',save_best_only=True)
                self.history = self.model.fit_generator(train_generator, 
                                        steps_per_epoch=train_generator.n / train_generator.batch_size / divide_steps_per_epoch_by,
                                        epochs=epochs, 
                                        callbacks=[model_saver])
            else:
                self.history = self.model.fit_generator(train_generator, 
                                        steps_per_epoch=train_generator.n / train_generator.batch_size / divide_steps_per_epoch_by,
                                        epochs=epochs,
                                        callbacks=[])

    def predict_generator(self, generator=None):
        """
        Make predictions on generator. if no generator is provided, use default test_generator.
        Returns predictions for generator.
        """
        if generator == None:
            if self.test_generator == None:
                raise ValueError("test_generator was not defined at the beginning and none was passed")
            else:
                test_generator = self.test_generator
        # make sure predictions start at the beginning of test_generator.filenames        
        generator.reset()

        return self.model.predict_generator(generator, steps=generator.n/generator.batch_size,verbose=1)

        
    def fit(self, X_train=None, y_train=None, batch_size=32, epochs=1, verbose=1, callbacks=None, validation_split=0.0, 
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0):
        """
        Calls the fit method of the underlying keras model to train on ararys.
        """
        self.model.fit(X_train=X_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, 
                       callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, 
                       shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, 
                       initial_epoch=initial_epoch)

    def predict(self, x, batch_size=32, verbose=0):
        """
        Calls the predict method of the underlying keras model to predict on ararys.
        """
        return self.model.predict(x=x, batch_size=batch_size, verbose=verbose)
    
    
    def save_model(self, path=None):

        if path == None:
            path = self.workdir+'/models/'
        self.model.save(path+self.name+'.hdf5')
        print('Model has been saved at ',path+self.name+'.hdf5')


    def evaluate_predictions(self, generator=None, show_confunsion_matrix=True):
        """
        Makes predictions on generator and returns indices of correct and incorrect classified images.
        Default behavior is predicting on validation generator
        """
        if generator == None:
            if self.val_generator == None:
                raise ValueError("val_generator was not defined at the beginning and none was passed")
            else:
                generator = self.val_generator
        predictions = self.predict_generator(generator)

        correct_preds = np.where(np.argmax(predictions, axis=1)==generator.classes)[0]
        wrong_preds = np.where(np.argmax(predictions, axis=1)!=generator.classes)[0]
        
        
        class_names = [name for name in generator.class_indices.keys()]
        conf_matrix = confusion_matrix(generator.classes, np.argmax(predictions, axis=1))
        self.plot_confusion_matrix(conf_matrix, class_names, title='Confusion matrix, without normalization')

        return correct_preds, wrong_preds


    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
            

    
    @property
    def learning_rate(self):
        """
        Returns the current learning rate of the model as an attribute
        """
        return K.get_value(self.model.optimizer.lr)

    @learning_rate.setter
    def learning_rate(self, learning_rate=0.001):
        """
        Allows to change the learning rate easily as an attribute
        """
        K.set_value(self.model.optimizer.lr, learning_rate)
        print("New learning rate is ", self.learning_rate)


class MimiResnet50(Mimi):
    """
    Creates a class with a pretrained Resnet50
    """


    def __init__(self, num_classes=None, model_to_load=None, workdir=os.getcwd(), 
                 train_generator=None, val_generator=None, test_generator=None):
        if num_classes == None and model_to_load == None:
            raise ValueError("Either num classes or loadmodel is required to be not None.")
        
        
        if model_to_load == None:
            self.model = self.create_model(num_classes)

        else:
            self.model = load_model(model_to_load)
            print("Model {} successfully loaded".format(model_to_load[model_to_load.rfind('/')+1:]))

        self.name = self.model.name
        super().__init__(workdir, train_generator, val_generator, test_generator)


    def create_model(self, num_classes):
        """
        Creates a new Resnet with custom head and returns it
        """

        resnet = ResNet50()
        inputs = resnet.layers[-2].output
        predictions = Dense(num_classes, activation='softmax')(inputs)
        resnet_model = Model(inputs=resnet.input, outputs=predictions)

        resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model compiled")

        return resnet_model
    
class MimiVGG16(Mimi):
    """
    Creates a class with pretrained VGG16
    """

    def __init__(self, num_classes=None, model_to_load=None, workdir=os.getcwd(), 
                 train_generator=None, val_generator=None, test_generator=None):
        if num_classes == None and model_to_load == None:
            raise ValueError("Either num classes or loadmodel is required to be not None.")
        
        if model_to_load == None:
            self.model = self.create_model(num_classes)

        else:
            self.model = load_model(model_to_load)
            print("Model {} successfully loaded".format(model_to_load[model_to_load.rfind('/')+1:]))
        self.name = self.model.name
        super().__init__(workdir, train_generator, val_generator, test_generator)


    def create_model(self, num_classes):
        """
        Creates a new Resnet with custom head and returns it
        """

        vgg = VGG16()
        inputs = vgg.layers[-2].output
        predictions = Dense(num_classes, activation='softmax')(inputs)
        vgg_model = Model(inputs=vgg.input, outputs=predictions)

        vgg_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model compiled")

        return vgg_model



class MimiCustomModel(Mimi):
    """
    Creates a class with pretrained VGG16
    """

    def __init__(self, model=None, model_to_load=None, workdir=os.getcwd(), 
                 train_generator=None, val_generator=None, test_generator=None):
        if model == None and model_to_load == None:
            raise ValueError("Either model or loadmodel is required to be not None.")
        
        if model_to_load == None:
            self.model = model

        else:
            self.model = load_model(model_to_load)
            print("Model {} successfully loaded".format(model_to_load[model_to_load.rfind('/')+1:]))

        self.name = self.model.name
        super().__init__(workdir, train_generator, val_generator, test_generator)