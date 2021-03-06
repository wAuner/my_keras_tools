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
import numpy as np
import itertools

class Mimi:
    """
    Abstract base class. Wraps keras models in a simpler API.
    """


    def __init__(self,  workdir=os.getcwd(), 
                 class_names=None, train_generator=None, val_generator=None, test_generator=None, name=None,):
        
        """
        workdir: path to working directory
        class_names: list of class names in order of the class indices
        generators: keras generators
        """

        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.workdir = workdir
        self.history = None
        if name == None:
            self.name = self.model.name
        else:
            self.name = name
            self.model.name = name
        self.model_path = workdir+'models/'+self.name+'.hdf5'
        

        if train_generator != None:
            self.class_names = [0] * len(train_gen.class_indices)
            for key,value in train_gen.class_indices.items():
                self.class_names[value] = key
        if class_names != None:
            self.class_names = class_names

    def fit(self, X_train=None, y_train=None, save_model=True, plan_learning_rate=False, 
            batch_size=32, epochs=1, verbose=1, callbacks=None, 
            validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, 
            sample_weight=None, initial_epoch=0, plot=True):
        """
        Calls the fit method of the underlying keras model to train on ararys.
        """
        callback = self.__get_callbacks(save_model, plan_learning_rate)

        
        history = self.model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, 
                       callbacks=callback, validation_split=validation_split, validation_data=validation_data, 
                       shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, 
                       initial_epoch=initial_epoch)
        
        self.__integrate_history(history)
        if plot:
            self.plot_history()

    def fit_generator(self, epochs, validate=True, 
                        train_generator=None, val_generator=None, 
                        divide_steps_per_epoch_by=1, save_model=True, plan_learning_rate=False):
        """
        Trains the model using the provided generators. If no generators are provided, it will use the generators provided
        at instanciation.
        """
        callback = self.__get_callbacks(save_model, plan_learning_rate)

        # if no train_generator was provided, use default one
        if train_generator == None:
            if self.train_generator == None:
                raise ValueError("train_generator was not defined at the beginning and none was passed")
            else:
                train_generator = self.train_generator

        # check if training with validation set
        if validate:
            if val_generator == None:
                if self.val_generator == None:
                    raise ValueError("val_generator was not defined at the beginning and none was passed")
                else:
                    val_generator = self.val_generator

            

            history = self.model.fit_generator(train_generator, 
                                    steps_per_epoch=train_generator.n / train_generator.batch_size / divide_steps_per_epoch_by,
                                    epochs=epochs, validation_data=val_generator, 
                                    validation_steps=val_generator.n / val_generator.batch_size,
                                    callbacks=callback)
            

        # if validate was set to false, train without using a validation set
        else:
            history = self.model.fit_generator(train_generator, 
                                    steps_per_epoch=train_generator.n / train_generator.batch_size / divide_steps_per_epoch_by,
                                    epochs=epochs, 
                                    callbacks=callback)
            
        self.__integrate_history(history)
    
    
    
    
    
    
    def __get_callbacks(self, save_model=True, plan_learning_rate=False):
        """
        Creates and returns the callbacks for training.
        """
        callback = []

        if save_model:  
            if not os.path.isdir(self.workdir+'/models/'):
                os.mkdir(self.workdir+'/models/')

            model_saver = ModelCheckpoint(self.model_path,save_best_only=True)
            callback.append(model_saver)

        if plan_learning_rate:
            # will divide the learning rate by 10 every other epoch
            lr_planner = LearningRateScheduler(lambda x: self.learning_rate * 0.1 ** (x//2))
            callback.append(lr_planner)

        return callback


    def __integrate_history(self, history):
        """
        Takes the new history values and concatenates them with preexisting, if any.
        """


        # history.history is a dict with the recorded metrics. each value for a key is a list of values
        if self.history == None:
            self.history = history.history
            self.history['total_epochs'] = len(history.history['loss'])
        else:
            # if history dict exists, append the new training metrics for plotting
            for key in history.history:
                self.history[key] += history.history[key]
            self.history['total_epochs'] += len(history.history['loss'])

    
    
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

    
    def summary(self):

        self.model.summary()

    def compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None):
        self.model.compile(self, optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, 
                            sample_weight_mode=sample_weight_mode)


    def evaluate_predictions_gen(self, generator=None, show_confunsion_matrix=True):
        """
        Makes predictions on generator and returns indices of correct and incorrect classified images.
        Default behavior is predicting on validation generator. Also plots confusion matrix.
        Returns a dict with filenames and predictions with structure;
        {correct: {filename: np.array(prediction),},
        wrong: {...}}
        """
        if generator == None:
            if self.val_generator == None:
                raise ValueError("val_generator was not defined at the beginning and none was passed")
            else:
                generator = self.val_generator
        predictions = self.predict_generator(generator)
        
        # indices of predictions
        correct_preds = np.where(np.argmax(predictions, axis=1) == generator.classes)[0]
        wrong_preds = np.where(np.argmax(predictions, axis=1) != generator.classes)[0]
        
        if show_confunsion_matrix:
            # getting class_names from generator
            class_names = [0] * len(train_gen.class_indices)
            for key,value in train_gen.class_indices.items():
                class_names[value] = key

            conf_matrix = confusion_matrix(generator.classes, np.argmax(predictions, axis=1))
            self.plot_confusion_matrix(conf_matrix, self.class_names, title='Confusion matrix, without normalization')

        # creates an dict of dict with filenames and predicitons of correct resp. wrong predictions
        # can later be used for plotting 
        pred_files = {'correct': \
                            {generator.directory+'/'+fname: pred for fname,pred in \
                            zip(np.array(generator.filenames)[correct_preds], predictions[correct_preds])}
                      'wrong': \
                            {generator.directory+'/'+fname: pred for fname,pred in \
                            zip(np.array(generator.filenames)[wrong_preds], predictions[wrong_preds])}
                    }
        return pred_files

    def evaluate_predictions_arr(self, X,  y_true, class_names=None, show_confunsion_matrix=True, batch_size=32):
        """
        Makes predictions on array X and returns indices of correct and incorrect classified images.
        Also plots confusion matrix.

        y_true: array with the correct labels
        class_names: list of strings with class names
        """
        # if labels are one_hot_encoded, decode them
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
            print("Labels have been decoded.")

        # if no class names were provided, try to find predefined names
        if class_names == None:
            class_names = self.class_names

        predictions = self.predict(X, batch_size=batch_size)

        correct_preds = np.where(np.argmax(predictions, axis=1) == y_true)[0]
        wrong_preds = np.where(np.argmax(predictions, axis=1) != y_true)[0]

        
        
        if show_confunsion_matrix:
            conf_matrix = confusion_matrix(y_true, np.argmax(predictions, axis=1))
            self.plot_confusion_matrix(conf_matrix, class_names, title='Confusion matrix, without normalization')

        return correct_preds, wrong_preds


    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        cm: confusion_matrix
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


    def reload(self):
        try:
            self.model = load_model(self.model_path)
        except OSError:
            print("File not found. Model was not saved under model_name in ./models")
            
    def plot_history(self):
        """
        Plots the recorded history values
        """
        x_values = range(1, self.history['total_epochs'] + 1)
        # first plot loss
        # training loss, blue
        _ = plt.plot(x_values, self.history['loss'], 'b', label='training_loss')
        
        # validation loss, red
        _ = plt.plot(x_values, self.history['val_loss'], 'r', label='validation_loss')
        plt.legend()
        plt.xticks(x_values)
        plt.tight_layout()
        plt.title('Loss over epochs')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        

        plt.figure()
        # plot accuracy
         # training loss, blue
        _ = plt.plot(x_values, self.history['acc'], 'b', label='training_acc')
        
        # validation loss, red
        _ = plt.plot(x_values, self.history['val_acc'], 'r', label='validation_acc')
        plt.legend()
        plt.xticks(x_values)
        plt.tight_layout()
        plt.title('Accuracy over epochs')
        plt.ylim([0,1])
        plt.ylabel('accuracy')
        plt.xlabel('epochs')

    
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



class PredefinedModel(Mimi):
    """
    Abstract base class for predefined models based on class Mimi.
    """


    def __init__(self, num_classes=None, model_to_load=None, workdir=os.getcwd(), class_names=None,
                 train_generator=None, val_generator=None, test_generator=None, name=None):
        if num_classes == None and model_to_load == None:
            raise ValueError("Either num classes or loadmodel is required to be not None.")
        
        
        if model_to_load == None:
            self.model = self.create_model(num_classes)

        else:
            self.model = load_model(model_to_load)
            print("Model {} successfully loaded".format(model_to_load[model_to_load.rfind('/')+1:]))

        super().__init__(workdir, train_generator, val_generator, test_generator, name)


class MimiResnet50(PredefinedModel):
    """
    Creates an instance with a pretrained Resnet50
    """

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
    
class MimiVGG16(PredefinedModel):
    """
    Creates a class with pretrained VGG16
    """

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

    def __init__(self, model=None, model_to_load=None, workdir=os.getcwd(), class_names=None,
                 train_generator=None, val_generator=None, test_generator=None, name=None):
        if model == None and model_to_load == None:
            raise ValueError("Either model or loadmodel is required to be not None.")
        
        if model_to_load == None:
            self.model = model

        else:
            self.model = load_model(model_to_load)
            print("Model {} successfully loaded".format(model_to_load[model_to_load.rfind('/')+1:]))

        self.name = self.model.name
        super().__init__(workdir, train_generator, val_generator, test_generator, name)