from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, ConvLSTM2D, Conv3D, MaxPooling2D, Dropout, \
    MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.losses import categorical_crossentropy
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils import plot_model
import json


class TransferLearningNN():
  
    def __init__(self, model_name):
        self.model_name = model_name
        self._init_model()

    def fit(self, x_train, y_train):
        raise NotImplementedError("Class %s doesn't implement fit() only fit_generator()" % self.__class__.__name__)

    def _init_model(self):

        base_model = self._get_base_model()

        top_layer_model = base_model.output
        top_layer_model = GlobalAveragePooling2D()(top_layer_model)
        top_layer_model = Dense(1024, activation='relu')(top_layer_model)
        prediction_layer = Dense(output_dim = 7, activation='softmax')(top_layer_model)
       
        model = Model(input=base_model.input, output=prediction_layer)

        print(model.summary())
        
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            ## Optimizer ?? = adams() 
        self.model = model

    
    def _get_base_model(self):
        
        if self.model_name == 'inception_v3':
            return InceptionV3(weights='imagenet', include_top=False)
        elif self.model_name == 'xception':
            return Xception(weights='imagenet', include_top=False)
        elif self.model_name == 'vgg16':
            return VGG16(weights='imagenet', include_top=False)
        elif self.model_name == 'vgg19':
            return VGG19(weights='imagenet', include_top=False)
        elif self.model_name == 'resnet50':
            return ResNet50(weights='imagenet', include_top=False)
        else:
            raise ValueError('Cannot find base model %s' % self.model_name)

'''if __name__ == '__main__':
    model = TransferLearningNN('inception_v3').model'''