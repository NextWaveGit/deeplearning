import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import math
import pandas as pd
from keras.models import model_from_json
import sys, getopt

# dimensions of our images.
img_width, img_height = 150, 150

top_model_json_path = 'top_model.json'
top_model_weights_path = 'top_model_weights.h5'

epochs = 10
batch_size = 128

train_dir = './data/train'
test_dir = './data/test'
score_dir = './data/score'

score_path = 'scores'

class Ipa:
    def train(self, filepath):
        datagen = ImageDataGenerator(rescale=1. / 255)

        generator = datagen.flow_from_directory(
            filepath,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        bottleneck_features_train = model.predict_generator(generator, len(generator))

        train_data = bottleneck_features_train

        train_labels = generator.classes

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', #optimizer='rmsprop',
                      loss='binary_crossentropy',  metrics=['accuracy'])

        model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
        model.save_weights(top_model_weights_path)

        model_json = model.to_json()
        with open(top_model_json_path, "w") as json_file:
            json_file.write(model_json)

        return model


    def predict(self, filepath):
        datagen = ImageDataGenerator(rescale=1. / 255)

        generator = datagen.flow_from_directory(
            filepath,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        vgg_features = model.predict_generator(generator, len(generator))

        json_file = open(top_model_json_path, 'r')
        model_json = json_file.read()
        json_file.close();
        model = model_from_json(model_json)
        model.load_weights(top_model_weights_path)

        yhat = model.predict(vgg_features)

        filenames=generator.filenames
        labels = generator.classes
        results=pd.DataFrame({"Filename":filenames,"Score":yhat[:,0], "Labels":labels})
        #results.to_csv(score_path,index=False)

        return results

def main(argv):
    
    try:
        opts, args = getopt.getopt(argv, "", ["train", "test", "score"])
    except getopt.GetoptError as err:
        print(err) # will print something like "option -a not recognized"
        sys.exit(2)
        
    ipa = Ipa()
    
    for opt, arg in opts:
        if opt == "--train":
            ipa.train(train_dir)
        elif opt == '--test':
            results = ipa.predict(test_dir)
            results.to_csv('test_scores',index=False)
        elif opt == '--score':
            results = ipa.predict(score_dir)
            results = results.drop(['Labels'], axis=1)
            results.to_csv('scores',index=False)
            
            
if __name__ == "__main__":
   main(sys.argv[1:])