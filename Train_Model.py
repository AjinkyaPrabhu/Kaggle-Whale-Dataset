from Keras_Model import create_model
import numpy as np
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,LearningRateScheduler, TensorBoard, EarlyStopping





##model = create_model(len(data1[1][1]))
##compile and fit model on dataset

##model.compile()

WIDTH,HEIGHT = 256,256
train_data_dir = './data/train'
batch_size = 8
epochs = 1
N_CLASSES = 4251
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 8
epochs = 2

model = create_model(N_CLASSES)

model.compile(loss = "categorical_crossentropy",
              optimizer =  optimizers.SGD(lr=0.001,momentum = 0.9),
              metrics =["accuracy"] )



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   horizontal_flip = True,
                                   fill_mode = "nearest",
                                   zoom_range = 0.3,
                                   width_shift_range = 0.3,
                                   height_shift_range=0.3,
                                   rotation_range=30)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                   target_size = (WIDTH, HEIGHT),
                                                   batch_size = batch_size,
                                                   class_mode = "categorical")
checkpoint = ModelCheckpoint("whale_checkpoint.h5",
                             monitor='val_acc',
                             verbose=1, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', period=1)

early = EarlyStopping(monitor='val_acc',
                      min_delta=0,
                      patience=10,
                      verbose=1,
                      mode='auto')

board = TensorBoard(log_dir="logs/logs",
                    write_graph=True,
                    batch_size=batch_size)



model.fit_generator(train_generator,
                    samples_per_epoch = nb_train_samples,
                    epochs = epochs,
                    nb_val_samples = nb_validation_samples,
                    callbacks = [checkpoint, early, board])


model.save("Whale_model_1.h5")
