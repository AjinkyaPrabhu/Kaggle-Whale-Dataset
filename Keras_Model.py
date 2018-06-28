from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten

def create_model(n_out_classes):
    model = Sequential()
    ##Conv2d Layer    
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(256,256,3)))
    ##3x3 pooling
    model.add(MaxPooling2D(pool_size=(3,3)))

    model.add(Conv2D(64,(3,3),activation='relu'))

    model.add(Dropout(0.25))
    
    model.add(Flatten())

    model.add(Dense(n_out_classes,activation='softmax'))

    return model


    

    
