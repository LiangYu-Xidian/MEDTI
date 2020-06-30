# coding=UTF-8
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, concatenate
from keras import regularizers


def build_MDA(input_dims, encoding_dims):
    """
    Function for building multimodal autoencoder.
    """
    # input layers
    input_layers = []
    for dim in input_dims:
        input_layers.append(Input(shape=(dim, )))

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):#sigmoid activation='elu'
        hidden_layers.append(Dense(int(encoding_dims[0]/len(input_dims)),
                                       #use_bias=False,
                                       kernel_regularizer=regularizers.l1(1e-5),
                                       activity_regularizer=regularizers.l1(1e-5),
                                       activation='sigmoid')(input_layers[j]))

    # Concatenate layers
    if len(encoding_dims) == 1:
        hidden_layer = concatenate(hidden_layers, name='middle_layer')
        #hidden_lyer=hidden_layers[0]
    else:
        hidden_layer = concatenate(hidden_layers)

    #middle layers
    for i in range(1, len(encoding_dims)-1):
        if i == len(encoding_dims)/2:#middle layers
            hidden_layer = Dense(encoding_dims[i],
                                 name='middle_layer',
                                 #use_bias=False,
                                 kernel_regularizer=regularizers.l1(1e-5),
                                 activity_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 #use_bias=False,
                                 kernel_regularizer=regularizers.l1(1e-5),
                                 activity_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)

    if len(encoding_dims) != 1:
        # reconstruction of the concatenated layer
        hidden_layer = Dense(int(encoding_dims[0]/len(input_dims)),#encoding_dims[0],
                             #kernel_regularizer=regularizers.l1(1e-5),
                             #use_bias=False,
                             activity_regularizer=regularizers.l1(1e-5),
                             activation='sigmoid')(hidden_layer)

    # hidden layers$
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(int(encoding_dims[-1]/len(input_dims)),#int(encoding_dims[-1]/len(input_dims))
                                       #kernel_regularizer=regularizers.l2(1e-5),
                                       #use_bias=False,
                                       activity_regularizer=regularizers.l1(1e-5),
                                       activation='sigmoid')(hidden_layer))

    # output layers
    output_layers = []
    for j in range(0, len(input_dims)):
        output_layers.append(Dense(input_dims[j],
                                   #kernel_regularizer=regularizers.l2(1e-5),
                                   #use_bias=False,
                                   activity_regularizer=regularizers.l1(1e-5),
                                   activation='sigmoid')(hidden_layers[j]))#hidden_layers[j]

    # autoencoder model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=False)
    model = Model(inputs=input_layers, outputs=output_layers)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print (model.summary())

    return model


