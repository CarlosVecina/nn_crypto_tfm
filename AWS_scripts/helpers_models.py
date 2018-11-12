from keras.layers import Conv1D, Flatten, MaxPooling1D, LSTM, GRU, InputLayer
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization


def modeloDNN(args, mode="regression", num_hidden_dense_layers=2,
              hidden_dense_layers_units=20, length=50, nfeatures=7):
    if mode == "classification":
        last_activation = "softmax"
        last_units = 3
    else:
        last_activation = "linear"
        last_units = 1
    # print(f'modeloCNNBasico... conv_size: {conv_size}')
    model = Sequential()
    model.add(InputLayer(input_shape=(length, nfeatures)))

    for i in range(num_hidden_dense_layers):
        model.add(Dense(hidden_dense_layers_units, activation=args.activation))
        model.add(BatchNormalization())
    model.add(Dense(hidden_dense_layers_units, activation=args.activation))
    model.add(Flatten())
    model.add(Dense(last_units, activation=last_activation))

    return model


def modeloCNNBasico(conv_size, args, mode="regression", num_hidden_dense_layers=2,
                    hidden_dense_layers_units=20, max_pool=False, n_filter=20, length=50, nfeatures=5, droprate=0.5):
    if mode == "classification":
        last_activation = "softmax"
        last_units = 3
    else:
        last_activation = "linear"
        last_units = 1
    print(f'modeloCNNBasico... conv_size: {conv_size}')
    model = Sequential()

    model.add(InputLayer(input_shape=(length, nfeatures)))

    model.add(Conv1D(filters=n_filter,
                     kernel_size=(conv_size,),
                     padding='causal',
                     activation='relu',
                     strides=4,
                     ))
    model.add(BatchNormalization())
    if max_pool:
        model.add(MaxPooling1D(2))
    model.add(Dropout(droprate))


    for i in range(num_hidden_dense_layers):
        model.add(Dense(hidden_dense_layers_units, activation=args.activation))
        model.add(BatchNormalization())
        model.add(Dropout(droprate))
    model.add(Flatten())
    model.add(Dense(last_units, activation=last_activation))

    return model


def modeloCNNDeep(conv_size, args, mode="regression", num_hidden_dense_layers=2,
                  hidden_dense_layers_units=20, max_pool=False, n_filter=20, num_convolutional_layers=2,
                  nfeatures=5, length=50, droprate=0.5):
    if mode == "classification":
        last_activation = "softmax"
        last_units = 3
    else:
        last_activation = "linear"
        last_units = 1
    print(f'modeloCNNDeep... conv_size: {conv_size}')
    model = Sequential()

    model.add(InputLayer(input_shape=(length, nfeatures)))

    for i in range(num_convolutional_layers):
        model.add(Conv1D(filters=n_filter,
                         kernel_size=(conv_size,),
                         padding='causal',
                         activation='relu',
                         strides=4,
                         ))
        model.add(BatchNormalization())
        if max_pool:
            model.add(MaxPooling1D(2))
    model.add(Dropout(droprate))



        #if max_pool:
            #model.add(MaxPooling1D(2))
    model.add(Dropout(droprate))
    for i in range(num_hidden_dense_layers):
        model.add(Dense(hidden_dense_layers_units, activation=args.activation))
        model.add(BatchNormalization())
        model.add(Dropout(droprate))
    model.add(Flatten())
    model.add(Dense(last_units, activation=last_activation))

    return model


def modeloCNN_LSTM(conv_size, args, mode="regression", num_hidden_dense_layers=2,
                   hidden_dense_layers_units=20, max_pool=False, n_filter=20, num_convolutional_layers=2,
                   length=50, nfeatures=5, droprate=0.5):
    if mode == "classification":
        last_activation = "softmax"
        last_units = 3
    else:
        last_activation = "linear"
        last_units = 1
    print(f'modeloCNN_LSTM... conv_size: {conv_size}')
    model = Sequential()

    model.add(InputLayer(input_shape=(length, nfeatures)))

    model.add(Conv1D(filters=n_filter,
                     kernel_size=(conv_size,),
                     padding='causal',
                     activation=args.activation,
                     strides=4,
                     ))

    for i in range(num_convolutional_layers):
        model.add(Conv1D(filters=n_filter,
                         kernel_size=(conv_size,),
                         padding='causal',
                         activation=args.activation,
                         strides=4
                         ))
        model.add(BatchNormalization())
        if max_pool:
            model.add(MaxPooling1D(2))
        else:
            pass
    model.add(Dropout(droprate))

    # model.add(Flatten())

    for i in range(num_hidden_dense_layers):
        model.add(Dense(hidden_dense_layers_units, activation=args.activation))
        model.add(Dropout(droprate))

    model.add(LSTM(20))
    model.add(BatchNormalization())
    model.add(Dense(last_units, activation=last_activation))
    model.add(BatchNormalization())


    return model


def modeloCNN_GRU(conv_size, args, mode="regression", max_pool=False, n_filter=20, num_convolutional_layers=2,
                  length=50, nfeatures=5, droprate=0.5):
    if mode == "classification":
        last_activation = "softmax"
        last_units = 3
    else:
        last_activation = "linear"
        last_units = 1
    print(f'modeloCNN_GRU... conv_size: {conv_size}')
    model = Sequential()

    model.add(InputLayer(input_shape=(length, nfeatures)))

    for i in range(num_convolutional_layers):
        model.add(Conv1D(filters=n_filter,
                         kernel_size=(conv_size,),
                         padding='causal',
                         activation=args.activation,
                         strides=4
                         ))
        model.add(BatchNormalization())
        if max_pool:
            model.add(MaxPooling1D(2))
        else:
            pass
    model.add(Dropout(droprate))

    # model.add(Conv1D(n_filter, 4, activation='relu'))
    model.add(GRU(32, dropout=0.5, recurrent_dropout=0.5))
    model.add(BatchNormalization())
    model.add(Flatten())

    # for i in range(num_hidden_dense_layers):
    #   model.add(Dense(hidden_dense_layers_units, activation="sigmoid"))

    model.add(Dense(last_units, activation=last_activation))

    return model

def modeloLSTM(conv_size, args, mode="regression", max_pool=False, n_filter=20, num_convolutional_layers=2,
                  length=50, nfeatures=5, droprate=0.5):
    if mode == "classification":
        last_activation = "softmax"
        last_units = 3
    else:
        last_activation = "linear"
        last_units = 1
    print(f'modeloLSTM... conv_size: {conv_size}')
    model = Sequential()

    model.add(InputLayer(input_shape=(length, nfeatures)))


    model.add(LSTM(200))
    model.add(BatchNormalization())
    model.add(Dense(last_units, activation=last_activation))
    model.add(BatchNormalization())

    # for i in range(num_hidden_dense_layers):
    #   model.add(Dense(hidden_dense_layers_units, activation="sigmoid"))

    model.add(Dense(last_units, activation=last_activation))

    return model

