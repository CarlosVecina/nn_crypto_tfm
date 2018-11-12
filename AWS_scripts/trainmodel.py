# encoding: utf8
import json
import os
import argparse
from pprint import pprint

import keras
from os.path import join

from keras.optimizers import *

from helpers_data import *
from helpers_parquet import *
from helpers_models import *

import errno
import os


# https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def train_model(args):

    pprint(args.__dict__)
    mkdir_p(join("models", args.modelname, "checkpoints"))

    with open(join("models", args.modelname, "parameters.json"), "w") as f:
        diccionario = args.__dict__.copy()
        json.dump(diccionario, f, indent=4)

    training, test = folder_parquet(f"data/{args.dataset}/gdax", dias_validacion=args.validation_days)
    training = procesar_datos(training, args)
    test = procesar_datos(test, args)

    if args.model == "modeloDNN":
        model = modeloDNN(args=args,
                          mode=args.mode,
                          num_hidden_dense_layers=args.num_hidden_dense_layers,
                          hidden_dense_layers_units=args.hidden_dense_layers_units)
    elif args.model == "modeloCNNBasico":
        model = modeloCNNBasico(args=args,
                                conv_size=args.conv_size,
                                mode=args.mode,
                                num_hidden_dense_layers=args.num_hidden_dense_layers,
                                hidden_dense_layers_units=args.hidden_dense_layers_units,
                                max_pool=args.max_pool,
                                nfeatures=args.nfeatures,
                                length=args.length,
                                droprate=args.droprate)
    elif args.model == "modeloCNNDeep":
        model = modeloCNNDeep(args=args,
                              conv_size=args.conv_size,
                              mode=args.mode,
                              num_hidden_dense_layers=args.num_hidden_dense_layers,
                              hidden_dense_layers_units=args.hidden_dense_layers_units,
                              max_pool=args.max_pool,
                              num_convolutional_layers=args.num_convolutional_layers,
                              nfeatures=args.nfeatures,
                              length=args.length,
                              droprate=args.droprate)
    elif args.model == "modeloCNN_LSTM":
        model = modeloCNNDeep(args=args,
                              conv_size=args.conv_size,
                              mode=args.mode,
                              num_hidden_dense_layers=args.num_hidden_dense_layers,
                              hidden_dense_layers_units=args.hidden_dense_layers_units,
                              max_pool=args.max_pool,
                              num_convolutional_layers=args.num_convolutional_layers,
                              nfeatures=args.nfeatures,
                              length=args.length,
                              droprate=args.droprate)
    elif args.model == "modeloCNN_GRU":
        model = modeloCNNDeep(args=args,
                              conv_size=args.conv_size, mode=args.mode,
                              num_hidden_dense_layers=args.num_hidden_dense_layers,
                              hidden_dense_layers_units=args.hidden_dense_layers_units,
                              max_pool=args.max_pool,
                              num_convolutional_layers=args.num_convolutional_layers,
                              nfeatures=args.nfeatures,
                              length=args.length,
                              droprate=args.droprate)
    elif args.model == "modeloLSTM":
        model = modeloCNNDeep(args=args,
                              conv_size=args.conv_size, mode=args.mode,
                              num_hidden_dense_layers=args.num_hidden_dense_layers,
                              hidden_dense_layers_units=args.hidden_dense_layers_units,
                              max_pool=args.max_pool,
                              num_convolutional_layers=args.num_convolutional_layers,
                              nfeatures=args.nfeatures,
                              length=args.length,
                              droprate=args.droprate)
    else:
        print("No has seleccionado ningún modelo")
        sys.exit(-1)

    # Compile
    loss = 'categorical_crossentropy' if args.mode == "classification" else 'mean_squared_error'
    metrics = [keras.metrics.categorical_accuracy] if args.mode == "classification" else [
        keras.metrics.mae]
    metric_name = "categorical_accuracy" if args.mode == "classification" else "mean_absolute_error"

    model.compile(optimizer=eval(args.optimizer), loss=loss, metrics=metrics)

    # Guardar algunas referencias útiles
    with open(join("models", args.modelname, "structure.json"), "w") as f:
        f.write(model.to_json())

    with open(join("models", args.modelname, "structure.txt"), "w", encoding="utf8") as f:
        keras.utils.print_summary(model, line_length=120, print_fn=lambda x: f.write(f"{x}\n"))

    keras.utils.plot_model(model, to_file=join("models", args.modelname, "structure.png"), show_shapes=True,
                           show_layer_names=True, rankdir='TB')
    keras.utils.plot_model(model, to_file=join("models", args.modelname, "structure-horizontal.png"), show_shapes=True,
                           show_layer_names=True, rankdir='LR')

    callbacks = [
        keras.callbacks.CSVLogger(join("models", args.modelname, "history.csv"), separator=',', append=False),
        keras.callbacks.ModelCheckpoint(
            join('models', args.modelname, 'checkpoints/checkpoint.{epoch:02d}-{' + metric_name + ':.4f}.hdf5'),
            monitor=metric_name, verbose=0, save_best_only=True,
            save_weights_only=False, period=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto'),
        keras.callbacks.TensorBoard(log_dir=join('models', args.modelname, 'tensorboard'), histogram_freq=0,
                                    batch_size=args.batch_size, write_graph=True,
                                    write_grads=True, write_images=True),
        keras.callbacks.ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.7, patience=5, verbose=1,
                                          mode='auto',
                                          min_delta=0.01, cooldown=0, min_lr=0)

    ]

    # Train
    model.fit_generator(training,
                        validation_data=test,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        validation_steps=args.validation_steps,
                        verbose=1,
                        workers=1,
                        use_multiprocessing=False,
                        callbacks=callbacks
                        )

    print(f"Guardando modelo {args.modelname}")
    predicciones = model.predict_generator(test, steps=10)
    np.savetxt("predicciones.csv", predicciones, delimiter=",")

def load_parameters_from_dict(parameters_from_json, args):
    if args is not None:
        parameters_from_console = args.__dict__.copy()
        if "json" in parameters_from_console:
            parameters_from_console.pop("json")  # Eliminar el json que no queremos que aparezca

        # Quitamos todos los None
        parameters_from_console = {k: v for k, v in parameters_from_console.items() if v is not None}

        parameters_from_json.update(parameters_from_console)  # Unimos el json y los parametros de consola

    if "json" in parameters_from_json and parameters_from_json["json"] is not None:
        # Unir con el otro json
        with open(parameters_from_json["json"], "r") as f:
            padre = load_parameters_from_dict(json.load(f), args=None)

        # Si no es un diccionario, convierto de namespace a diccionario
        if isinstance(parameters_from_json, argparse.Namespace):
            parameters_from_json = parameters_from_json.__dict__

        if isinstance(padre, argparse.Namespace):
            padre = padre.__dict__

        nuevo = padre.copy()
        nuevo.update(parameters_from_json)
        parameters_from_json = nuevo

    return argparse.Namespace(**parameters_from_json)


def devolver_conteo(args):
    training, test = folder_parquet(f"data/{args.dataset}/gdax", dias_validacion=args.validation_days)
    training = procesar_datos_para_conteo(training, args)
    test = procesar_datos_para_conteo(test, args)

    for i, k in enumerate(training):
        print(f"Training {i}")

    for test_i, k in enumerate(test):
        print(f"Test {test_i}")

    print({
        "steps_per_epoch": i,
        "validation_steps": test_i,

    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--forbidden_timesteps', help='Horizonte temporal de la prediccion', type=eval)
    parser.add_argument('--modelname', help='nombre del modelo')
    parser.add_argument('--dataset', help='fichero de datos a cargar')
    parser.add_argument('--frequency', help='frecuencia de los datos', type=eval)
    parser.add_argument('--validation_days', help='días del final del dataset utilizados para la validación', type=eval)
    parser.add_argument('--model', help='modelo a usar')

    parser.add_argument('--stride', help='stride para los fragmentos de aprendizaje', type=eval)
    parser.add_argument('--length', help='tamaño de los fragmentos de aprendizaje', type=eval)
    parser.add_argument('--batch_size', help='tamaño del batch size', type=eval)
    parser.add_argument('--nfeatures', help='numero de features', type=eval)

    parser.add_argument("--activation", help='función de activación')
    parser.add_argument('--conv_size', help='tamaño de la convolución', type=eval)
    parser.add_argument('--optimizer', help="optimizador a usar", default="Adadelta()", type=str)
    parser.add_argument('--epochs', help='Epocas', type=int)
    parser.add_argument('--mode', help='Modo del modelo (regression, classification, delta)')
    parser.add_argument('--min_delta',
                        help='Mínima diferencia (normalizada) para considerar incremento o decremento en modo classification',
                        type=float)
    parser.add_argument('--droprate', help='Drop Rate', type=float)
    parser.add_argument('--num_hidden_dense_layers', help='numero capas ocultas', type=eval)
    parser.add_argument('--hidden_dense_layers_units', help='neuronas capas ocultas', type=eval)
    parser.add_argument('--max_pool', help='Hacer o no el MaxPooling', type=bool)
    parser.add_argument("--num_convolutional_layers", help='numero de capas convolucionales', type=eval)

    parser.add_argument("--validation_steps", help='numero de steps de validacion por epoch', type=eval)
    parser.add_argument("--steps_per_epoch", help='numero de steps de training por epoch', type=eval)

    parser.add_argument("--json", help='json con parametros como base')
    parser.add_argument("-count", help='devuelve el numero de steps de training y test', action="store_true")

    args = parser.parse_args()

    if args.json is not None:
        with open(args.json, "r") as f:
            parameters_from_json = json.load(f)

        args = load_parameters_from_dict(parameters_from_json, args)

    if args.count:
        devolver_conteo(args)
    else:
        train_model(args)
