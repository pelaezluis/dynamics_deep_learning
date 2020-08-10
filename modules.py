# funciones propias para creación de conjunto de entrenamiento, validacion
# y prueba
import numpy as np


# Deep learning with tensorflow
print("Cargando tensorflow...")
from tensorflow import keras


def split_train_test_val(data, test_size=0.2, val_size=0.3):
    """
    Separando los datos en train y test
    """

    num_data = len(data)
    indices_data = list(range(num_data))
    print("*********************************************************************************")
    print(f"Cantidad de filas de datos:\t{num_data}")
    print(f"Cantidad de datos:\t\t{num_data * 23}")
    np.random.shuffle(indices_data)
    split_train_test = int(np.floor(num_data * test_size))
    train_set_idx, test_set_idx = indices_data[split_train_test:], indices_data[:split_train_test]
    train_set = data.loc[train_set_idx]
    test_set = data.loc[test_set_idx]
    print(f"Conjunto de entrenamiento:\t{len(train_set)}\tConjunto de prueba:\t{len(test_set)}")


    """
    Separando los datos en validación y entrenamiento
    """
    num_train = len(train_set)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_train_val = int(np.floor(val_size * num_train))
    train_idx, val_idx = indices_train[split_train_val:], indices_train[:split_train_val]
    train_set = data.loc[train_idx]
    val_set = data.loc[val_idx]
    print(f"conjunto de entrenamiento:\t{len(train_set)}\tConjunto de validacion:\t{len(val_set)}")
    print("*********************************************************************************\n\n")

    return (train_set, test_set, val_set)

def create_tf_model(neurons_per_layer, activation_function="relu", dropout=0.2):

    """
    This function create the NN model and
    add layers of neurons
    """
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(neurons_per_layer[0],)))
    
    for layer in neurons_per_layer[1:-1]:
        model.add(keras.layers.Dense(layer, activation=activation_function))
        model.add(keras.layers.Dropout(dropout))
    
    model.add(keras.layers.Dense(neurons_per_layer[-1], activation='softmax'))
    #model.summary()
    print("\n\n")
    return model