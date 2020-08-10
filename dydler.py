# limpiando la pantalla
from os import system
system('cls')
# Importación de libreías para manejo de datos y deep learning 

print("\nCargando librerias para análisis de datos...")
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from sys import argv



# Importando modulo para creacion de conjuntos de entrenamiento, validacion y prueba
from modules import split_train_test_val, create_tf_model
print("Carga completa!")


# carga de datos
print("\nCargando datos...\n")
archivo = argv[1]
datos = pd.read_csv(archivo)
print(datos)
print("\nDatos cargados correctamente!\n\n")


# creando conjuntos de datos para entrenamiento, validacion y prueba
train_set, test_set, val_set = split_train_test_val(datos)


# parametros para los datos, extrayendo caracteristicas y columna de resultados
num_columnas = 24
x_train, y_train = train_set.iloc[:,2:num_columnas], train_set.iloc[:,num_columnas]
x_test, y_test = test_set.iloc[:,2:num_columnas], test_set.iloc[:,num_columnas]
x_val, y_val = val_set.iloc[:,2:num_columnas], val_set.iloc[:,num_columnas]


# modelo de red neuronal con tf
#layers = [22, 100, 50, 25, 3]
layers = [22, 50, 25, 12, 3]

model = create_tf_model(layers)

# optimizador y perdida del modelo
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# entrenando al modelo
epochs = int(argv[2])
model_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs)

# mostrando resultados
pd.DataFrame(model_history.history).plot()
plt.title("Error durante entrenamiento y validación")
plt.xlabel("Epocas")
plt.ylabel("Error")

option = ""

while option != "0" or option != "1":
	option = input("\n\nGuardar imagen (0) o mostrar (1): ")

	if option == "0":
		output_name = input("Ingrese el nombre para la figura: ")
		plt.savefig(output_name + ".svg")
		break
	elif option == "1":
		plt.show()
		break
	else:
		print("Opción incorrecta!")


# resultados obtenidos
print(f"\n")

# Evaluando el modelo
print("\nEvaluando la precisión del módelo...\n")
loss, accuracy = model.evaluate(x_test, y_test)

print(f"\nLa precisión del módelo es {round(accuracy * 100, 3)}")
