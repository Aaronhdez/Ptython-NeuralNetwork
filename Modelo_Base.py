import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import backend as K
import keras
from time import time


# DATA SOURCE --------------------------------------------------

batch_size = 20

train_data_dir = 'D:/Universidad/OneDrive - Universidad de Las Palmas de Gran Canaria/TERCERO/Fundamentos de los Sistemas Inteligentes/RedesNeuronales/cars/train'
validation_data_dir = 'D:/Universidad/OneDrive - Universidad de Las Palmas de Gran Canaria/TERCERO/Fundamentos de los Sistemas Inteligentes/RedesNeuronales/cars/test'

# Rescalamos todas las imagenes diviendo cada canal de color color entre 255
# para pasarlo a un formato real --> (255,255,255) = (1,1,1)
# Las rotamos 15 grados izqierda y derecha para ganar muestras
# Hacemos zoom para ganar muestras
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1
)

# Apliacamos el ismo cambio para las imagenes de validación
validation_datagen = ImageDataGenerator(
        rescale=1./255
)

# El train generator genera un modelo de entrnemiento sencillo para entrenar un
# conjunto de muestras
# Definimos el directorio, el tamaño de muestras (150x150), el tamaño de cada lote y el
# modo de diferenciacion, que debe ser categorical
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(250, 150),
        batch_size=batch_size,
        class_mode='categorical')

# Aplicamos los mismos cambios para el conjunto de validación
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(250, 150),
        batch_size=batch_size,
        class_mode='categorical')

# MODEL --------------------------------------------------

# Estabñecemos el modo de trabajo (sequential)
model = Sequential()

# Establecemos el numero de canales, el tamaño del kernel (3x3 para cada neurona),
# su función de activacion (Rectification Linear Unit) y su ajuste
# Establecemos llos cnaales que tendrá la capa base (150x150x3 canales)

# Después de cada capa convolutiva, aplicamos una reducción (filtrado al maximo con MaxPooling)
# empleando de base grupos de 2x2
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(250, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Si nos fijamos, en la primera  reducción se pierde n-1 por cada lateral del tensor
# y le palicamos 64 filtros para compensar.
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Colocamos el flatten y la ultima capa full connected de 128
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Una vez establecidas las capas intermedias y los reductores, configuramos el dropout,
# el cual se sitúa para este caso
model.add(Dropout(0.5))

#Cambiado de 10 a 6 debido a que son 6 categorías de imágenes
model.add(Dense(6, activation='softmax'))

# Compilamos el modelo y especificamos la formula de perfida. Para este caso establecemos
# la función pérdida en categoritcal_crossentropy. Especificamos además la fórmula de
# aproximación (Adadelta es una variante del Descenso por Gradiente). La metrica que
# queremos es la precisión, por lo que usaremos accuracy para verificar el progreso
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# TRAINING --------------------------------------------------

# Indicamos las epoch
epochs = 50

# Indicamos el límite de parada
#es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3, restore_best_weights=True)

# Entrenamos la red neuronal con el train generator y le indicamos que valide los datos necesarios
model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data = validation_generator,
        #callbacks = [es]
)

# # SAVING --------------------------------------------------
#
# model.save("mimodelo.h5")
#
# # PRODUCTION ----------------------------------------------
#
# from matplotlib.pyplot import imshow
# import numpy as np
# from PIL import Image
# import keras
#
# model = keras.models.load_model("mimodelo.h5")
#
# %matplotlib inline
# pil_im = Image.open('./data/Sign-Language/5/IMG_1123.jpg', 'r')
# im = np.asarray(pil_im.resize((150, 150)))
# imshow(im)
# print(im.shape) # La imagen es un array de dimensión: 150x150x3
#
# # El método `predict` hace la predicción de un lote de entradas, no solo una.
# # En el caso de que tengamos solo una entrada deberemos añadirle una dimensión más
# # al array numpy para que la entrada tenga la dimensión: 1x150x150x3
#
# im = im.reshape(1,150,150,3)
# model.predict(im)