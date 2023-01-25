import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


# OPGAVE 1a
def plot_image(img, label):
    # Deze methode krijgt een matrix mee (in img) en een label dat correspondeert met het 
    # plaatje dat in de matrix is weergegeven. Zorg ervoor dat dit grafisch wordt weergegeven.
    # Maak gebruik van plt.cm.binary voor de cmap-parameter van plt.imgshow.

    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(label)
    plt.show()


# OPGAVE 1b
def scale_data(X):
    # Deze methode krijgt een matrix mee waarin getallen zijn opgeslagen van 0..m, en hij 
    # moet dezelfde matrix retourneren met waarden van 0..1. Deze methode moet werken voor 
    # alle maximale waarde die in de matrix voorkomt.
    # Deel alle elementen in de matrix 'element wise' door de grootste waarde in deze matrix.

    return X / X.max()


# OPGAVE 1c
def build_model():
    # Deze methode maakt het keras-model dat we gebruiken voor de classificatie van de mnist
    # dataset. Je hoeft deze niet abstract te maken, dus je kunt er van uitgaan dat de input
    # layer van dit netwerk alleen geschikt is voor de plaatjes in de opgave (wat is de 
    # dimensionaliteit hiervan?).
    # Maak een model met een input-laag, een volledig verbonden verborgen laag en een softmax
    # output-laag. Compileer het netwerk vervolgens met de gegevens die in opgave gegeven zijn
    # en retourneer het resultaat.

    # Het staat je natuurlijk vrij om met andere settings en architecturen te experimenteren.

    img_size = 28
    img_pixels = img_size ** 2
    model = keras.Sequential([
        layers.Dense(img_pixels, input_shape=(img_size, img_size)),  # Laag 1
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  # Laag 2
        layers.Dense(10, activation='softmax')  # Laag 3
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# OPGAVE 2a
def conf_matrix(labels, pred):
    # Retourneer de econfusion matrix op basis van de gegeven voorspelling (pred) en de actuele
    # waarden (labels). Check de documentatie van tf.math.confusion_matrix:
    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix

    # YOUR CODE HERE
    return tf.math.confusion_matrix(labels, pred)


# OPGAVE 2b
def conf_els(conf, labels):
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is 
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de 
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel 
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)

    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
    # https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html

    # YOUR CODE HERE
    return_values = []
    tp_list = np.diagonal(conf)
    label_count = len(labels)
    for i in range(label_count):
        tp = tp_list[i]
        fp = np.sum(conf[:, i]) - tp
        fn = np.sum(conf[i]) - tp
        tn = np.sum(conf) - tp - fp - fn
        return_values.append((labels[i], tp, fp, fn, tn))

    return return_values


# OPGAVE 2c
def conf_data(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en 
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for item in metrics:
        tp += item[1]
        fp += item[2]
        fn += item[3]
        tn += item[4]

    tpr = tp / (tp + fn)
    ppv = tp / (tp + fp)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)

    rv = {'tpr': tpr, 'ppv': ppv, 'tnr': tnr, 'fpr': fpr}
    return rv
