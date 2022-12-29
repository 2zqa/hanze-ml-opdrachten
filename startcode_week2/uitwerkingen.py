import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


# ==== OPGAVE 1 ====
def plot_number(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
    nrVector = nrVector.reshape((20, 20), order='F')
    plt.matshow(nrVector)
    plt.show()


# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.
    return 1 / (1 + np.exp(-z))


# ==== OPGAVE 2b ====
def get_y_matrix(y, unused=0):
    # Gegeven een vector met waarden x van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van
    # y en m
    width = np.max(y)
    cols = y.flatten() - 1
    rows = [i for i in range(len(cols))]
    data = np.ones(len(cols))
    return csr_matrix((data, (rows, cols)), shape=(len(rows), width)).toarray()


# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predict_number(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk.

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    # L=1
    m, n = X.shape
    a1 = np.c_[np.ones(m), X]  # Add bias nodes (5000, 401)

    # L=2
    z2 = a1.dot(Theta1.T)  # (5000, 401) @ (401, 25) := (5000, 25)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(m), a2]  # Add bias nodes (5000, 26)

    # L=L
    z3 = a2.dot(Theta2.T)  # (5000, 26) @ (26, 10) := (5000, 10)
    a3 = sigmoid(z3)

    return a3


# ===== deel 2: =====
def compute_cost(Theta1, Theta2, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta1 en Theta2) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een matrix. 
    y_mat = get_y_matrix(y)

    m, n = X.shape
    prediction = predict_number(Theta1, Theta2, X)
    return 1 / m * np.sum(-y_mat * np.log(prediction) - (1 - y_mat) * np.log(1 - prediction))


# ==== OPGAVE 3a ====
def sigmoid_gradient(z):
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.

    sigmoid_z = sigmoid(z)
    return sigmoid_z * (1 - sigmoid_z)


# ==== OPGAVE 3b ====
def nn_check_gradients(Theta1, Theta2, X, y):
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    # ==Forward propagation==
    # L=1
    m, n = X.shape
    a1 = np.c_[np.ones(m), X]  # Add bias nodes (5000, 401)

    # L=2
    z2 = a1.dot(Theta1.T)  # (5000, 401) @ (401, 25) := (5000, 25)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(m), a2]  # Add bias nodes (5000, 26)

    # L=L=3
    z3 = a2.dot(Theta2.T)  # (5000, 26) @ (26, 10) := (5000, 10)
    a3 = sigmoid(z3)  # (5000, 10)

    # ==Back propagation==
    y_vec = get_y_matrix(y)
    delta_layer3 = a3 - y_vec  # (5000, 10)
    # Could also run sigmoid_gradient but pre-existing a2 variable already has the added 1 column :)
    quick_sigmoid_gradient = a2 * (1 - a2)
    delta_layer2 = delta_layer3 @ Theta2 * quick_sigmoid_gradient  # (5000, 26)

    # Theta1/delta2: (25, 401)
    # Theta2/delta3: (10, 26)
    delta3 = delta_layer3.T @ a2  # Should be (10, 26)
    delta2 = delta_layer2.T[1:] @ a1  # [1:] removes the first row. Should be (25, 401)

    delta2_grad = delta2 / m
    delta3_grad = delta3 / m

    return delta2_grad, delta3_grad
