import keras.backend as K
from itertools import product
from functools import partial
import numpy as np

# We create a new loss function that will penalize more when we do no detect fire when there is (false negative)


def w_categorical_crossentropy(y_true, y_pred, weights):

    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    #y_pred_max_mat = K.cast(y_pred_max_mat, dtype='float32')
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p], K.floatx()) * K.cast(y_pred_max_mat[:, c_p], K.floatx()) * K.cast(
            y_true[:, c_t], K.floatx()))
    return K.categorical_crossentropy(y_true, y_pred) * final_mask

labels= {'Fire': 0, 'Fog': 1, 'Not_fire': 2, 'Red_object': 3}

nb_cl = len(labels)
w_array = np.ones((nb_cl, nb_cl))
for i in range(1, nb_cl):
    w_array[0, i] = 1.5  # Coefficient avec lequel on veut pénaliser les faux négatifs de feu

ncce = partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'
