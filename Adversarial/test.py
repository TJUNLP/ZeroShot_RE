import numpy as np
batch_size = 8
domain_labels = np.vstack([np.tile([0, 1], [batch_size // 2, 1]),
                           np.tile([1., 0.], [batch_size // 2, 1])])
print(domain_labels)
import keras.backend as K

K.switch