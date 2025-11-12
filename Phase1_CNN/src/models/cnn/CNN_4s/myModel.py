"""
CNN Model for 4-second window microsleep detection
Adapted for binary classification with 2 EOG channels
"""

from keras import backend as K
from keras.layers import concatenate
from keras import optimizers
from keras.layers import Layer, Dense, Dropout, Input, Activation, TimeDistributed, Reshape
from keras.layers import GRU, Bidirectional
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D, GlobalAveragePooling2D, SpatialDropout2D
from keras.models import Model
from keras.layers import GaussianNoise


def build_model(data_dim, n_channels, n_cl):
    """
    Build CNN model for microsleep detection
    
    Args:
        data_dim: dimension of input data (number of samples in window)
        n_channels: number of input channels (2 for EOG)
        n_cl: number of classes (2 for binary)
        
    Returns:
        tuple: (cnn_eeg, model)
            - cnn_eeg: feature extraction model
            - model: full classification model
    """
    eeg_channels = 1
    act_conv = 'relu'
    init_conv = 'glorot_normal'
    dp_conv = 0.3
    
    def cnn_block(input_shape):
        """
        CNN block for feature extraction
        """
        input_layer = Input(shape=input_shape)
        x = GaussianNoise(0.0005)(input_layer)
        
        # First conv block
        x = Conv2D(32, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
        x = BatchNormalization()(x)
        x = Activation(act_conv)(x)
        x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        
        # Second conv block
        x = Conv2D(64, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
        x = BatchNormalization()(x)
        x = Activation(act_conv)(x)
        x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        
        # Four blocks of Conv2D(128)
        for i in range(4):
            x = Conv2D(128, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
            x = BatchNormalization()(x)
            x = Activation(act_conv)(x)
            x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        
        # Six blocks of Conv2D(256)
        for i in range(6):
            x = Conv2D(256, (3, 1), strides=(1, 1), padding='same', kernel_initializer=init_conv)(x)
            x = BatchNormalization()(x)
            x = Activation(act_conv)(x)
            x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        
        flatten1 = Flatten()(x)
        cnn_eeg = Model(inputs=input_layer, outputs=flatten1)
        return cnn_eeg
    
    hidden_units1 = 256
    dp_dense = 0.5
    
    eeg_channels = 1
    eog_channels = 2
    
    # Input shape: (data_dim, 1, 2) for 2 EOG channels
    input_eeg = Input(shape=(data_dim, 1, n_channels))
    cnn_eeg = cnn_block((data_dim, 1, n_channels))
    x_eeg = cnn_eeg(input_eeg)
    x = BatchNormalization()(x_eeg)
    x = Dropout(dp_dense)(x)
    x = Dense(units=hidden_units1, activation=act_conv, kernel_initializer=init_conv)(x)
    x = BatchNormalization()(x)
    x = Dropout(dp_dense)(x)
    
    # Output layer for n_cl classes
    predictions = Dense(units=n_cl, activation='softmax', kernel_initializer=init_conv)(x)
    
    model = Model(inputs=[input_eeg], outputs=[predictions])
    return [cnn_eeg, model]

