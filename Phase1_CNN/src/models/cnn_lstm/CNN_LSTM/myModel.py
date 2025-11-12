"""
CNN-LSTM Model for microsleep detection
Hybrid architecture: CNN for feature extraction + LSTM for temporal modeling
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


def cnn_block(input_shape):
    """
    CNN block for feature extraction from EOG signals
    """
    input_layer = Input(shape=input_shape)
    dp_conv = 0.4
    act_conv = 'relu'
    
    x = GaussianNoise(0.0005)(input_layer)
    x = Conv2D(64, (3, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(act_conv)(x)
    x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
    
    x = Conv2D(128, (3, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(act_conv)(x)
    
    # 8 blocks of Conv2D(256)
    for i in range(8):
        x = Conv2D(256, (3, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(act_conv)(x)
        x = MaxPooling2D(padding="same", pool_size=(2, 1))(x)
    
    flatten1 = Flatten()(x)
    
    cnn_eeg = Model(inputs=input_layer, outputs=flatten1)
    return cnn_eeg


def build_model(data_dim, n_channels, n_cl):
    """
    Build CNN-LSTM model for microsleep detection
    
    Args:
        data_dim: dimension of input data (number of samples in window)
        n_channels: number of input channels (2 for EOG)
        n_cl: number of classes (2 for binary)
        
    Returns:
        Model: the full CNN-LSTM model
    """
    hidden_units = 256
    init_conv = 'glorot_normal'
    dp = 0.4
    
    # Input shape: (None, data_dim, 1, n_channels) where None is sequence length
    input_eeg = Input(shape=(None, data_dim, 1, n_channels))
    cnn_eeg = cnn_block((data_dim, 1, n_channels))
    
    print(cnn_eeg.summary())
    
    # Apply CNN to each time step
    x_eeg = TimeDistributed(cnn_eeg)(input_eeg)
    
    # First LSTM layer
    x = BatchNormalization()(x_eeg)
    x = Bidirectional(LSTM(units=32,
                          return_sequences=True, 
                          activation='tanh',
                          recurrent_activation='sigmoid', 
                          dropout=dp, 
                          recurrent_dropout=dp))(x)
    x = BatchNormalization()(x)
    
    # Second LSTM layer
    x = Bidirectional(LSTM(units=32,
                          return_sequences=True, 
                          activation='tanh',
                          recurrent_activation='sigmoid', 
                          dropout=dp, 
                          recurrent_dropout=dp))(x)
    x = BatchNormalization()(x)
    
    # Output layer for n_cl classes
    predictions = TimeDistributed(Dense(units=n_cl, activation='softmax', kernel_initializer=init_conv))(x)
    
    model = Model(inputs=[input_eeg], outputs=[predictions])
    return model

