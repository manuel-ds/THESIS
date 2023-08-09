from keras.layers import Concatenate, Dense, Embedding, Flatten, Input,  Dropout, Reshape
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

# Create the EarlyStopping callback
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

def negative_log_likelihood(y_true, y_pred):
    mean = y_pred[:, :, 0]
    std = y_pred[:, :, 1]

    dist = tfp.distributions.Normal(mean, std)
    log_likelihood = dist.log_prob(y_true)
    return -tf.reduce_mean(log_likelihood)

class GaussianLayer(Layer):
    def __init__(self, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        mean = x[:, :, :1]
        std = K.exp(0.5 * x[:, :, 1:])
        return K.concatenate([mean, std], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2)

log_dir='logs'
# Create a dictionary to map embedding layer names to their metadata files
embeddings_metadata = {
    'embedding_hour': 'metadata/metadata_hour.tsv',
    'embedding_dow': 'metadata/metadata_dow.tsv',
    'embedding_month': 'metadata/metadata_month.tsv',
    'embedding_dom': 'metadata/metadata_dom.tsv',
    'embedding_year': 'metadata/metadata_year.tsv',
}

# Initialize the TensorBoard callback with the embeddings metadata
tensorboard_callback = TensorBoard(log_dir=log_dir, embeddings_freq=1, embeddings_metadata=embeddings_metadata)



predictions = [] # store predicitons

def my_probmodel_dist_gaus(neurons, activation, past_regressors):

    # Define the input layers
    hour_input = Input(shape=(1,), name='hour')
    dow_input = Input(shape=(1,), name='day_of_week')
    month_input = Input(shape=(1,), name='month')
    dom_input = Input(shape=(1,), name='day_of_month')
    year_input = Input(shape=(1,), name='year')


    past_regressors_input = Input(shape=(past_regressors.shape[1],), name='past_regressors')

    # Embedding layer for "hour" variable
    hour_embedding_dim = 8
    embedded_hour = Embedding(input_dim=24, output_dim=hour_embedding_dim, input_length=1, name='embedding_hour')(hour_input)
    flatten_hour = Flatten()(embedded_hour)

    # Embedding layer for "day of week" variable
    dow_embedding_dim = 3
    embedded_dow = Embedding(input_dim=9, output_dim=dow_embedding_dim, input_length=1, name='embedding_dow')(dow_input)
    flatten_dow = Flatten()(embedded_dow)

    # Embedding layer for "month" variable
    month_embedding_dim = 4
    embedded_month = Embedding(input_dim=12, output_dim=month_embedding_dim, input_length=1, name='embedding_month')(month_input)
    flatten_month = Flatten()(embedded_month)

    # Embedding layer for "day of month" variable
    dom_embedding_dim = 10
    embedded_dom = Embedding(input_dim=31, output_dim=dom_embedding_dim, input_length=1, name='embedding_dom')(dom_input)
    flatten_dom = Flatten()(embedded_dom)

    # Embedding layer for "year" variable
    year_embedding_dim = 4
    embedded_year = Embedding(input_dim=6, output_dim=year_embedding_dim, input_length=1, name='embedding_year')(year_input)
    flatten_year = Flatten()(embedded_year)



    # Concatenate the embedding layers and past_deviations_input
    merged_inputs = Concatenate()([flatten_hour,
                                    flatten_dow, flatten_month,
                                flatten_dom, flatten_year,
                                past_regressors_input])
    # Rest of the model code
    # Add dense layers
    h = Dense(neurons, activation=activation)(merged_inputs)
    h = Dense(neurons, activation=activation)(h)
    drop = Dropout(0.2)(h)
    # Create Gaussian output layer
    output = Dense(48)(drop)  # Output shape: (None, 24, 48)
    output = Reshape((24, 2))(output)  # Reshape to (None, 24, 2)
    output = GaussianLayer()(output)  # Apply Gaussian layer

    # Create the model
    model = keras.Model(inputs=[hour_input, dow_input, month_input, dom_input, year_input, past_regressors_input],
                        outputs=output)
    return model

# use the model

def fit_predict_model(Y_trainval_scaled_df, Y_test_scaled_df, Y_val_scaled_df, Y_trainval_scaled_y, Y_val_scaled_y, calendar, exog, target):
    
    past_regressors = Y_trainval_scaled_df[exog].values

    #Prepare test input data
    test_calendar_features = Y_test_scaled_df[calendar].values
    test_past_regressors = Y_test_scaled_df[exog].values

    # training + validation input data
    trainval_calendar_features = Y_trainval_scaled_df[calendar].values

    trainval_past_regressors = Y_trainval_scaled_df[exog].values

    trainval_target = Y_trainval_scaled_y[target].values

    # Prepare val input data
    calendar_features_val = Y_val_scaled_df[calendar].values

    past_regressors_val = Y_val_scaled_df[exog].values

    target_variable_val = Y_val_scaled_y[target].values
   
    model = my_probmodel_dist_gaus(160, 'relu')


    # Compile the model with the loss function
    opt = Adam(0.0001)
    model.compile(optimizer=opt, loss=negative_log_likelihood)

    # Train the model
    history = model.fit(
        x=[
            trainval_calendar_features[:, 0],
            trainval_calendar_features[:, 1],
            trainval_calendar_features[:, 2],
            trainval_calendar_features[:, 3],
            trainval_calendar_features[:, 4],
            trainval_past_regressors
        ],
        y=trainval_target,
        validation_data=(
            [
                calendar_features_val[:, 0],
                calendar_features_val[:, 1],
                calendar_features_val[:, 2],
                calendar_features_val[:, 3],
                calendar_features_val[:, 4],
                past_regressors_val
            ],
            target_variable_val
        ),
        epochs=20,
        batch_size=32,
        callbacks=[tensorboard_callback, early_stopping]
    )

    # Predict on the test data
    preds = model.predict(
        [
            test_calendar_features[:, 0],
            test_calendar_features[:, 1],
            test_calendar_features[:, 2],
            test_calendar_features[:, 3],
            test_calendar_features[:, 4],
            test_past_regressors
        ]
    )

    import tensorflow as tf
    import tensorflow_probability as tfp
    import numpy as np

    medians = np.zeros((4560, 24))
    lower_quantiles = np.zeros((4560, 24))
    upper_quantiles = np.zeros((4560, 24))

    # Compute medians and quantiles for each time step
    for i in range(24):
        # Extract mean and standard deviation for the current time step
        mean = preds[:, i, 0]
        stddev = preds[:, i, 1]

        # Normal distribution object
        distribution = tfp.distributions.Normal(loc=mean, scale=stddev)

        # quantiles
        medians[:, i] = distribution.mean()
        lower_quantiles[:, i] = distribution.quantile(0.05)
        upper_quantiles[:, i] = distribution.quantile(0.95)

    preds_2 = np.stack((lower_quantiles, medians, upper_quantiles), axis=2)
    # Transpose
    preds_2 = np.transpose(preds, (2, 1, 0))
    return preds_2


