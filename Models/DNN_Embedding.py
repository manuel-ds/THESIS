from keras.layers import Concatenate, Dense, Embedding, Flatten, Input, BatchNormalization, Dropout
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

def quantile_loss(q, y_true, y_pred):
    error = (y_true - y_pred)
    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1)


def validate_model(calendar, exog, target, Y_train_scaled_df, Y_train_scaled_y, Y_val_scaled_df, Y_val_scaled_y):

    # Prepare train input data
    calendar_features = Y_train_scaled_df[calendar].values

    past_regressors = Y_train_scaled_df[exog].values

    target_variable = Y_train_scaled_y[target].values

    # Prepare val input data
    calendar_features_val = Y_val_scaled_df[calendar].values

    past_regressors_val = Y_val_scaled_df[exog].values

    target_variable_val = Y_val_scaled_y[target].values
    # Define the hyperparameter search space
    def build_model(hp):
    # Define the input layers
        hour_input = Input(shape=(1,), name='hour')
        dow_input = Input(shape=(1,), name='day_of_week')
        month_input = Input(shape=(1,), name='month')
        dom_input = Input(shape=(1,), name='day_of_month')
        year_input = Input(shape=(1,), name='year')


        past_regressors_input = Input(shape=(past_regressors.shape[1],), name='past_regressors')

        # Embedding layer for "hour" variable
        hour_embedding_dim = hp.Int('hour_embedding_dim', min_value=4, max_value=10)
        embedded_hour = Embedding(input_dim=24, output_dim=hour_embedding_dim, input_length=1, name='embedding_hour')(hour_input)
        flatten_hour = Flatten()(embedded_hour)

        # Embedding layer for "day of week" variable
        dow_embedding_dim = hp.Int('dow_embedding_dim', min_value=3, max_value=5)
        embedded_dow = Embedding(input_dim=9, output_dim=dow_embedding_dim, input_length=1, name='embedding_dow')(dow_input)
        flatten_dow = Flatten()(embedded_dow)

        # Embedding layer for "month" variable
        month_embedding_dim = hp.Int('month_embedding_dim', min_value=3, max_value=6)
        embedded_month = Embedding(input_dim=12, output_dim=month_embedding_dim, input_length=1, name='embedding_month')(month_input)
        flatten_month = Flatten()(embedded_month)

        # Embedding layer for "day of month" variable
        dom_embedding_dim = hp.Int('dom_embedding_dim', min_value=6, max_value=14)
        embedded_dom = Embedding(input_dim=31, output_dim=dom_embedding_dim, input_length=1, name='embedding_dom')(dom_input)
        flatten_dom = Flatten()(embedded_dom)

        # Embedding layer for "year" variable
        year_embedding_dim = hp.Int('year_embedding_dim',  min_value=2, max_value=4)
        embedded_year = Embedding(input_dim=6, output_dim=year_embedding_dim, input_length=1, name='embedding_year')(year_input)
        flatten_year = Flatten()(embedded_year)


        # Concatenate the embedding layers and past_deviations_input
        merged_inputs = Concatenate()([flatten_hour,
                                    flatten_dow, flatten_month,
                                    flatten_dom, flatten_year,
                                    past_regressors_input])

        quantiles = [0.1, 0.5, 0.9]  # Example quantiles
        outputs = []
        # Define the hidden layers
        units_1 = hp.Int('units_1', min_value=32, max_value=512, step=32)
        units_2 = hp.Int('units_2', min_value=32, max_value=512, step=32)

        hidden_1 = Dense(units=units_1, activation='relu')(merged_inputs)
        dropout = hp.Choice('dropout', values=[True, False])
        if dropout == True:
            dropout_layer = Dropout(rate=hp.Choice('dropout_value', values=[0.0, 0.2, 0.4]))(hidden_1)
            hidden_2 = Dense(units=units_2, activation='relu')(dropout_layer)
        else:
            hidden_2 = Dense(units=units_2, activation='relu')(hidden_1)
        # Create separate output neurons for each time step
        #outputs = []
        for i in range(24):
            output = Dense(1, activation='linear', name=f'time_step_{i+1}')(hidden_2)
            outputs.append(output)


        # Create the model
        model = keras.Model(inputs=[hour_input,
                                    dow_input, month_input,
                                    dom_input, year_input,
                                    past_regressors_input],
                                    outputs=output)
        
        # Compile the model with MAE as the loss function
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                            loss='mse',
                            metrics=['mae'])

        return model

    # Instantiate the tuner
    tuner = RandomSearch(build_model,
                        objective='val_loss',
                        max_trials=10,
                        executions_per_trial=1,
                        directory='my_dir',
                        project_name='my_proj',
                        overwrite = True)

    # Train the tuner
    tuner.search(x=[calendar_features[:, 0], calendar_features[:, 1], calendar_features[:, 2],
                            calendar_features[:, 3], calendar_features[:, 4],
                        past_regressors],
                        y=[target_variable[:, i] for i in range(24)],

                validation_data=([calendar_features_val[:, 0], calendar_features_val[:, 1], calendar_features_val[:, 2],
                        calendar_features_val[:, 3], calendar_features_val[:, 4],
                        past_regressors_val],
                        [target_variable_val[:, i] for i in range(24)]),

                epochs=10, batch_size=32)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps



def model(exog, calendar, target, best_hps, Y_test_scaled_df, Y_trainval_scaled_df, Y_trainval_scaled_y, Y_val_scaled_df, Y_val_scaled_y):
    
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

    # Define the input layers
    hour_input = Input(shape=(1,), name='hour')
    dow_input = Input(shape=(1,), name='day_of_week')
    month_input = Input(shape=(1,), name='month')
    dom_input = Input(shape=(1,), name='day_of_month')
    year_input = Input(shape=(1,), name='year')


    past_regressors_input = Input(shape=(past_regressors.shape[1],), name='past_regressors')

    # Embedding layer for "hour" variable
    hour_embedding_dim = best_hps.values['hour_embedding_dim']
    embedded_hour = Embedding(input_dim=24, output_dim=hour_embedding_dim, input_length=1, name='embedding_hour')(hour_input)
    flatten_hour = Flatten()(embedded_hour)

    # Embedding layer for "day of week" variable
    dow_embedding_dim = best_hps.values['dow_embedding_dim']
    embedded_dow = Embedding(input_dim=9, output_dim=dow_embedding_dim, input_length=1, name='embedding_dow')(dow_input)
    flatten_dow = Flatten()(embedded_dow)

    # Embedding layer for "month" variable
    month_embedding_dim = best_hps.values['month_embedding_dim']
    embedded_month = Embedding(input_dim=12, output_dim=month_embedding_dim, input_length=1, name='embedding_month')(month_input)
    flatten_month = Flatten()(embedded_month)

    # Embedding layer for "day of month" variable
    dom_embedding_dim = best_hps.values['dom_embedding_dim']
    embedded_dom = Embedding(input_dim=31, output_dim=dom_embedding_dim, input_length=1, name='embedding_dom')(dom_input)
    flatten_dom = Flatten()(embedded_dom)

    # Embedding layer for "year" variable
    year_embedding_dim = best_hps.values['year_embedding_dim']
    embedded_year = Embedding(input_dim=6, output_dim=year_embedding_dim, input_length=1, name='embedding_year')(year_input)
    flatten_year = Flatten()(embedded_year)



    # Concatenate the embedding layers and past_deviations_input
    merged_inputs = Concatenate()([flatten_hour,
                                    flatten_dow, flatten_month,
                                flatten_dom, flatten_year,
                                past_regressors_input])

    quantiles = [0.1, 0.5, 0.9]  # Example quantiles
    num_models=len(quantiles)
    predictions = [] # store predicitons

    for i, q in enumerate(quantiles):
        # Add dense layers
        hidden_1 = Dense(best_hps.values['units_1'], activation='relu')(merged_inputs)
        hidden_2 = Dense(best_hps.values['units_2'], activation='relu')(hidden_1)

        # Create separate output neurons for each time step
        outputs = []
        for i in range(24):
            output = Dense(1, activation='linear', name=f'time_step_{i+1}')(hidden_2)
            outputs.append(output)

        # Create the model
        model = keras.Model(inputs=[hour_input,
                                    dow_input, month_input,
                                dom_input, year_input,
                                past_regressors_input],
                                outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=best_hps.values['learning_rate'])
        model.compile(loss=lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam')

        # Train the model on the combined training and validation sets
        model.fit(x=[trainval_calendar_features[:, 0], trainval_calendar_features[:, 1], trainval_calendar_features[:, 2],
                            trainval_calendar_features[:, 3], trainval_calendar_features[:, 4],
                            trainval_past_regressors],
                            y=[trainval_target[:, i] for i in range(24)],

                validation_data=([calendar_features_val[:, 0], calendar_features_val[:, 1], calendar_features_val[:, 2],
                        calendar_features_val[:, 3], calendar_features_val[:, 4],
                        past_regressors_val],
                        [target_variable_val[:, i] for i in range(24)]),

                    epochs=10, batch_size=32,
                    callbacks=[tensorboard_callback],
                    verbose=2)

        # Predict on the test data
        prediction = model.predict([test_calendar_features[:, 0], test_calendar_features[:, 1],
                                        test_calendar_features[:, 2], test_calendar_features[:, 3],
                                        test_calendar_features[:, 4],
                                        test_past_regressors])
        predictions.append(prediction)

    predictions = np.asarray(predictions)
    preds = np.squeeze(predictions)
    return preds
