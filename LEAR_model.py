from sklearn.linear_model import LassoLarsIC, Lasso
import pandas as pd

def laer_model(Y_trainval_scaled_df, Y_trainval_scaled_y, Y_test_scaled_df, Y_test_scaled_y, Y_val_scaled_df, Y_test_df, target, scaler_2):

    calibration_window= 365 * 3

    # trainval
    Y_trainval_scaled_df.reset_index(inplace=True)
    Y_trainval_scaled_df.drop('index', axis=1, inplace=True)

    Y_trainval_scaled_y.reset_index(inplace=True)
    Y_trainval_scaled_y.drop('index', axis=1, inplace=True)

    # test
    Y_test_scaled_df.reset_index(inplace=True)
    Y_test_scaled_df.drop('index', axis=1, inplace=True)

    new_index = pd.RangeIndex(start=1711, stop=1711+len(Y_test_scaled_df))
    Y_test_scaled_df.set_index(new_index, inplace=True)
    Y_test_scaled_y.set_index(new_index, inplace=True)

    Y_scaled_df = pd.concat([Y_trainval_scaled_df, Y_test_scaled_df], axis=0)
    Y_scaled_y = pd.concat([Y_trainval_scaled_y, Y_test_scaled_y], axis=0)

    from sklearn.linear_model import Lasso, LassoLarsIC

    start_val = len(Y_trainval_scaled_df)-len(Y_val_scaled_df) # index of trainval used for validation

    start_test = len(Y_scaled_df)-len(Y_val_scaled_df) # start for test set

    models = {}
    df_predictions_all = pd.DataFrame()

    for i in range(start_test, len(Y_scaled_df), 7):

        preds = {}

        for h in range(24):

            # Estimating lambda hyperparameter using LARS
            param_model = LassoLarsIC(criterion='aic', max_iter=2500)
            param = param_model.fit(Y_scaled_df.iloc[i-calibration_window:i, :].values, Y_scaled_y.iloc[i-calibration_window:i, h].values).alpha_

            # Re-calibrating Lasso using standard LASSO estimation technique
            model = Lasso(max_iter=2500, alpha=param)

            model.fit(Y_scaled_df.iloc[i-calibration_window:i, :].values, Y_scaled_y.iloc[i-calibration_window:i, h].values)

            models[h] = model

            index = Y_scaled_df.iloc[i-calibration_window:i, :].index[-1]
            next_day = index + 1
            next_week = index + 8

            preds[f'time_step_{h+1}_q2'] = models[h].predict(Y_scaled_df.iloc[next_day:next_week, :])

        df_predictions = pd.DataFrame.from_dict(preds)
        df_predictions_all = pd.concat([df_predictions_all, df_predictions], axis=0)

        print(f'{i} done')
        df_predictions_rescaled = pd.DataFrame(scaler_2.inverse_transform(df_predictions_all), columns=df_predictions_all.columns)

        test_df = Y_test_df[target]

        test_df.reset_index(inplace=True)

        merged_df = pd.concat([test_df, df_predictions_rescaled], axis=1)

        actual = []
        pred_q2 = []

        total_rows = merged_df.shape[0]
        for row in range(0, total_rows):
            for i in range(0,24):
                actual.append(merged_df[f'y_t+{i}'].iloc[row])

                pred_q2.append(merged_df[f'time_step_{i+1}_q2'].iloc[row])

        df_final = pd.DataFrame(list(zip( actual, pred_q2)), columns=['actual',  'pred_q2'])
        return df_final
