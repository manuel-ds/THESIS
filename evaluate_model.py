import pandas as pd

def evaluate(preds, Y_test_df, scaler_2, target):
    rescaled_all = pd.DataFrame()

    for index, element in enumerate(preds):
        df_predictions = pd.DataFrame()
        time_step_predictions = {}
        # Access predictions for each time step
        for i in range(0, 24):
            time_step = i + 1
            time_step_prediction = element[i]

            # Store the predictions in the dictionary
            time_step_predictions[f'time_step_{time_step}_q{str(index+1)}'] = pd.Series(time_step_prediction.flatten())

            # Create a DataFrame from the dictionary
            df_predictions = pd.concat(time_step_predictions, axis=1)

        df_predictions_rescaled = pd.DataFrame(scaler_2.inverse_transform(df_predictions), columns=df_predictions.columns)
        rescaled_all = pd.concat([rescaled_all, df_predictions_rescaled], axis=1)

    test_df = Y_test_df[target]
    test_df.reset_index(inplace=True)

    merged_df = pd.concat([test_df, rescaled_all], axis=1)
    # Determine the total number of rows in the DataFrame
    total_rows = merged_df.shape[0]

    df_final = pd.DataFrame()
    # Iterate over the indices in steps of 24
    for start_index in range(0, total_rows, 24):
        # Calculate the end index for each group
        end_index = start_index + 24

        # Extract the group of rows based on the start and end indices
        group_df = merged_df.iloc[start_index:end_index]

        date = []
        actual = []
        pred_q1 = []
        pred_q2 = []
        pred_q3 = []
        total_rows_group = group_df.shape[0]
        for i in range(0, total_rows_group):
            # Calculate the end index for each group
            date.append(group_df['ds'].iloc[i])
            actual.append(group_df['y_t+0'].iloc[i])
            for index, element in enumerate(preds):
                if index == 0:
                    pred_q1.append(group_df[f'time_step_{i+1}_q{str(index+1)[-1]}'].iloc[0])

                elif index == 1:
                    pred_q2.append(group_df[f'time_step_{i+1}_q{str(index+1)[-1]}'].iloc[0])

                elif index == 2:
                    pred_q3.append(group_df[f'time_step_{i+1}_q{str(index+1)[-1]}'].iloc[0])

        df_group = pd.DataFrame(list(zip(date, actual, pred_q1, pred_q2, pred_q3)),
                                columns=['date', 'actual', 'pred_q1', 'pred_q2', 'pred_q3'])


        df_final = pd.concat([df_final, df_group], axis=0)

    import numpy as np
    def winkler_score(y_true, y_lower, y_upper, alpha=0.1):
        delta = y_upper - y_lower

        score = np.where((y_true >= y_lower) & (y_true <= y_upper), delta, 0)
        score += np.where(y_true < y_lower, delta + (2 / alpha) * (y_lower - y_true), 0)
        score += np.where(y_true > y_upper, delta + (2 / alpha) * (y_true - y_upper), 0)

        return score.mean()

    def pi_width(y_lower, y_upper):
        delta = y_upper - y_lower
        return delta.mean()

    def coverage_error(y_true, y_lower, y_upper, alpha):
        coverage_prob = (y_lower <= y_true) & (y_true <= y_upper)
        return np.abs(coverage_prob.mean() - alpha)

    def pinball(y_true, y_quantile, alpha):
        pinball_loss = np.maximum((y_true - y_quantile) * alpha, (y_quantile - y_true) * (1 - alpha))
        return np.mean(pinball_loss)

    def unconditional_coverage_score(y_true, y_lower, y_upper):
        coverage_scores = ((y_true >= y_lower) & (y_true <= y_upper)).astype(int)
        return np.mean(coverage_scores)

    from neuralforecast.losses.numpy import mae, mape, smape
    from sklearn.metrics import mean_pinball_loss
    import matplotlib.pyplot as plt
    df_final['year_month'] = [x[:7] for x in df_final.index.astype('str')]

    results_df = pd.DataFrame()

    for month in df_final['year_month'].drop_duplicates():
        df_err = df_final[df_final['year_month']==month]
        plt.figure(figsize=(20,5))
        plt.plot(df_err.index, df_err['actual'], label='True', c='black')
        plt.plot(df_err.index, df_err['pred_q3'], c='steelblue', label='Pred')

        plt.fill_between(x=df_err.index,
                            y1=df_err['pred_q1'], y2=df_err['pred_q5'],
                            alpha=0.3, label='level 90', color='steelblue')
        plt.fill_between(x=df_err.index,
                            y1=df_err['pred_q2'], y2=df_err['pred_q4'],
                            alpha=0.3, label='level 90', color='steelblue')

        plt.legend()
        plt.plot()

        my_results = {'date': pd.to_datetime(month),
                    'MAE': round(mae(df_err['actual'], df_err['pred_q3']), 2),
                    'sMAPE': round(smape(df_err['actual'], df_err['pred_q3'])*100, 2),
                    'Pinball Q90': round(mean_pinball_loss(df_err['actual'], df_err['pred_q4'], alpha=0.9), 2),
                    'UC': round(unconditional_coverage_score(df_err['actual'], df_err['pred_q2'], df_err['pred_q4'])*100, 2),
                    'Winkler score': round(winkler_score(df_err['actual'], df_err['pred_q2'], df_err['pred_q4'], alpha=0.2), 2),
                    'PI width': round(pi_width(df_err['pred_q2'], df_err['pred_q4']), 2)}

        results = pd.DataFrame([my_results])
        results['sMAPE'] = results['sMAPE'].astype(str) + '%'
        results['ACE'] = round((80 - results['UC']), 2).astype(str) + '%'
        results['UC'] = results['UC'].astype(str) + '%'
        results['Penalty'] = results['Winkler score'] - results['PI width']
        results_df = pd.concat([results_df, results], axis=0, ignore_index=True)
        
        return results_df