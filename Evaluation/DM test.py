import numpy as np
import pandas as pd
import statsmodels


merged_df = pd.read_csv() # import dataframe
real_price = merged_df['actual']

# list of models
models = ['LSTM', 'DNN1', 'DNN2', 'DNN3', 'DNN_ens']

# list of model labels
model_labeles = ['LSTM', 'DNN-Ordinal', 'DNN-Onehot', 'DNN-Emb', 'DNN-Ens']


def Diebold(p_real, pred_1, pred_2, loss='pinball', type=1): #Get the forecast errors for each model

    from scipy.stats import norm
    from sklearn.metrics import mean_absolute_error
    def pinball_loss(y_true, y_pred, tau):
      error = y_true - y_pred
      loss = np.maximum(tau * error, (tau - 1) * error)
      return loss

    def winkler_score(y_true, y_lower, y_upper, alpha=0.2):
      delta = y_upper - y_lower

      score = np.where((y_true >= y_lower) & (y_true <= y_upper), delta, 0)
      score += np.where(y_true < y_lower, delta + (2 / alpha) * (y_lower - y_true), 0)
      score += np.where(y_true > y_upper, delta + (2 / alpha) * (y_true - y_upper), 0)
      return score

    def relative_mae(y_true, y_pred):
      rmae = np.abs(y_true - y_pred)/mean_absolute_error(merged_df['actual'], merged_df['Naive_q2'])
      return rmae


    # real
    p_real = p_real.values.reshape(-1, 24)


    # model 1
    p_pred_1_q1 = pred_1.iloc[:, 0].values.reshape(-1, 24)
    p_pred_1_q2 = pred_1.iloc[:, 1].values.reshape(-1, 24)
    p_pred_1_q3 = pred_1.iloc[:, 2].values.reshape(-1, 24)

    # model 2
    p_pred_2_q1 = pred_2.iloc[:, 0].values.reshape(-1, 24)
    p_pred_2_q2 = pred_2.iloc[:, 1].values.reshape(-1, 24)
    p_pred_2_q3 = pred_2.iloc[:, 2].values.reshape(-1, 24)

    if loss == 'pinball':
      # Computing pinball losses of each forecast
      errors_pred_1 = pinball_loss(p_real, p_pred_1_q3, tau=0.9)
      errors_pred_2 = pinball_loss(p_real, p_pred_2_q3, tau=0.9)

    elif loss =='rmae':
      errors_pred_1 = relative_mae(p_real, p_pred_1_q2)
      errors_pred_2 = relative_mae(p_real, p_pred_2_q2)

    else:
      # Computing winkler scores of each forecast
      errors_pred_1 = winkler_score(p_real, p_pred_1_q1, p_pred_1_q3, alpha=0.2)
      errors_pred_2 = winkler_score(p_real, p_pred_2_q1, p_pred_2_q3, alpha=0.2)


    # Computing the loss differential series for the multivariate test
    if type == 1:
        diff = np.mean(np.abs(errors_pred_1), axis=1) - np.mean(np.abs(errors_pred_2), axis=1)
    else:
        diff = np.mean(errors_pred_1**2, axis=1) - np.mean(errors_pred_2**2, axis=1)

    # Computing the loss differential size
    N = diff.size

    # Computing the test statistic
    mean_d = np.mean(diff)
    var_d = np.var(diff, ddof=0)
    DM_stat = mean_d / np.sqrt((1 / N) * var_d)

    p_value = 1 - norm.cdf(DM_stat)

    return p_value


import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_DM_test(real_price, models, loss, norm=1, savefig='no'):

    # Computing the multivariate DM test for each forecast pair
    p_values = pd.DataFrame(index=models, columns=models)

    for model1 in models:
        for model2 in models:
            # For the diagonal elemnts representing comparing the same model we directly set a
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = Diebold(p_real=real_price,
                                                  pred_1=merged_df[[f'{model1}_q1', f'{model1}_q2', f'{model1}_q3']],
                                                  pred_2=merged_df[[f'{model2}_q1', f'{model2}_q2', f'{model2}_q3']],
                                                  loss=loss,
                                                  type=1)
    # Defining color map
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1),
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

    # Generating figure
    plt.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    plt.xticks(range(len(models)), model_labeles, rotation=90., size=12)
    plt.yticks(range(len(models)), model_labeles, size=12)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    plt.colorbar()
    #plt.title('rMAE DM test')
    plt.tight_layout()

    # savefig ?
    if savefig == 'yes':
      plt.savefig('/content/drive/MyDrive/DM_test_Price_{}.png'.format(loss))
      plt.show()
    else:
      plt.show()

