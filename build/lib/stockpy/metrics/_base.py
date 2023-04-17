from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape

def evaluate(y_test, y_pred, show=False):
    mse = mean_squared_error(y_test, y_pred, squared=True)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    # dtw_distance = fastdtw(y_test, y_pred, dist=euclidean)[0] 
    if show:
        print('Model Performance')
        print("Mean squared error = {:0.3f}".format(mse))
        print("Root mean squared error = {:0.3f}".format(rmse))
        print('Mean absolute percentage error = {:0.3f}%.'.format(mape))
        # print('DTW distance = {:0.3f} '.format(dtw_distance))
    else:
        return mse, rmse, mape
    
def mean_squared_error(y_test, y_pred, squared):
    return mse(y_test, y_pred, squared=squared)

def mean_absolute_percentage_error(y_test, y_pred):
    return mape(y_test, y_pred) * 100