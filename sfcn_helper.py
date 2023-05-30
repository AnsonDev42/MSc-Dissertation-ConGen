from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd


def get_bin_range_step(age):
    """
    "The output layer contains 40 digits that represent the predicted probability that the subject’s age falls into
    a one-year age interval between 42 to 82 (for UK Biobank) or a two-year age interval between 14 to 94 (for PAC 2019)."


    :param age: age of the subject
    :return: the most suitable range of the bin, and the step size of the bin
    """
    UBB_margin = 2
    PAC_margin = 3
    if 42 + UBB_margin <= age <= 82 - UBB_margin:  # UK Biobank has default margin of 3,i.e., 44-80 real age
        return [42, 82], 1
    elif 14 + PAC_margin <= age <= 94 - PAC_margin:  # PAC 2019 has default margin of 3,i.e., 17-90 real age
        return [14, 94], 2
    else:
        print("Age out of range")
        return [14, 94], 2


def fit_and_score(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return mean_squared_error(y_test, y_pred)


def correct_bias(x_pred, y_actual, x_test):
    # Reshape to 2D array as required by the sklearn model
    x_pred = x_pred.reshape(-1, 1)
    y_actual = y_actual.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    model = LinearRegression()

    # mse = fit_and_score(model, x_pred, y_actual, x_test, y_test)
    # print(f'Train MSE: {mse}')
    model.fit(x_pred, y_actual)
    # y_pred = model.predict(x_test)
    a, b = model.coef_[0], model.intercept_
    x_corrected = (x_test - b) / a  # Apply bias correction

    return x_corrected


def linear_bias_correction():
    # read the brain_age_info.csv and get the age and predicted age as numpy array using pandas
    df = pd.read_csv('brain_age_info.csv')
    x_pred = df['brain_age'].to_numpy()
    y_actual = df['age'].to_numpy()
    status = df['status'].to_numpy()  # Assuming 'status' indicates whether it's a HC sample or a patient sample

    # Get data for the HC group and the patient group
    # x_pred_HC = x_pred[status == 'HC']
    # y_actual_HC = y_actual[status == 'HC']
    # x_pred_patient = x_pred[status == 'patient']
    # y_actual_patient = y_actual[status == 'patient']
    x_pred_HC = x_pred[:10]
    y_actual_HC = y_actual[:10]
    x_pred_test = x_pred[10:]
    y_actual_test = y_actual[10:]

    # Perform bias correction
    x_corrected_test = correct_bias(x_pred_HC, y_actual_HC, x_pred_test)
    # x_corrected_patient = correct_bias(x_pred_HC, y_actual_HC,
    #                                    x_pred_patient)  # We fit on HC data and correct the patient data
    mean_squared_error(y_actual_hc, x_corrected_test)
    # Print out corrected results
    print(f'Corrected brain ages for HC group: {x_corrected_test.flatten()}')
    print(f'Actual brain ages for HC group: {y_actual_test.flatten()}')
    # print(f'Corrected brain ages for patient group: {x_corrected_patient.flatten()}')


from sklearn.linear_model import LinearRegression
import numpy as np


def bias_correction(y: np.ndarray, y_pred: np.ndarray):
    """
        :param y: chronological age (ground truth)
        :param y_pred: brain age before bias correction (predictions)
        :return: bias-corrected brain age
    """
    # Smith2019 bias correction
    linear_fit = LinearRegression(fit_intercept=True).fit(
        X=y.reshape(-1, 1),
        y=y_pred
    )
    intercept, slope = linear_fit.intercept_, linear_fit.coef_[0]
    y_pred_unbiased = (y_pred - intercept) / (slope + np.finfo(np.float32).eps)  # avoid division by 0
    print(f'Intercept: {intercept}, slope: {slope}')
    return y_pred_unbiased


if __name__ == '__main__':
    # linear_bias_correction()
    df = pd.read_csv('brain_age_info.csv')
    y_pred = df['brain_age'].to_numpy()
    y_actual = df['age'].to_numpy()
    y_pred_unbiased = bias_correction(y_actual, y_pred)
    print(y_pred_unbiased)
