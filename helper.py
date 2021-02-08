import pandas as pd
import numpy as np


def expected_one_trade_return(lag, pairs, data):
    """
    :param lag: int of how many lags
    :param pairs: list ['n','c'] or ['c', 'c']
    :param data: the data file must have return, sign, and event in its column
    :return: float scalar
    """
    data['temp_sign'] = data['Sign'].shift(lag)
    data['temp_event'] = data['Event'].shift(lag)
    sub_dta = data[(data['temp_event'] == pairs[0]) & (data['Event'] == pairs[1])]
    if sub_dta.shape[0] != 0:
        res = (sub_dta['Return'] * sub_dta['temp_sign']).mean()
        return res
    else:
        return 0


def cross_correlation(lag, pairs, data):
    """
    :param lag: lag: int of how many lags
    :param pairs: list ['n','c'] or ['c', 'c'] or ['c','n'] or ['n', 'n']
    :param data: the data file must have return, sign, and event in its column
    :return: float scalar
    """
    data['temp_sign'] = data['Sign'].shift(lag)
    data['temp_event'] = data['Event'].shift(lag)
    sub_dta = data[(data['temp_event'] == pairs[1]) & (data['Event'] == pairs[0])]
    if sub_dta.shape[0] != 0:
        res = (sub_dta['Sign'] * sub_dta['temp_sign']).mean()
        return res
    else:
        return 0


def expected_return_vector(lag, data):
    """
    :param lag: int maximum lag for the data
    :param data: the data file to calculate expected return
    :return: a column vector 2l + 2 length
    """
    res_nc = [0]
    res_cc = [0]
    for i in range(1, lag):
        s_nc = expected_one_trade_return(i, ['n', 'c'], data)
        s_cc = expected_one_trade_return(i, ['c', 'c'], data)
        res_nc.append(s_nc)
        res_cc.append(s_cc)
    res_nc.extend(res_cc)

    return np.array(res_nc)


def cross_correlation_matrix(lag, data):
    """
    :param lag: int maximum lag for the data
    :param data: the data file to calculate expected return
    :return: block matrix from top left: C_nn, C_cn, C_nc, C_cc
    """
    res_nn, res_nc, res_cn, res_cc = [1], [1], [1], [1]

    matrix_nn = res_nn + [0] * (lag-1)
    matrix_nc = res_nc + [0] * (lag - 1)
    matrix_cn = res_cn + [0] * (lag - 1)
    matrix_cc = res_cc + [0] * (lag - 1)

    for i in range(1, lag):
        c_nn = cross_correlation(i, ['n', 'n'], data)
        c_nc = cross_correlation(i, ['n', 'c'], data)
        c_cn = cross_correlation(i, ['c', 'n'], data)
        c_cc = cross_correlation(i, ['c', 'c'], data)

        res_nn.append(c_nn)
        res_nc.append(c_nc)
        res_cn.append(c_cn)
        res_cc.append(c_cc)

        temp_nn = res_nn[::-1]
        temp_nc = res_nc[::-1]
        temp_cn = res_cn[::-1]
        temp_cc = res_cc[::-1]

        matrix_nn = np.vstack((matrix_nn, temp_nn + [0] * (lag - 1 - i)))
        matrix_nc = np.vstack((matrix_nc, temp_nc + [0] * (lag - 1 - i)))
        matrix_cn = np.vstack((matrix_cn, temp_cn + [0] * (lag - 1 - i)))
        matrix_cc = np.vstack((matrix_cc, temp_cc + [0] * (lag - 1 - i)))

    idx = np.tril_indices(lag)
    matrix_nn.T[idx] = matrix_nn[idx]
    matrix_nc.T[idx] = matrix_nc[idx]
    matrix_cn.T[idx] = matrix_cn[idx]
    matrix_cc.T[idx] = matrix_cc[idx]

    return np.block([[matrix_nn, matrix_cn],
                     [matrix_nc, matrix_cc]])


def est_kernel(lag, data):
    S = expected_return_vector(lag, data)
    C = cross_correlation_matrix(lag, data)
    assert S.shape[0] == C.shape[0]
    K = np.linalg.pinv(C) @ S

    return K


def propagator(K, data):
    """
    :param K: propagator kernel estimated by C and S
    :param data: has to be the same length as K/2
    :return: return of midprice to the next time
    """
    if data.iloc[-1]['Event'] != 'c':
        return 0
    else:
        data['k_nc'] = K[:len(K)//2][::-1]
        data['k_cc'] = K[len(K)//2:][::-1]
        temp = data[data.Event == 'c']
        res1 = (temp['Sign'] * temp['k_cc']).sum()
        temp = data[data.Event == 'n']
        res2 = (temp['Sign'] * temp['k_nc']).sum()
        return res1 + res2


def prop_predict(lag, data):
    """
    :param lag: integer how many lags when building propagator kernel
    :param data: entire day dataset
    :return: list of prediction
    """
    K = est_kernel(lag, data)
    pred = []
    for i in range(lag, data.shape[0]):
        sub_dta = data.iloc[i-lag:i]
        pred.append(propagator(K, sub_dta))

    return np.array(pred)



