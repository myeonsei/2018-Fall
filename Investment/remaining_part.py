import numpy as np; import pandas as pd; from matplotlib import pyplot as plt
from sklearn import linear_model; import statsmodels.api as sm
import pymysql

def return_car(abre): # cumulative sum 반환
    car = {}
    for i in abre.keys():
        car[i] = np.cumsum(abre[i])
    return car

def avg(return_dict, idx): # idx should be a list containing targetted stock codes
    tmp = []
    for i in idx:
        tmp.append(return_dict[i])
    tmp = np.array(tmp)
    return np.mean(tmp, axis=0)

def stat_test(avg, num_forward = 10):
    x = np.arange(len(avg))
    d = np.append(np.zeros(num_forward), np.ones(len(avg)-num_forward))
    d_x = d * x
    X = np.append(np.append(x.reshape(-1,1), d.reshape(-1,1), axis=1), d_x.reshape(-1,1), axis=1)
    X = sm.add_constant(X)

    regr = sm.OLS(avg, X)
    result = regr.fit()
    return result

# new_codes, coef = train(cursor, 20140101, 20150101, codes, kospi)
# abre = return_abre(cursor, 20150301, 20150501, new_codes, kospi, coef)
# car = return_car(abre)
# avg = avg(car, ['A000020', 'A000030', 'A000040'])
# plt.plot(avg)
# plt.show()
# result = stat_test(avg)
# print(result.summary())
