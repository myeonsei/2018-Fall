import numpy as np; import pandas as pd
from sklearn import linear_model
import pymysql

db = pymysql.connect(host = "localhost", user = "root", passwd = "###", db = "kospi",  charset = "utf8")
cursor = db.cursor()
codes = pd.read_csv(r'C:\Users\myeon\Documents\카카오톡 받은 파일\stockCodeInfo.csv', engine='python', encoding='utf-8')
kospi = pd.read_csv(r'C:\Users\myeon\Documents\카카오톡 받은 파일\kospi.csv', engine='python', encoding='utf-8', index_col=0)
kospi['종가'] = kospi['종가'].apply(lambda x: float(x.replace(',','')))

def make_rr(price):
    a = np.array(price[:-1]); b = np.array(price[1:])
    return (b-a)/a

kospi['rr'] = [None]+(list(make_rr(kospi['종가'])))

def train(cursor, start, end, codes, kospi): # 각각의 coef 구하기. 날짜는 8자리 숫자로 입력
    regr = linear_model.LinearRegression()
    coef = {}
    for i in range(codes['stockCode']):
        cursor.execute('select logDate, priceClose from stock where stockCode = \'{0}\' and logDate >= {1} and logDate <= {2}'.format(i, start, end))
        tmp = np.array(cursor.fetchall())
        tmp = np.append(tmp.T, [[None] + (list(make_rr(tmp[:,1])))], axis=0).T
        regr.fit(tmp[1:,2].reshape(-1, 1), kospi[(kospi['년/월/일']>=tmp[1,0]) & (kospi['년/월/일']<=tmp[-1,0])]['rr'])
        coef[i] = [regr.intercept_, regr.coef_]
    return coef # intercept, slope를 dictionary 형태로 반환
        
def return_abre(cursor, start, end, codes, kospi, coef): # start는 하루 더 앞으로 할 것. 왜냐면 첫 날은 생략됨.
    abre = {}
    for i in range(codes['stockCode']):
        cursor.execute('select logDate, priceClose from stock where stockCode = \'{0}\' and logDate >= {1} and logDate <= {2}'.format(i, start, end))
        tmp = np.array(cursor.fetchall())
        tmp = np.append(tmp.T, [[None] + (list(make_rr(tmp[:,1])))], axis=0).T
        ab = tmp[1:,2] - coef[i][0] - coef[i][1] * kospi[(kospi['년/월/일']>=tmp[1,0]) & (kospi['년/월/일']<=tmp[-1,0])]
        abre[i] = list(ab)
    return abre
