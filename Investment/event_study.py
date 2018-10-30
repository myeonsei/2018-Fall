import numpy as np; import pandas as pd; from matplotlib import pyplot as plt
from sklearn import linear_model; import statsmodels.api as sm
import pymysql

db = pymysql.connect(host = "localhost", user = "root", passwd = "apdvkd68", db = "kospi",  charset = "utf8")
cursor = db.cursor()
code = pd.read_csv(r'C:\Users\myeon\Documents\카카오톡 받은 파일\stockCodeInfo.csv', engine='python', encoding='utf-8')
kospi = pd.read_csv(r'C:\Users\myeon\Documents\카카오톡 받은 파일\kospi.csv', engine='python', encoding='utf-8', index_col=0).sort_values(by=['년/월/일'])
kospi['종가'] = kospi['종가'].apply(lambda x: float(x.replace(',','')))

def make_rr(price):
    a = np.array(price[:-1]); b = np.array(price[1:])
    return (b-a)/a

kospi['rr'] = [None]+(list(make_rr(kospi['종가'])))
codes = code[(code['marketKind']=='거래소         ')&(code['sectionKind']=='주권                                                        ')]['stockCode'] # 일부만 추리고 싶으면 이걸루..
# 코스피에서 거래되는 주식만 포함-

def train(cursor, start, end, codes, kospi): # 각각의 coef 구하기. 날짜는 8자리 숫자로 입력
    print(len(codes))
    regr = linear_model.LinearRegression()
    coef = {}
    new_codes = []
    market_idx = np.array(kospi[(kospi['년/월/일']>=start) & (kospi['년/월/일']<=end)]['rr']).reshape(-1,1)
 
    length = len(market_idx)
    cnt = 0
    
    for i in codes: 
        #print(i)
        cursor.execute('select logDate, priceClose from stock where stockCode = \'{0}\' and logDate >= {1} and logDate <= {2} order by logDate asc'.format(i, start, end))
        tmp = np.array(cursor.fetchall())
        if tmp.shape[0] != length: continue # 분석 기간 내 데이터 다 있는 것들만.
            
        tmp = np.append(tmp.T, [[None] + (list(make_rr(tmp[:,1])))], axis=0).T
        regr.fit(market_idx[1:], tmp[1:,2])
        coef[i] = [regr.intercept_, regr.coef_]; new_codes.append(i)
        cnt = cnt + 1
        if cnt % 10 == 0: print(cnt)
    print('끝')
    return new_codes, coef # intercept, slope를 dictionary 형태로 반환
        
def return_abre(cursor, start, end, new_codes, kospi, coef): # start는 하루 더 앞으로 할 것. 왜냐면 첫 날은 생략됨.
    abre = {}
    market_idx = kospi[(kospi['년/월/일']>=start) & (kospi['년/월/일']<=end)]['rr']
    length = len(market_idx)
    cnt = 0
    
    for i in new_codes:
        cursor.execute('select logDate, priceClose from stock where stockCode = \'{0}\' and logDate >= {1} and logDate <= {2} order by logDate asc'.format(i, start, end))
        tmp = np.array(cursor.fetchall())
        if tmp.shape[0] != length: continue
            
        tmp = np.append(tmp.T, [[None] + (list(make_rr(tmp[:,1])))], axis=0).T
        ab = tmp[1:,2] - coef[i][0] - coef[i][1] * market_idx[1:]
        abre[i] = list(ab)
        cnt = cnt + 1
        if cnt % 10 == 0: print(cnt)
    print('끝')
    return abre

def return_car(abre): # cumulative sum 반환
    car = {}
    for i in abre.keys():
        car[i] = np.cumsum(abre[i])
    return car

chaebeol = ['A003480','A000120','A003490','A003550','A000400','A002270','A005300','A000150','A001040','A000270','A004990','A009830','A005380','A002320','A005930','A000370','A000810','A001120','A004000','A000880','A001250','A001740','A011760','A006400','A009150','A006360','A000720','A004170','A001510','A003530','A004020','A012450','A016360','A003555','A011150','A003535','A001045','A005440','A001450','A001515','A004565','A012330','A005385','A000725','A005305','A009155','A017670','A009835','A006125','A006405','A001745','A011155','A000885','A003495','A000815','A001500','A008770','A011170','A008775','A010140','A010145','A011200','A027390','A000660','A011790','A018670','A030000','A005387','A005389','A009540','A000157','A034020','A042670','A051900','A051905','A051910','A051915','A066570','A066575','A034300','A034220','A078930','A078935','A079160','A079430','A086280','A023530','A035510','A032350','A089470','A029780','A096770','A096775','A097230','A097950','A097955','A032640','A011070','A108670','A108675','A034730','A088350','A031440','A032830','A057050','A126560','A011210','A071840','A031430','A007070','A037560','A180640','A18064K','A018260','A028260','A210980','A03473K','A039570','A227840','A00088K','A207940','A241560','A267250','A267260','A267270','A00499K','A280360','A285130','A28513K','A294870','A286940']

def return_car2(abre, chaebeol): # cumulative sum 반환
    car = {}
    for i in abre.keys():
        if i in chaebeol:
            car[i] = np.cumsum(abre[i])
    return car

def avg(return_dict): # idx should be a list containing targetted stock codes
    tmp = []
    for i in return_dict.keys():
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
    
####

new_codes, coef = train(cursor, 20130615, 20140614, codes, kospi) # 예상된 금리 인하: 2014 08 14
abre = return_abre(cursor, 20140731, 20140910, new_codes, kospi, coef)
print(abre)
car = return_car(abre)
avg = avg(car)
plt.plot(avg)
plt.show()
result = stat_test(avg)
print(result.summary())

new_codes2, coef2 = train(cursor, 20150410, 20160409, codes, kospi) # 예상치 못한 금리 인하: 2016 06 09
abre2 = return_abre(cursor, 20160526, 20160707, new_codes2, kospi, coef2)
car2 = return_car(abre2)
avg2 = avg(car2)
plt.plot(avg2)
plt.show()
result2 = stat_test(avg2)
print(result2.summary())

# 예상된 금리 인하 - 재벌
car3 = return_car2(abre, chaebeol)
avg3 = avg(car3)
plt.plot(avg3)
plt.show()
result3 = stat_test(avg3)
print(result3.summary())

# 예상치 못한 금리 인하 - 재벌
car4 = return_car2(abre2, chaebeol)
avg4 = avg(car4)
plt.plot(avg4)
plt.show()
result4 = stat_test(avg4)
print(result3.summary())

# 간단한 실험용 코드
new_codes, coef = train(cursor, 20140101, 20150101, codes, kospi)
abre = return_abre(cursor, 20150301, 20150401, new_codes, kospi, coef)
car = return_car(abre)
avg = avg(car)
plt.plot(avg)
plt.show()
result = stat_test(avg)
print(result.summary())
