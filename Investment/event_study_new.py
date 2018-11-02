import numpy as np; import pandas as pd; from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model; import statsmodels.api as sm
import pymysql
import pickle

db = pymysql.connect(host = "localhost", user = "root", passwd = "apdvkd68", db = "kospi",  charset = "utf8")
cursor = db.cursor()
code = pd.read_csv(r'C:\Users\myeon\Documents\카카오톡 받은 파일\stockCodeInfo.csv', engine='python', encoding='utf-8')
kospi = pd.read_csv(r'C:\Users\myeon\Documents\카카오톡 받은 파일\kospi.csv', engine='python', encoding='utf-8', index_col=0).sort_values(by=['년/월/일'])
kospi['종가'] = kospi['종가'].apply(lambda x: float(x.replace(',','')))

dates = [(20140814, 20130815, 20140800), # 조정 시점/start/end
(20141015, 20140828, 20141001),
(20150312, 20141029, 20150227),
(20150611, 20150326, 20150528),
(20160609, 20150625, 20160526),
(20171130, 20170130, 20171116),]

def make_rr(price):
    a = np.array(price[:-1]); b = np.array(price[1:])
    return (b-a)/a

kospi['rr'] = [None]+(list(make_rr(kospi['종가'])))
kospi = kospi.reset_index(); del kospi['index']
codes = code[(code['marketKind']=='거래소         ')&(code['sectionKind']=='주권                                                        ')]['stockCode'] # 일부만 추리고 싶으면 이걸루..
# 코스피에서 거래되는 주식만 포함-

print(len(codes)) ####
regr = linear_model.LinearRegression()
coef = defaultdict(list); abre = defaultdict(list) # 리스트 안에 여러 튜플 있는 걸루..
new_codes = []
market_idx = {}; mk_idx_for_abre = {}; ks = {}

for date, start, end in dates:
    market_idx[date] = np.array(kospi[(kospi['년/월/일']>=start) & (kospi['년/월/일']<=end)]['rr']).reshape(-1,1)
    k = kospi.index[kospi['년/월/일'] == date][0]; ks[date] = k
    #print(k)
    mk_idx_for_abre[date] = np.array(kospi.iloc[k-10:k+20,2])
    
length = np.sum(kospi['년/월/일']<=20180928)
cnt = 0

for i in codes: 
    cursor.execute('select logDate, priceClose from stock where stockCode = \'{0}\' order by logDate asc'.format(i))
    tmp = np.array(cursor.fetchall())
    if tmp.shape[0] != length: continue # 분석 기간 내 데이터 다 있는 것들만.
    print(i, cnt)
    
    tmp = np.append(tmp.T, [[None] + (list(make_rr(tmp[:,1])))], axis=0).T
    tmp = pd.DataFrame(tmp)  # 0-날짜, 1-가격, 2-RR. # 여기까지는 문제 없음.
    
    for date, start, end in dates:
        regr.fit(market_idx[date], tmp[(tmp[0]>=start) & (tmp[0]<=end)][2])
        coef[i].append((regr.intercept_, regr.coef_[0]))
        
        ab = tmp.iloc[ks[date]-10:ks[date]+20,2] - regr.intercept_ - regr.coef_[0] * mk_idx_for_abre[date]
        abre[i].append(list(ab))     
    
    new_codes.append(i)
    cnt = cnt + 1
    if cnt % 10 == 0: print(cnt)
        
db.close()

with open('abre.pickle', 'wb') as handle:
    pickle.dump(abre, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('coef.pickle', 'wb') as handle:
    pickle.dump(coef, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def return_car(abre): # cumulative sum 반환
    car = defaultdict(list)
    for i in abre.keys():
        for j in abre[i]:
            car[i].append(np.cumsum(j))
    return car

def average(car): # idx should be a list containing targetted stock codes
    #print(type(return_dict))
    avgs = []
    
    for i in range(6):
        akkk=0
        for j in car.keys():
            if akkk==0:
                tmp = car[j][i].reshape(1,-1)
                akkk += 1
            else:
                tmp = np.append(tmp, car[j][i].reshape(1,-1), axis=0)
        avgs.append(np.mean(tmp, axis=0))
    return avgs

car = return_car(abre)
caar = average(car)

# chaebeol = ['A003480','A000120','A003490','A003550','A000400','A002270','A005300','A000150','A001040','A000270','A004990','A009830','A005380','A002320','A005930','A000370','A000810','A001120','A004000','A000880','A001250','A001740','A011760','A006400','A009150','A006360','A000720','A004170','A001510','A003530','A004020','A012450','A016360','A003555','A011150','A003535','A001045','A005440','A001450','A001515','A004565','A012330','A005385','A000725','A005305','A009155','A017670','A009835','A006125','A006405','A001745','A011155','A000885','A003495','A000815','A001500','A008770','A011170','A008775','A010140','A010145','A011200','A027390','A000660','A011790','A018670','A030000','A005387','A005389','A009540','A000157','A034020','A042670','A051900','A051905','A051910','A051915','A066570','A066575','A034300','A034220','A078930','A078935','A079160','A079430','A086280','A023530','A035510','A032350','A089470','A029780','A096770','A096775','A097230','A097950','A097955','A032640','A011070','A108670','A108675','A034730','A088350','A031440','A032830','A057050','A126560','A011210','A071840','A031430','A007070','A037560','A180640','A18064K','A018260','A028260','A210980','A03473K','A039570','A227840','A00088K','A207940','A241560','A267250','A267260','A267270','A00499K','A280360','A285130','A28513K','A294870','A286940']
# chaebeol_car = {}

# for i in chaebeol:
#     if i in car.keys():
#         chaebeol_car[i] = car[i]
    
# chaebeol_caar = average(chaebeol_car)

# with open('caar.pickle', 'wb') as handle:
#     pickle.dump(caar, handle, protocol=pickle.HIGHEST_PROTOCOL)

def stat_test(avg, num_forward = 10):
    x = np.arange(len(avg))
    d = np.append(np.zeros(num_forward), np.ones(len(avg)-num_forward))
    d_x = d * x
    X = np.append(np.append(x.reshape(-1,1), d.reshape(-1,1), axis=1), d_x.reshape(-1,1), axis=1)
    X = sm.add_constant(X)
    regr = sm.OLS(avg, X)
    result = regr.fit()

    return result
    
def stat_test2(avg, num_forward = 10): # 추세 
    d = np.append(np.zeros(num_forward), np.ones(len(avg)-num_forward))
    d = sm.add_constant(d)
    regr = sm.OLS(avg, d)
    result = regr.fit()

    return result    
    
chaebeol = ['A003480','A000120','A003490','A003550','A000400','A002270','A005300','A000150','A001040','A000270','A004990','A009830','A005380','A002320','A005930','A000370','A000810','A001120','A004000','A000880','A001250','A001740','A011760','A006400','A009150','A006360','A000720','A004170','A001510','A003530','A004020','A012450','A016360','A003555','A011150','A003535','A001045','A005440','A001450','A001515','A004565','A012330','A005385','A000725','A005305','A009155','A017670','A009835','A006125','A006405','A001745','A011155','A000885','A003495','A000815','A001500','A008770','A011170','A008775','A010140','A010145','A011200','A027390','A000660','A011790','A018670','A030000','A005387','A005389','A009540','A000157','A034020','A042670','A051900','A051905','A051910','A051915','A066570','A066575','A034300','A034220','A078930','A078935','A079160','A079430','A086280','A023530','A035510','A032350','A089470','A029780','A096770','A096775','A097230','A097950','A097955','A032640','A011070','A108670','A108675','A034730','A088350','A031440','A032830','A057050','A126560','A011210','A071840','A031430','A007070','A037560','A180640','A18064K','A018260','A028260','A210980','A03473K','A039570','A227840','A00088K','A207940','A241560','A267250','A267260','A267270','A00499K','A280360','A285130','A28513K','A294870','A286940']
chaebeol_car = {}

for i in chaebeol:
    if i in car.keys():
        chaebeol_car[i] = car[i]
    
chaebeol_caar = average(chaebeol_car)

# 결과 실행 코드 예시

plt.plot(caar[0])
plt.axvline(x=10.5, c='r')
plt.show()

result = stat_test(caar[0])
print(result.summary())
