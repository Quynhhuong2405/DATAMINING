
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

df = pd.read_csv(r'c:\Users\Acer\OneDrive\Desktop\Final_project_DM\DATAMINING\WA_Fn-UseC_-HR-Employee-Attrition.csv')
df['Attrition_bin'] = (df['Attrition'] == 'Yes').astype(int)

print('=== 7. CORRELATION WITH ATTRITION (numeric) ===')
num_cols = ['Age','DailyRate','DistanceFromHome','HourlyRate','JobLevel','MonthlyIncome',
            'MonthlyRate','NumCompaniesWorked','PercentSalaryHike','StockOptionLevel',
            'TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany',
            'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
corr = df[num_cols + ['Attrition_bin']].corr()['Attrition_bin'].drop('Attrition_bin').sort_values()
print(corr.to_string())
print()

print('=== 8. HIGH-RISK PROFILE ===')
high_risk = df[(df['MaritalStatus']=='Single') & (df['OverTime']=='Yes') & (df['JobLevel']==1)]
rate1 = high_risk['Attrition_bin'].mean()*100
print(f'Single + OverTime + JobLevel1: n={len(high_risk)}, Attrition={rate1:.1f}%')

sr_ot = df[(df['JobRole']=='Sales Representative') & (df['OverTime']=='Yes')]
rate2 = sr_ot['Attrition_bin'].mean()*100
print(f'Sales Rep + OverTime: n={len(sr_ot)}, Attrition={rate2:.1f}%')

young_single_ot = df[(df['Age']<30) & (df['MaritalStatus']=='Single') & (df['OverTime']=='Yes')]
rate3 = young_single_ot['Attrition_bin'].mean()*100
print(f'Age<30 + Single + OverTime: n={len(young_single_ot)}, Attrition={rate3:.1f}%')

no_stock_ot_jl1 = df[(df['StockOptionLevel']==0) & (df['OverTime']=='Yes') & (df['JobLevel']==1)]
rate4 = no_stock_ot_jl1['Attrition_bin'].mean()*100
print(f'NoStock + OverTime + JobLevel1: n={len(no_stock_ot_jl1)}, Attrition={rate4:.1f}%')
print()

print('=== 9. PROMOTION STAGNATION ANALYSIS ===')
df['PromotionStagnant'] = df['YearsSinceLastPromotion'] >= 5
stag = df.groupby('PromotionStagnant')['Attrition_bin'].agg(['sum','count','mean'])
stag['Rate_%'] = (stag['mean']*100).round(1)
print('Promotion Stagnant (>=5yr):')
print(stag[['sum','count','Rate_%']].to_string())
print()

stag_lowsat = df[(df['YearsSinceLastPromotion']>=5) & (df['JobSatisfaction']<=2)]
rate5 = stag_lowsat['Attrition_bin'].mean()*100
print(f'Stagnant>=5yr + Low JobSat<=2: n={len(stag_lowsat)}, Attrition={rate5:.1f}%')
print()

print('=== 10. DISTANCE FROM HOME EFFECT ===')
df['DistBin'] = pd.cut(df['DistanceFromHome'], bins=[0,5,10,20,30], labels=['0-5km','5-10km','10-20km','20+km'])
dist = df.groupby('DistBin', observed=False)['Attrition_bin'].agg(['sum','count','mean'])
dist['Rate_%'] = (dist['mean']*100).round(1)
print(dist[['sum','count','Rate_%']].to_string())
print()

print('=== 11. LOYALTY SCORE: YearsWithCurrManager vs Attrition ===')
df['LongManager'] = df['YearsWithCurrManager'] >= 7
lm = df.groupby('LongManager')['Attrition_bin'].agg(['sum','count','mean'])
lm['Rate_%'] = (lm['mean']*100).round(1)
print(lm[['sum','count','Rate_%']].to_string())
print()

print('=== 12. TRAINING EFFECT ===')
train = df.groupby('TrainingTimesLastYear')['Attrition_bin'].agg(['sum','count','mean'])
train['Rate_%'] = (train['mean']*100).round(1)
print(train[['sum','count','Rate_%']].to_string())
print()

print('=== 13. NEW COMPANY HOPPER ANALYSIS ===')
df['JobHopper'] = df['NumCompaniesWorked'] >= 5
jh = df.groupby('JobHopper')['Attrition_bin'].agg(['sum','count','mean'])
jh['Rate_%'] = (jh['mean']*100).round(1)
print('Job Hopper (>=5 companies):')
print(jh[['sum','count','Rate_%']].to_string())
print()

# Job hopper x tenure
jh_new = df[(df['NumCompaniesWorked']>=5) & (df['YearsAtCompany']<=2)]
rate6 = jh_new['Attrition_bin'].mean()*100
print(f'Hopper(>=5co) + Tenure<=2yr: n={len(jh_new)}, Attrition={rate6:.1f}%')
