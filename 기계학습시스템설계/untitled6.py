import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "C:\\Users\\kui45\\"
data1 = pd.read_csv(path + '2014.csv', encoding='CP949')
data2 = pd.read_csv(path + '2015.csv', encoding='CP949')
data3 = pd.read_csv(path + '2016.csv', encoding='CP949')

grouped_data1 = data1.groupby('구분', sort=False).sum()
grouped_data2 = data2.groupby('구분', sort=False).sum()
grouped_data3 = data3.groupby('구분', sort=False).sum()

data = pd.concat([grouped_data1, grouped_data2, grouped_data3])

print(data)

data1['년도'] = 2014
data2['년도'] = 2015
data3['년도'] = 2016

data1['월'] = data1['구분'].str.extract('(\d+)월').astype(int)
data2['월'] = data2['구분'].str.extract('(\d+)월').astype(int)
data3['월'] = data3['구분'].str.extract('(\d+)월').astype(int)

data1.drop(columns=['구분'], inplace=True)
data2.drop(columns=['구분'], inplace=True)
data3.drop(columns=['구분'], inplace=True)

grouped_data1 = data1.groupby(['년도', '월'], sort=False).sum()
grouped_data2 = data2.groupby(['년도', '월'], sort=False).sum()
grouped_data3 = data3.groupby(['년도', '월'], sort=False).sum()

data = pd.concat([grouped_data1, grouped_data2, grouped_data3])

print(data)

total_death_data = data.groupby('년도')[['사망(명)']].mean()
print(total_death_data)

month_death_data = data.groupby('월')[['사망(명)']].mean()
print(month_death_data)

total_accident = data.loc[2016, '사고(건)'].sum()
total_death = data.loc[2016, '사망(명)'].sum()
total_rate = (total_death / total_accident) * 100
print(total_accident)
print(total_death)
print(total_rate)

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

data_2014_death = data.loc[2014, '사망(명)']

data_2014_hurt = data.loc[2014, '부상(명)']

month = range(1, 13)
width = 0.35

plt.figure()
plt.bar(month, data_2014_death, width, color='b', label='사망(명)')
plt.bar([x + width for x in month], data_2014_hurt, width, color='orange', label='부상(명)')
plt.xticks([x + width / 2 for x in month], month)
plt.legend()
plt.xlabel("월")