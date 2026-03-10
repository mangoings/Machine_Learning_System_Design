import numpy as np
import pandas as pd

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
data2['월'] = data1['구분'].str.extract('(\d+)월').astype(int)
data3['월'] = data1['구분'].str.extract('(\d+)월').astype(int)

data1.drop(columns=['구분'], inplace=True)
data2.drop(columns=['구분'], inplace=True)
data3.drop(columns=['구분'], inplace=True)

new_grouped_data1 = data1.groupby(['년도', '월'], sort=False).sum()
new_grouped_data2 = data2.groupby(['년도', '월'], sort=False).sum()
new_grouped_data3 = data3.groupby(['년도', '월'], sort=False).sum()

new_data = pd.concat([new_grouped_data1, new_grouped_data2, new_grouped_data3])

print(new_data)

total_death_year = new_data.groupby('년도')[['사망(명)']].mean()
print(total_death_year)

total_death_month = new_data.groupby('월')[['사망(명)']].mean()
print(total_death_month)

total_accident = new_data.loc[2016, '사고(건)'].sum()
total_death = new_data.loc[2016, '사망(명)'].sum()
rate = (total_death / total_accident) *100

print("전체사고(건): {:d}, 사망자(명): {:d}, 사고대비사망율: {:.2f}%".format(total_accident, total_death, rate))

import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

data_2014_death = new_data.loc[2014, '사망(명)']
data_2014_hurt = new_data.loc[2014, '부상(명)']

month = np.arange(1,13,1)
width = 0.35

plt.figure()
plt.bar(month, data_2014_death, width, color='b', label='사망(명)')
plt.bar([x + width for x in month], data_2014_hurt, width, color='orange', label='부상(명)')
plt.legend()
plt.xlabel("월")
plt.xticks([x + width / 2 for x in month], month)

data_2015 = new_data.loc[2015, '사망(명)']
data_2016 = new_data.loc[2016, '사망(명)']

data_difference = data_2016 - data_2015

large_data = data_difference.nlargest(2).index
largest_months = [str(month) + '월' for month in large_data]

print("2015년 대비 2016년 사망이 가장 많이 증가한 달: {:s}, {:s}".format(largest_months[0], largest_months[1]))
