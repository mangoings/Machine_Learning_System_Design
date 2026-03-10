import numpy as np
import pandas as pd

data = {
        'year' : [2016, 2017, 2018],
        'car' : ['그랜저', '그랜저', '소나타'],
        'name' : ['홍길동', '고길동', '김둘리'],
        'number' : ['123하4567', '123허4567', '123호4567']
        }

DF = pd.DataFrame(data)

print(DF)

new_data = {
    'year' : 2017,
    'car' : '테슬라',
    'name' : '일론',
    'number' : '987하6543'
    }

DF.loc[3] = new_data
print(DF)

print(DF[['year', 'car', 'number']])

print(DF[DF['year'] < 2018])
