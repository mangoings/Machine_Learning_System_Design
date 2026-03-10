import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# a =마력, b = 연비
a = np.array([130, 250, 190, 300, 210, 220, 170]).reshape(-1, 1)
b = np.array([16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2])

regr = linear_model.LinearRegression()
regr.fit(a,b)
b_pred = regr.predict(a)

coef = regr.coef_
inter = regr.intercept_
print(coef)
print(inter)
r2 = r2_score(b, b_pred)
print(r2)

new_a = np.array([[270]])
new_b_pred = regr.predict(new_a)
print(new_b_pred)

c = np.array([1900, 2600, 2200, 2900, 2400, 2300, 2100]).reshape(-1,1)

x = np.concatenate((a, c), axis=1)

regr.fit(x, b.reshape(-1,1))
pred_b = regr.predict(x)

new_coef = regr.coef_
new_inter = regr.intercept_
print(new_coef)
print(new_inter)
new_r2 = r2_score(b, pred_b)
print(new_r2)

new_c = np.array([[2500]])
new_x = np.concatenate((new_a, new_c), axis=1)
new_b_pred = regr.predict(new_x)

print(new_b_pred)
