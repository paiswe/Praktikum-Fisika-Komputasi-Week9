from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

x = [[0], [5], [10], [15], [20], [25], [30], [35], [40], [45]]
y = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]

predict = np.array ([[12]])
poly = PolynomialFeatures(degree=2)
x_ = poly.fit_transform(x)
predict_ = poly.fit_transform(predict)
regr = linear_model.LinearRegression()
regr.fit (x_,y)

print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict(predict_))
