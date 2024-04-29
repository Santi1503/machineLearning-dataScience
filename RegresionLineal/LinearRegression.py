import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

calculoEquipos = 15


x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.randn(100,1)

print("La longitud del conjunto es: ", len(x))


data = {'equiposAfectados': x.flatten(), 'costeIncidente': y.flatten()}
df = pd.DataFrame(data)
df.head(10)

df['equiposAfectados'] = df['equiposAfectados'] * 1000
df['equiposAfectados'] = df['equiposAfectados'].astype(int)
df['costeIncidente'] = df['costeIncidente'] * 10000
df['equiposAfectados'] = df['equiposAfectados'].astype(int)
df.head(10)

linReg = LinearRegression()
linReg.fit(df['equiposAfectados'].values.reshape(-1,1), df['costeIncidente'].values)

print("El coeficiente de la regresión lineal es: ", linReg.coef_)
print("El intercepto de la regresión lineal es: ", linReg.intercept_)

xMinMax = np.array([[df['equiposAfectados'].min()], [df['equiposAfectados'].max()]])
yTrainPred = linReg.predict(xMinMax)

xNew = np.array([[calculoEquipos]])
costeIncidente = linReg.predict(xNew)
print("El coste del incidente sería:", int(costeIncidente[0]), "€")


plt.plot(df['equiposAfectados'], df['costeIncidente'], "b.")
plt.plot(xMinMax, yTrainPred, "g-")
plt.plot(xNew, costeIncidente, "rx")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()