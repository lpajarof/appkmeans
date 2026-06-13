from joblib import load
import numpy as np

kmeans = load('modelo.kmeans.pkl')

nuevo_dato1 = np.array([[3.5, 2.5]])

prediccion1 = kmeans.predict(nuevo_dato1)
print(f'El nuevo dato {nuevo_dato1} pertenece al cluster: {prediccion1[0]}')