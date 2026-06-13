import numpy as np
from sklearn.cluster import KMeans
from joblib import dump

# Datos
X = np.array([
    [1,1],
    [2,1.5],
    [4,3],
    [7,5],
    [5,3.5],
    [5,4.5]
])

# Modelo
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Guardar modelo
dump(kmeans, 'modelo.kmeans.pkl')