from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

kmeans = load('modelo.kmeans.pkl')

def get_label(cluster):
    if cluster == 0:
        return 'Alto'
    elif cluster == 1:
        return 'Bajo'
    else:
        return 'Desconocido'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/saludo')
def saludo():
    return render_template('hola.html')


@app.route('/predict', methods=['POST'])
def predict():
    ingresos = float(request.form['ingresos'])
    gastos = float(request.form['gastos'])
    
    nuevo_dato = np.array([[ingresos, gastos]])
    prediccion = kmeans.predict(nuevo_dato)
    label = get_label(prediccion[0])

    return render_template('index.html', prediccion=label, ingresos=ingresos, gastos=gastos)  # O redirigir a una página de resultados

if __name__ == '__main__':
    app.run(debug=True)