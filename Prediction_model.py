import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Prediction_model:
    def __init__(self, training_set: pd.DataFrame, dep_var: str, ind_vars: list[str]):
        self.data_set = training_set[ind_vars + [dep_var]]
        self.dep_var = dep_var
        self.ind_vars = ind_vars
        self.weights: pd.Series[float] = None
    
    
    '''
    Función que retorna el vector con los pesos de las variables independientes en un objeto 
    Series de pandas. El algoritmo solo se ejecuta la primera vez que el metodo es invocado,
    en cada llamada subsecuente al método se retorna el vector de pesos ya calculados anteriormente.
    '''
    def train_model(self, learning_rate: float = 0.001, treshold: float = 0.001, max_iter: int = None):
        #Si el modelo ya se calculó se retorna
        if self.weights:
            return self.weights
        else:
            #Se inicializan los pesos en cero
            self.weights = pd.Series([0.0]*(len(self.ind_vars)+1))

            hw = self.data_set[self.dep_var]
            x = self.data_set[self.ind_vars]

            x.insert(0, "ind_term", [1.0]*len(x)) #Insertar columna de 1's para el termino independiente w0

            i = 0

            iter_num_plot = []
            err_plot = []

            while True:
                y_pred = np.dot(self.weights, x.T)
                err = hw - y_pred
                dw = (2/len(x)) * np.dot(err.T, x)
                self.weights = self.weights + (learning_rate*dw)
                i += 1
                
                iter_num_plot.append(i)
                err_plot.append(np.mean(abs(err)))

                if (np.mean(abs(err)) < treshold or (not max_iter is None and max_iter <= i)):
                    print("Number of iterations:", i)
                    break

            plt.figure().set_figheight(10)
            plt.figure().set_figwidth(15)
            plt.plot(iter_num_plot, err_plot)
            plt.title('Number of iterations vs. Mean of error')
            plt.xlabel('Number of iterations')
            plt.ylabel('Mean of error')
            plt.show()

    '''
    Recibe el conjunto de datos de prueba y muestra al usuario datos sobre la 
    precisión del modelo.
    '''
    def test_model(self, test_set: pd.DataFrame):

        if not np.any(self.weights):
            print("El modelo aún no ha sido entrenado. Entrenar el modelo antes de testear")
            return
        y_test = test_set[self.dep_var]
        x_test = test_set[self.ind_vars]

        #Variable independiente
        x_test.insert(0, "ind_term", [1.0]*len(x_test))

        #Se calcula el valor de la hipótesis para cada uno de los datos de prueba
        model_results = np.dot(self.weights, x_test.T)
        err_vec = y_test - model_results

        print("Resumen de error del modelo:")
        print(f'\tValor máximo del error: {np.max(err_vec)}')
        print(f'\tValor mínimo del error: {np.min(err_vec)}')
        print(f'\tMedia del error: {np.mean(err_vec)}')
        print(f'\tMediana del error: {np.median(err_vec)}')
        print(f'\tSuma total del error: {np.sum(err_vec)}')

