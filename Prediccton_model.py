import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class Prediction_model:
    def __init__(self, data_set: pd.DataFrame, dep_var: str, ind_vars: list[str]):
        self.data_set = data_set
        self.dep_var = dep_var
        self.ind_vars = ind_vars
        self.weights: pd.Series[float] = None
    
    '''
    Función que retorna el vector con los pesos de las variables independientes en un objeto 
    Series de pandas. El algoritmo solo se ejecuta la primera vez que el metodo es invocado,
    en cada llamada subsecuente al método se retorna el vector de pesos ya calculados anteriormente.
    '''
    def get_model(self, learning_rate: float = 0.001, treshold: float = 0.001, max_iter: int = None):
        #Si el modelo ya se calculó se retorna
        if self.weights:
            return self.weights
        else:
            #Se inicializan los pesos en cero
            self.weights = pd.Series([0.0]*(len(self.ind_vars)+1))

            y = self.data_set[self.dep_var]
            x = self.data_set[self.ind_vars]
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

            x_train.insert(0, "ind_term", [1.0]*len(x_train)) #Insertar columna de 1's para el termino independiente w0

            i = 0

            x_plot = []
            y_plot = []

            while True:
                y_pred = np.dot(self.weights, x_train.T)
                err = y_train - y_pred
                dw = (2/len(x_train)) * np.dot(err.T, x_train)
                self.weights = self.weights + (learning_rate*dw)
                i += 1
                
                x_plot.append(i)
                y_plot.append(sum(err))

                if (sum(err) < treshold or (not max_iter is None and max_iter <= i)):
                    print("Number of iterations:", i)
                    break
                
            x_test.insert(0, "ind_term", [1.0]*len(x_test))
            y_test_pred = np.dot(self.weights, x_test.T)
            res = np.mean(abs(y_test - y_test_pred))

            print(f'Sum of error during testing: {round(res, 4)}')

            plt.figure().set_figheight(10)
            plt.figure().set_figwidth(15)
            plt.plot(x_plot, y_plot)
            plt.title('Number of iterations vs. Sum of error')
            plt.xlabel('Number of iteracions')
            plt.ylabel('Mean of error')
            plt.show()

            return self.weights