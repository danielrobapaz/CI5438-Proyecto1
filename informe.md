# Informe

## Descripción del problema

Se solicitó la implementación de un algoritmo de regresión lineal mediante descenso de gradiente para la generación de un modelo
matemático que facilite la predicción del precio de un vehículo automotriz dados uno o más atributos del vehículo.

## Descripción de los datos

El conjunto de datos a partir del cual se genera el modelo se proveyó en el archivo `CarDekho.csv`. En este se encuentra la información de un total
de 2059 vehículos diferentes repartida en diferentes atributos correspondientes a cada columna del archivo. Para cada vehículo se incluyen en el archivo 
los siguientes datos:

* Manufacturante del vehículo (Make)
* Modelo del vehículo (Model)
* Precio del vehículo (Price)
* Año de salida al mercado del vehículo (Year)
* Kilometraje del vehículo (Kilometer)
* Tipo de combustible del vehículo (Fuel Type)
* Tipo de transmisión del vehículo (Transmission)
* Ubicación de venta del vehículo (Location)
* Color del vehículo (Color)
* Desplazamiento del motor (Engine)
* Potencia máxima (Max Power)
* Torque máximo (Max Torque)
* Tracción del vehículo (Drivetrain)
* Longitud (Length)
* Anchura (Width)
* Altura (Height)
* Número de Asientos (Seating Capacity)
* Capacidad del tanque de combustible (Fuel Tank Capacity)

Cabe destacar que no todos los campos están completos para todos los vehículos.

## Tratamiento de los datos

Antes de proceder a la generación del modelo fue primero necesario efectuar un pre-procesamiento de los datos para manejar los casos de información faltante, 
establecer un mecanismo de manejo para las variables categóricas del dataset, y transformar todos los datos existentes a una escala común. Dicho pre-procesamiento
se encuentra en el *Jupyter Notebook* `etl.ipynb`, y en líneas generales, se efectuaron las siguientes manipulaciones:

* Para cada variable categórica presente en el dataset con *n* posibles valores se crearon n nuevas variables con cada uno de los 
valores posibles de la variable original. Así si el dato *k* pertence a la categoría *X*, tendrá un valor de 1 en la variable dedicada
a la categoría *X* y 0 en caso contrario.
* En el caso de datos con valores faltantes, se creó una versión del dataset donde estos datos son removidos completamente y otra donde los valores faltantes
fueron sustituidos con la media en el caso de las variables cuantitativas, y con la moda en el caso de las variables categóricas.
* Se efectuó una normalización de todos los valores en pro de llevar todos los datos a una misma escala común.

## Implementación del algoritmo

La implementación del algoritmo se encuentra en la clase *Prediction_model* contenida en el archivo `Prediction_model.py`, a continuación se da un resumen
de los atributos y métodos de la clase:

### Atributos

* data_set: Data Frame de pandas que contiene los datos de entrenamiento para el modelo.
* dep_var: String conteniendo el nombre de la variable dependiente (Precio en nuestro caso)
* ind_vars: Lista de Strings conteniendo los nombres de las variables independientes alrededor de las cuales se elaborará el modelo
* weights: Serie de pandas conteniendo el valor de los pesos para cada una de las variables independientes.

### Métodos

* train_model: Recibe la tasa de aprendizaje, el límite de error y el número máximo de iteraciones y actualiza los pesos del modelo tras ejecutar la regresión lineal con descenso de gradiente. Posteriormente muestra en una gráfica la evolución de la media del error en función del número de iteraciones efectuadas. Por defecto tanto la tasa de aprendizaje como el delta mínimo de error tienen un valor de 0.001
* test_model: Recibe un dataframe de pandas con el conjunto de prueba contra el cual se va a evaluar el modelo, e imprime en pantalla un resumen estadístico de los errores obtenidos, incluyendo valor máximo, valor mínimo, media y mediana del error, y la suma total de los errores.

### Detalles misceláneos

* Se decidió optar por hacer uso de DataFrames y Series de pandas para facilitar el manejo de los datos e incrementar el desempeño del algoritmo.
* El uso de arreglos de numpy, que usa números de precisión fija, generó problemas al momento de intentar ejecutar el algoritmo sobre el conjunto de datos sin normalizar. Nuestra hipótesis es que dado que atributos como el precio y el kilometraje están en escalas sumamente grandes los números de punto flotante terminan haciendo overflow y llevando los resultados a NaN. En consecuencia de esto solo se pudo hacer experimentos con el conjunto de datos normalizado.

## Experimentación y prueba del modelo

Para entrenar y probar el modelo se optó por un método de validación cruzada, donde el conjunto total de datos se separa en un conjunto de entrenamiento (En nuestro caso 80% de los datos) y un conjunto de pruebas (En nuestro caso el 20% restante). Para el particionamiento del modelo utilizamos a su vez la función *train_test_split* contenida en el paquete *sklearn* de Python.

Posteriormente al particionamiento de los datos se hicieron varias corridas del algoritmo utilizando diferentes combinaciones de atributos en pro de encontrar la combinación que minimice las magnitudes del error. Cada uno de estos experimentos se encuentran en el *Jupyter Notebook* `testing.ipynb`. Para cada experimento se realizaron 3 lotes de ejecuciones distintos, con máximo de iteraciones de 50 mil, 70 mil, y 100 mil iteraciones respectivamente. Además para cada lote de ejecuciones se efectuaron 3 ejecuciones distintas en pro de verificar la consistencia del modelo. Cabe destacar que entre distintas ejecuciones no se hicieron modificaciones a atributos como la tasa de aprendizaje, ni el delta mínimo de error, para todas las ejecuciones del algoritmo ambos valores se mantuvieron en sus valores por defecto de 0.001.

 A continuación se presenta un resumen de los resultados obtenidos:

### Primera serie de ejecuciones: Max 50k iteraciones

| Experimento | avg Maximo error | avg Minimo error | avg Media del error | avg Mediana del error | avg Suma total del error |
| ----------- | ---------------- | ---------------- | ------------------- | --------------------- | ------------------------ |
| 1           | 0,8737           | \-0,1083         | \-0,0025            | \-0,0032              | \-0,9391                 |
| 2           | 0,8708           | \-0,0997         | 0,0007              | \-0,0027              | 0,3024                   |
| 3           | 0,8611           | \-0,1098         | \-0,0023            | \-0,0047              | \-0,8927                 |
| 4           | 0,8597           | \-0,0955         | 0,0009              | \-0,0038              | 0,3583                   |
| 5           | 12,1486          | \-1,6908         | \-0,0308            | \-0,1007              | \-11,5680                |
| 6           | 12,1681          | \-1,4836         | 0,0117              | \-0,0797              | 4,8386                   |
| 7           | 12,1412          | \-1,4726         | \-0,0304            | \-0,0924              | \-11,4124                |
| 8           | 12,1616          | \-1,3541         | 0,0197              | \-0,0776              | 8,1364                   |
| 9           | 0,8544           | \-0,0826         | \-0,0027            | \-0,0058              | \-1,0311                 |
| 10          | 0,8556           | \-0,0844         | \-0,0028            | \-0,0058              | \-1,0849                 |
| 11          | 0,8503           | \-0,0920         | \-0,0021            | \-0,0048              | \-0,8184                 |
| 12          | 0,8740           | \-0,1139         | \-0,0023            | \-0,0027              | \-0,8457                 |
| 13          | 0,8863           | \-0,0917         | \-0,0028            | \-0,0082              | \-1,0794                 |


### Segunda serie de ejecuciones: Max 70k iteraciones
| Experimento | avg Maximo error | avg Minimo error | avg Media del error | avg Mediana del error | avg Suma total del error |
| ----------- | ---------------- | ---------------- | ------------------- | --------------------- | ------------------------ |
| 1           | 0,8658           | \-0,1269         | \-0,0023            | \-0,0039              | \-0,9251                 |
| 2           | 0,8626           | \-0,0998         | 0,0008              | \-0,0038              | 0,3498                   |
| 3           | 0,8562           | \-0,1201         | \-0,0022            | \-0,0041              | \-0,8556                 |
| 4           | 0,8542           | \-0,0950         | 0,0011              | \-0,0041              | 0,4485                   |
| 5           | 12,1868          | \-1,8768         | \-0,0300            | \-0,1075              | \-11,2789                |
| 6           | 12,2068          | \-1,4571         | 0,0134              | \-0,0779              | 5,5472                   |
| 7           | 12,1772          | \-1,6704         | \-0,0292            | \-0,0898              | \-10,9540                |
| 8           | 12,1867          | \-1,3242         | 0,0218              | \-0,0659              | 9,0107                   |
| 9           | 0,8422           | \-0,0818         | \-0,0027            | \-0,0067              | \-1,0443                 |
| 10          | 0,8444           | \-0,0841         | \-0,0029            | \-0,0062              | \-1,1052                 |
| 11          | 0,8375           | \-0,0906         | \-0,0021            | \-0,0062              | \-0,7916                 |
| 12          | 0,8656           | \-0,1261         | \-0,0020            | \-0,0024              | \-0,7709                 |
| 13          | 0,8855           | \-0,0894         | \-0,0029            | \-0,0087              | \-1,0890                 |

### Tercera serie de ejeciciones: Max 100k iteraciones (Por hacer)

## Conclusiones

Al momento de escoger un modelo se optó por elegir aquel que tuviera una media de error de menor magnitud. En consecuencia de esto se presenta como hipótesis final aquella obtenida a partir de la ejecución del experimento 2, que contiene el conjunto de variables sugeridas por el enunciado y emplea completamiento de datos mediante inclusión de media y moda. Así para un vehículo *X* la hipótesis escogida da la siguiente ecuación para calcular una aproximación de su precio:

Price(X) = -0.01667 + $0.1032Year(X) - 0.0198Kilometer(X) + 0.1351FuelTankCap(X) - 0.0307SeatingCap(X) - 0.0141Manual(X) - 0.0025Automatic(X) - 0.009Petrol(X) - 0.0167Diesel(X) - 0.0143CNG(X) + 0.0008CNGPlusCNG(X) + 0.0031LPG(X) + 0.0263Hybrid(X) - 0.0007PetrolPlusCNG(X) - 0.02FirstOwner(X) - 0.0303SecondOwner(X) - 0.003ThirdOwner(X) + 0.0415UnRegistered(X) - 0.0297Honda(X) - 0.0313MarutiSuzuki(X) - 0.036Hyundai(X) - 0.0227Toyota(X) + 0.0256BMW(X) - 0.0312Skoda(X) - 0.0191Nissan(X) - 0.031Renault(X) - 0.0296Tata(X) - 0.0351Volkswagen(X) - 0.0242Ford(X) + 0.0228MercedesBenz(X) - 0.0119Audi(X) - 0.0341Mahindra(X) - 0.017MG(X) - 0.0201Jeep(X) + 0.0975Porsche(X) - 0.0226Kia(X) + 0.0807LandRover(X) + 0.0104Volvo(X) + 0.0134Maserati(X) + 0.0091Jaguar(X) - 0.0044Isuzu(X) + 0.008MINI(X) + 0.0Ferrari(X) - 0.0032Mitsubishi(X) - 0.0132Datsun(X) - 0.0172Chevrolet(X) - 0.0075Ssangyong(X) - 0.0046Fiat(X) + 0.1007RollsRoyce(X) + 0.0099Lexus(X)