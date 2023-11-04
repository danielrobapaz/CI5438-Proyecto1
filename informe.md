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
* Por ultimo, hay variables, que por su naturaleza, tienen magnitudes muy grandes. Un ejemplo es la variable Price y Kilometer. Para manejar estos valores, se aplicaron dos tipos de normalizacion. La primera es la sugerida y la ultima en donde aplicamos el z-score

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

Posteriormente al particionamiento de los datos se hicieron varias corridas del algoritmo utilizando diferentes combinaciones de atributos y métodos de normalización en pro de encontrar la combinación que minimice las magnitudes del error. A continuación se presenta el listado de experimentos realizados:

* Experimento 1: Usando las variables sugeridas, usando el conjunto de datos con nulos eliminados y aplicando la normalizacion sugerida.
* Experimento 2: Usando las variables sugeridas, usando el conjunto de datos rellenando los nulos y aplicando la normalizacion sugerida.
* Experimento 3: Usando las variables sugerdas agregando Length y Width, usando el conjunto de datos con nulos eliminados y aplicando la normalizacion sugerida.
* Experimento 4: Usando las variables sugeridas agregando Length y Width, usando el conjunto de datos con nulos rellenados y aplicando la normaliazcion sugerida.
* Experimento 5: Experimento 1 aplicando la normalizacion del z-score.
* Experimento 6: Experimento 2 aplicando la normalizacion del z-score
* Experimento 7: Experimento 3 aplicando la normalizacion del z-score
* Experimento 8: Experimento 4 aplicando la normalizacion del z-score
* Experimento 9: Usando las variables sugeridas excluyendo Make, con el conjunto de datos con nulos eliminados y aplicando la normalizacion sugerida.
* Experimento 10: Usando las variables sugeridas excluyendo Make y Fuel Type, con el conjunto de datos con nulos eliminados y apicando la normalizacion sugerida.
* Experimento 11: Usando las variables sugeridas excluyendo Make y Owner, con el conjunto de datos con nulos eliminados y aplicando la normalizacion sugerida.
* Experimento 12: Usando las variables sugeridas excluyendo Owner y Fuel Type, con el conjunto de datos con nulos eliminados y aplicando la normalizacion sugerida.
* Experimento 13: Usando las variables sugeridas excluyendo Make, Fuel Type, Seating Capacity e incluyendo Length y Width, con el conjunto de datos con nulos eliminados y aplicando la normalizacion sugerida

Cada uno de los experimentos listados se ejecutó en 3 modalidades distintas variando el número de iteraciones máximas permitidas. Así para cada experimento se efectuó una primera corrida con un máximo de 50 mil iteraciones, una segunda corrida con un máximo de 70 mil iteraciones, y una última corrida con un máximo de 100 mil iteraciones. Para cada una de estas modalidades cada experimento se ejecutó 3 veces, con el fin de validar la consistencia de los resultados.

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

### Tercera serie de ejeciciones: Max 100k iteraciones

| Experimento | avg Maximo error | avg Minimo error | avg Media del error | avg Mediana del error | avg Suma total del error |
| ----------- | ---------------- | ---------------- | ------------------- | --------------------- | ------------------------ |
| 1           | 0,8587           | \-0,1476666667   | \-0,0022            | \-0,0045              | \-0,8467                 |
| 2           | 0,8553           | \-0.0995         | 0,0009              | \-0,0042              | 0,4068                   |
| 3           | 0,8527           | \-0,1352         | \-0,0021            | \-0,0039              | \-0,8154                 |
| 4           | 0,8500666667     | \-0,09253333333  | 0,0004              | \-0,0046              | 0,1824                   |
| 5           | 12,2315          | \-2,0896         | \-0,0203            | \-0,3968666667        | \-10,9084                |
| 6           | 12,2419          | \-1,4307         | 0,01535333333       | \-0,0745              | 6,413566667              |
| 7           | 12,2136          | \-1,8953         | \-0,04373333333     | \-0,08463333333       | \-10,38553333            |
| 8           | 12,2292          | \-1,329466667    | 0,008036666667      | \-0,0594              | 9,9338                   |
| 9           | 0,8316           | \-0,08411        | \-0,0045            | \-0,0065              | \-3477,6954              |
| 10          | 0,8348           | \-0,0845         | \-0,003233333333    | \-0,0069              | \-1,1145                 |
| 11          | 0,826            | \-0,0921         | \-0,002             | \-0,0063              | \-0,7682666667           |
| 12          | 0,8581           | \-0,1477         | \-0,0018            | \-0,0026              | \-0,6976666667           |
| 13          | 0,8845           | \-0,06036666667  | \-0,0029            | \-0,0092              | \-1,0894                 |

## Conclusiones

Al momento de escoger un modelo se optó por elegir aquel que tuviera una media de error de menor magnitud. En consecuencia de esto se presenta como hipótesis final aquella obtenida a partir de la ejecución del experimento 2, que contiene el conjunto de variables sugeridas por el enunciado y emplea completamiento de datos mediante inclusión de media y moda. Para facilitar la legibilidad, en la siguiente tabla se muestran los coeficientes correspondientes a cada variable independiente en la hipótesis originada del modelo resultante del experimento 2 con un límite de iteraciones de 70 mil:

|Variable          |Coeficiente          |
|------------------|---------------------|
|Término Ind.      |-0.01667             |
|Year              |0.10322951607224     |
|Kilometer         |-0.0198414133221436  |
|Fuel Tank Capacity|0.135100608036014    |
|Seating Capacity  |-0.0307287188886536  |
|Manual            |-0.0141267553256291  |
|Automatic         |-0.00254336604548104 |
|Petrol            |-0.00903824469501806 |
|Diesel            |-0.0166857349996346  |
|CNG               |-0.0142605487282244  |
|CNG + CNG         |0.000763281892580848 |
|LPG               |0.00308513496500433  |
|Hybrid            |0.0262691044992729   |
|Petrol + CNG      |-0.000665255010195871|
|First             |-0.0199709114981087  |
|Second            |-0.0303162982878992  |
|Third             |-0.00297494448305441 |
|UnRegistered Car  |0.0414894628047259   |
|Honda             |-0.029745717617814   |
|Maruti Suzuki     |-0.0312868066794956  |
|Hyundai           |-0.036008347304129   |
|Toyota            |-0.0227163347356961  |
|BMW               |0.0256010613170309   |
|Skoda             |-0.0311992686539649  |
|Nissan            |-0.019076057718537   |
|Renault           |-0.0310036958410967  |
|Tata              |-0.0295613258922825  |
|Volkswagen        |-0.0350624802735722  |
|Ford              |-0.0241813714064751  |
|Mercedes-Benz     |0.022795787295267    |
|Audi              |-0.0118779566327994  |
|Mahindra          |-0.034137357260063   |
|MG                |-0.0170261547166387  |
|Jeep              |-0.0201277384846687  |
|Porsche           |0.0974557311382034   |
|Kia               |-0.0226377038593539  |
|Land Rover        |0.080661544960024    |
|Volvo             |0.0104218811868576   |
|Maserati          |0.0134060110932614   |
|Jaguar            |0.00906949126431803  |
|Isuzu             |-0.00441275413839045 |
|MINI              |0.00797579237896356  |
|Ferrari           |0                    |
|Mitsubishi        |-0.00323017321966775 |
|Datsun            |-0.013170628561697   |
|Chevrolet         |-0.0172172240802316  |
|Ssangyong         |-0.00745520002665654 |
|Fiat              |-0.00459578315438114 |
|Rolls-Royce       |0.100696458378839    |
|Lexus             |0.0099323500684573   |