> README actualizado el día 13/04/2026
# Desarrollo de bot de trading algorítmico de futuros a partir de swing charts y apayado en modelos de ML.

## Diseño del bot V2
- Bot de trading algorítmico de futuros usando ML y swing charts
### 1. Datos
- Datos (Bronze) sacados de txt, time frame de 1 min solo con las columnas `<TICKER>,<PER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`
  - 13 futuros [AD (Dólar Australiano), BP (Libra Esterlina), CL (Petróleo Crudo WTI), EC (Euro FX), ES (E-mini S&P 500), GC (Oro), MFXI (Micro Euro FX), NG (Gas Natural), NQ (E-mini Nasdaq 100), YM (E-mini Dow Jones), ZB (Bono del Tesoro 30Y), ZN (Nota del Tesoro 10Y), ZS (Soja)]
- JSON con información de los futuros
  - (nombre, símbolo, exchange, multiplier, tick size, tick value, currency, trading hours, liquid hours, aprox total fee per side, min slipagge tick, initial margin, maint margin)
- JSON para el entrenamiento y la operación
  - (sampling, horizonte de futuro, max drawdown pct, max contratos, max pct maint margin, lookback (días aprox), etc)
### 2. Limpieza y Visualiación
- Cargar cada futuro y realizar limpieza básica, pasar a datetime DTYYYYMMDD y TIME y añadir una nueva columna con la información temporal completa, una vez logrado pasar todos los datetime a UTC
- Para los futuros con horario extendido y hora de descanso, indicar esa hora de descanso en una columna. Para el resto de horas fuera del horario normal, principalmente sabados y por las noches, eliminar esas filas. (Silver)
- EDA simple antes de feature engineering para ver qué datos limpiar
- EDA más completo una vez realizado el feature engineering
### 3. Feature Engineering y Filtrado de Features
- Sampling de los datos originales a 4 horas, sampleando hasta la última vela usando right
- Generación de 100+ features intradía
- Generación de features interdía
- Features de comparación de activos una vez obtenido los clusters de UL
- Generación de features a partir de swings, dirección de swing y derivados
- Varios pasos para reducir el número de features (Gold)
  - Excluir directamente datos en crudo, los que puedan provocar leakage, identificadores temporales no cíclicos, no estacionarias, volumen absoluto
  - Filtro de alta correlación
  - Percentil 50 o filtro por importancia usando LASSO y un RandomForestClassifier
### 4. Unsupervised Learning
- Dos objetivos
  - Clustering de activos, una vez obtenidos la clasificación generar métricas comparativas con otros activos del cluster y comparación con los centros de los demas clusters (GMM) calcular cuál es el número óptimo de régimenes 
  - Clustering de régimenes de mercado, usar otras features para determinar qué regimen es cada id (GMM) calcular cuál es el número óptimo de régimenes 
### 5. Supervised Learning
- Cálculo del target por triple barrier, usando 2 dias como límite temporal y buscar mejores multiplicadores para SL y TP que superen los costes operativos de IB
- Modelo ensemble en tres paso, para tener clases más balanceadas
- L1 Movimiento (Clasificaicón binaria): LSTM + GRU, predicen si hay movimiento, toca SL o TP sin importar cuál
- L2 Dirección (Clasificaicón binaria): LSTM + GRU, predicen la dirección de aquellos que si han tenido movimiento
- L3 Retorno logarítmico (Regresión con confianza): LSTM + GRU, predecir el retorno esperado junto a la confianza del cálculo
### 6. UL + SL Train y Backtest
- Ambos sistemas se entrenan a la vez usando walk forward con rolling window, en cada ventana (2 años y predecir 6 meses quizás)
- Durante el entrenamiento guardar todos los scaler de cada ventana para el backtest
- Usar métricas clasicas para una primera valoración de los modelos
- Usar monte carlo
- (La definición de como se hará la operación se hace más abajo)
- Crear una clase para los bots para mayor facilidad de uso
### 7. Optimización de Cartera con Pyomo
- Una vez obtenidos los datos Gold de todos los futuros y entrenado los modelos usar parte del train para backtest de pyomo (después)
- Optimizar ganancias
- Añadir limitaciones como información de qué activos se pueden operar a dicha hora del día
- De momento Pyomo no va a tener la habilidad de cerrar operaciones solo de abrirlas
### 8. Backtest de Pyomo
- Usar los mismos parámetros de walk forward con rolling window para comparar el rendimiento global respecto de operar con un solo activo
### 9. Gestión de Riesgos y vigilancia
- Código para obtener datos de bid y ask y calcular posible spread para evitar spreads grandes, en tal caso recalcular spread teórico durante 30 minutos y si no cambia la cosa desistir.
- Comprobar si algún mercado acaba de abrir o está por cerrar (también para evitar spreads altos), en tal caso no operar durante los 15 minutos anteriores o posteriores
- Vigilar si se han cerrado operaciones para añadir la información al histórico
### 10. Operación en Vivo
- Al usar sampling de cada 4 horas, también opero cada 4 horas, quizás con un margen de 15-30 minutos
- Al iniciar el código el sistema lee un json con el nombre de todos los activos a operar en esa sesión, por defecto los 13 pero se puede cambiar manualmente. Se cargan los datos secuencialmente para no saturar la API de IB
- Cada 4 horas se lanzan las predicciones y se pasan los vectores de confianza a Pyomo quien decide qué comprar y en principio también cuanto comprar
- Para cada futuro, se calcula SL y TP usando ATR y redondeando al número correcto de decimales
- En general vigilar la operación
- Cada vez que se activa el sistema se comprueba primero si hay posiciones abiertas, para esos activos en concretos se usan esas fechas para la operación (añadir que una vez cerrado esas operaciones se puede lanzar otra búsqueda del contrato concreto con mayor volumen), si no hay ningún contrato activo del futuro se busca el contrato concreto con mayor volumen
- Añadir la posibilidad de cerrar todas las operaciones si se alcanza max drawdown pct permitido o solo de aquellos activos altamente correlados
- Ambos clusterings se calculan en cada time frame
- Usar logging para mostrar info por consola
### 11. Logging de Contratos
- Dos JSON
  - Posiciones abiertas con toda la información necesaria
  - Posiciones históricas con información para revisar a futuro, incluyendo max drawdown, PnL, sharpe ratio
- Archivos con los datos OHLCV usados por cada futuro para las predicciones, para revisar a futuro
- A futuro desplegar contenedor de SQL para guardar métricas
### 12. Reentrenamiento
- De momento reentreno/fine tuning semanal manual (a futuro mlflow/airflow) tanto todos los modelos, como los scalers
### 13. (Una vez pasado a Docker) Grafana con Alertmanager
- Usar los loggings de la terminal para mostrar información en la IU de grafana
- Usar Alertmanager para reporting diario y reporting de actividad
## Organización
- **tradepy**
  - **load_data**
  - **features**
  - **eda**
  - clean
  - filter_features
  - sl
  - ul
  - opti
  - models
  - load_models
  - backtest_l
  - backtest_p
  - operation
  - risk_n_vig

## Mejoras a realizar
Después de haber realizado el primer despliegue del primero proyecto mínicamente viable me he dado cuenta de diferentes mejoras que se pueden realizar al sistema. Por ello, a continuación voy a listas como estoy pensado actualmente de va a ser la arquitectura de V2
### Datos
- En primer lugar voy a seguir usando datos históricos minuto a minuto, ahora de 13 activos [AD (Dólar Australiano), BP (Libra Esterlina), CL (Petróleo Crudo WTI), EC (Euro FX), ES (E-mini S&P 500), GC (Oro), MFXI (Micro Euro FX), NG (Gas Natural), NQ (E-mini Nasdaq 100), YM (E-mini Dow Jones), ZB (Bono del Tesoro 30Y), ZN (Nota del Tesoro 10Y), ZS (Soja)]
- Paso de sampling de 1 hora a 4 horas para pasar de una estratégia intraday a interday
- Mantener el json (nombre, símbolo, exchange, multiplier, tick size, tick value, currency, trading hours, liquid hours, aprox total fee per side, min slipagge tick, initial margin, maint margin)
- Añadir otro json de configuracion de todas las etapas, pero principalmente para la operacion (max drawdown pct, max contratos, max pct maint margin, sampling, lookback (días aprox), etc)
- Separar carpetas Data, Config, Logs, Imágenes y Posiciones abiertas e histórico [todavía por decidir si un json solo pero con key de estado de la posicion o dos json uno histórico y otro actual]
- Para guardar los modelos mantener de momento la estructura actual
- Usar regex en vez de slicing para obtener las keys de los activos
- Revisar bien los tick size y que en el resto de partes usarlo bien
### Limpieza y visualización
- Mejorar función de carga, intentar evitar tener que hacer if else para cada activo, principal problema pasar todas los timestamps a UTC (mirar si hay manera de no hacerlo manualmente (quizás teniendo los trading hours y sabiendo cuantas horas de descanso hay buscar las N horas sin volumen o con volumen mínimo))
- Mejorar función de limpieza (no eliminar datos de N minutos despues y antes de los descansos y aperturas)
- Revisar la función de resampling para que la última vela se vaya actualizando aunque no hayan pasado 4 horas desde la vela anterior (ahora mismo creo que no hace eso y sería como estar 4 horas sin noticias de lo último)
- Mover las funciones de carga y guardado de modelos a su propio archivo para diferenciar carga modelos y carga de datos
- Mejorar el eda para la limpieza, crear una función de plot más simple para buscar outliers y decidir cómo pasar a UTC
### Feature engineering
- Seguir generando las 100+ features, incluso aumentar este número, features interdía, comparación de activos (no todos con todos intentar buscar los que más correlación tengan) y swing charts. Adaptar las ventanas ver si 14 y 20 siguen siendo los números ideales (56 horas y 80 horas, quizás usar el número de horas de operación para calcular el número de velas necesarias)
- Revisar el tema de NaN, ver cuál es la mejor estratégia
- Para los swings usar el criterio de Gann, tambien añadir otras columnas como la distancia al último swing y la duración
- Considerar usar ffill en vez de dropna para velas vacias
### Aprendizaje supervisado
- Una vez implementado aprendizaje no supervisado, crear un modelo LSTM + GRU para cada régimen de mercado y mantener uno general mientras esa parte no esté completa
- Seguir manteniendo excluir features absolutas, no numéricas, no estacionarias
- Mantener la generación de target usando tiple barrier pero actualizando los multiplicadores para SL y TP y actualizando el timeout a 5 días hábiles (o 120 horas)
- Revisar filtro de liquidez (NY o Londres abierto), en vez de filtrar solo añadir nueva columna o quitarlo para el entrenamiento
- Mantener filtro de correlación
- Revisar el cálculo de los parámetros de entrenamiento de los modelos (lookback, train size, test size, gap) y quizás añadir otros parámetros a calcular automáticamente
- Mantener el SelectModel para seguir filtrando más features
- Revisar un poco el entrenamiento con walk forward con rolling window, en principio bien. También mirar la preparación de las ventanas de datos
- Añadir una tercera capa para mejorar el softvoting y cálculo del log return
- Para L1 y L2 añadir una función que calcule el threshold ideal automáticamente para cada uno de los régimenes de mercado, además de intentar hacerlo dinámico
- Ver qué se puede hacer con los scaler para la fase de operación, reentrenar el scaler cada semana/mes
### Aprendizaje no supervisado
- Usar UL para dos fines, clasificación de régimenes de mercado y agrupación de activos
- Usar Gaussian Mixture Models para el clustering de régimenes
- Para los régimes de mercado habría que una vez agrupados revisar algunas features para ver cuales serían las etiquetas de cada cluster, habría que hacer esto en cada fold del entrenamiento
- Agrupar activos quizás para el cálculo de ciertas features de comparación
- Usar Variational Autoencoder para buscar correlaciones entre los diferentes futuros
### Backtest SL + UL
- Backtest lo más parecido posible a la realidad, intentar simular slipage usando distribuciones de probabilidad
### Optimización de cartera con pyomo
- Una vez entrenados y testeados los 13 sistemas distintos diseñar un sistema de pyomo para maxima beneficio sin arriesgar demasiado, para ello hay que definir diferentes variables como número máximo de contratos, max pct drawdown
- Pyomo devuelve cuantos contratos comprar de cada futuro
- Tiene infomración de todos los futuros, para cada predicción puede consultar información del margen inicial, de si está disponible o no el futuro, porcentaje máximo de equity disponible para cada futuro, del pct máximo a operar
### Backtest
- Intentar reutilizar el backtest anterior pero ahora con pyomo integrado
### Operación
- Añadir los códigos necesarios para evitar problemas que he tenido en la primera versión, mejorar la coordinación entre tws ib y mis json con la información de los contratos abiertos, crear mecanismos que recuperen de forma adecuada los contratos que no se han registrado o se han registrado mal, añadir nivel de redundancia
- Tener dos ciclos distintos, uno de 4 horas donde se hacen las predicciones y de decide a qué entrar y otro más corto cada minuto o X minutos para vigilar las posiciones abiertas
- Usar límites dinámicos dependiendo del régimen de mercado para decidir el máximo riesgo a tener, por defecto 30% de la cartera
- Añadir una función de pánico para el ciclo corto en caso de que se reciban datos anómalos
- Usar los tick size y tick value para redondear correctamente los sl y tp y para calcularlos de manera correcta
- Añadir que al iniciar para cada activo comprueba si hay posiciones abiertas, si las hay opera con esa fecha de caducidad si no busca el que mayor volumen tenga
## Organización de los archivos
- Una carpeta dedicada a los notebooks, enumerados como lo tengo en Robotrader
- **tradepy**: Módulo donde añadir todas las funciones a usar. Usar subcarpetas para mejor organización de los códigos, intentar no superrar un par de cientos de línea por archivo
  - **load_data**: Lo que ya tenía como load
  - **features**: Carpeta subdividida con las funciones para calcular las features ordenadas por tipo de dato (tendencia, momento, volatilidad, volumen, relacionados con el tiempo, features interdiarios, comparativa vs otros activos)
  - **eda**: lo que tenia en analysis pero con otro nombre más entendible, plot con decenas de features para ver el comportamiento de un futuro y quizas poder adaptar una estratégia completa, añadir quizas un plot más básico al inicio (ver lo de las horas para alinear a UTC y los días de la semana) y dejar este para una vez sacadas las featurse y así no tener que calcularlas en el plot
  - **sl**: lo que tengo ahora mismo en models, la función de preparar todo el df con una sola llamada, la función para el target, los filtros de correlación, calculo de número de params ideal, feature importance con sklearn para mayor filtro y el entrenamiento como tal y plot de los resultados de la clasificación/es (si es demasiado largo dividir en preentreno, entreno y postentreno)
  - **ul**: 
  - **opti**: pyomo
  - **models**: definición de las arquitecturas de los modelos para entrenar y para poder cargarlos a la hora de operar
  - **load_models**
  - **backtest**
  - **operación**: separar funciones en diferentes archivos (carga de la info tanto los datos estáticos (JSON estático, semanal, activos) como los datos pedidos a la API (un activo a la vez para no saturar la API), guardar la información de los contratos pasados y presentes con info relevate para review futura (MAE MFE para maxdrawdown y sharpe ratio PnL), operación en sí (abrir y cerrar operaciones (con sl y tp, además de número de contratos), abrir normal, tail, cierre de emergencia, calcular sl y tp), obtener información actualizada (obtener los datos ohlcv necesarios, estado de la cuenta y las posiciones), manejo de errores (reconexiones, igualar json e info del broker, cerrar posiciones que lleven más de la cuenta abiertos o que no se hayan registrado y estén en positivo las que estén en negativo se puede intentar actualizar los sl y tp o si no directamente también cerrarlas), preparar los datos (pasar a UTC, poner los mismos nombres y mismo formato antes de generar features), logs (quizás junto a lo de guardar datos), generar predicción, comprobar márgenes además de que lo haga pyomo)