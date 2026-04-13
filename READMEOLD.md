# Desarrollo de bot de trading algorítmico de futuros a partir de swing charts y apayado en modelos de ML.

**21/01/2026** 

## Diseño ideal del bot
```mermaid
---
title: En progreso
---
graph TD
    %% --- DEFINICIÓN DE CLASES ---
    classDef done fill:#c8f7c5,stroke:#2e8b57,stroke-width:2px,color:#1b4d3e;
    classDef progress fill:#ffe9b3,stroke:#ff9900,stroke-width:2px,color:#8a4f00;
    classDef todo fill:#e0e0e0,stroke:#9e9e9e,stroke-width:2px,color:#4f4f4f;
    classDef improve fill:#d6e4ff,stroke:#3366cc,stroke-width:2px,color:#1a3d7c;

    %% --- 0. DATOS HISTÓRICOS ---
    subgraph "0. Datos históricos"
        H0[Datis OHLCV históricos en formato txt] --> DL0
        H1[Datis OHLCV históricos obtenidos directamente de TWS IB] --> DL0
    end

    %% --- 1. DATA LAKE ---
    subgraph "1. CAPA DE DATOS (DATA LAKE)"
        DL0[Limpieza y Wavelet Denoising] --> DL1
        DL1[Resampling a N minutos y calculo de features acumuladas] --> DL2
        DL2[Feature Engineering: 149+ Cols] --> DL3
        DL3[Selección de features, eliminando las estacionarias y no numéricas] --> DL4
        DL4[Definición de targets: Binario o Trinario] --> DL5
        DL5[Filtrar targets en horas líquidas] --> SL0
    end

    %% --- 2.1 SUPERVISED LEARNING ---
    subgraph "2.1 Supervised Learning"
        SL0[Definir modelos: LSTM, GRU, etc.] --> SL1
        SL1[Entrenamiento walk-forward] --> SL2
        SL2[Validación: Accuracy, Precision, Recall] --> SL3
        SL3[Selección de modelos y umbrales] --> BT0
    end

    %% --- 2.2 UNSUPERVISED LEARNING ---
    subgraph "2.2 Unsupervised Learning"
        UL0[Clustering / Regímenes] --> UL1
        UL1[Validación de clusters]
    end

    %% --- 2.3 REINFORCEMENT LEARNING / PYOMO ---
    subgraph "2.3 Pyomo / RL"
        RL0[Definición del objetivo] --> RL1
        RL1[Definición de variables]
    end
    
    %% --- 3. BACKTESTING ---
    subgraph "3. Backtesting & Estrés"
        BT0[Modelos finalistas] --> BT1
        BT1[BacktestEngine] --> BT2
        BT2[Simulación de CAOS] --> BT3
        BT2 --> BT4
        BT2 --> BT5
        BT3[Lag: Poisson]
        BT4[Slippage: T-Student]
        BT5[Monte Carlo: 100 Iteraciones]
    end

    %% --- 4. OPERACIÓN EN VIVO ---
    subgraph "4. Operación en Vivo"
        BT5 --> LIVE0
        LIVE0[Pipeline de Datos en Vivo] --> LIVE1
        LIVE1[Predicciones cada N minutos] --> LIVE2
        LIVE2[Gestión de Posiciones] --> LIVE3
        LIVE3[Monitoreo y Alertas]
    end

    %% --- 5. REGISTRO Y MONITOREO ---
    subgraph "5. Registro y Monitoreo"
        LIVE3 --> LOG0[Json decisiones]
        BT5 --> LOG1[Json datos de predicción]
    end

    %% --- CONEXIONES ADICIONALES ---
    UL1 --> SL0
    UL1 --> BT0
    UL1 --> LIVE0

    SL3 --> RL0
    UL1 --> RL0
    RL1 --> BT0
    RL1 --> LIVE2

    %% --- EJEMPLO DE ESTADO ---
    class H0 done
```
## Diseño ideal de la arquitectura completa (todo dockerizado)
```mermaid
graph TD

    %% ============================
    %% 1. BOT DE TRADING
    %% ============================
    subgraph BOT["BOT de Trading (Pipeline Completo)"]
        B0[Data Lake + Feature Engineering]
        B1[Supervised Learning]
        B2[Unsupervised Learning]
        B3[Reinforcement Learning / Pyomo]
        B4[Backtesting Engine]
        B5[Live Trading Engine]
        B6[JSON Logs + Métricas]
        
        B0 --> B1
        B0 --> B2
        B1 --> B3
        B2 --> B3
        B3 --> B4
        B4 --> B5
        B5 --> B6
    end

    %% ============================
    %% 2. WEB UI
    %% ============================
    subgraph WEB["Web Dashboard (UI tipo TWS)"]
        W0[Resumen de Bots]
        W1[Estado de Cuenta]
        W2[Posiciones y Predicciones]
        W3[Embeds: MLflow / Airflow]
    end

    %% ============================
    %% 3. MONITORIZACIÓN
    %% ============================
    subgraph MON["Monitorización (Prometheus + Grafana + Alertmanager)"]
        M0[Prometheus]
        M1[Grafana]
        M2[Alertmanager]
    end

    %% ============================
    %% 4. NOTIFICACIONES
    %% ============================
    subgraph NOTIF["Notificaciones"]
        N0[WhatsApp / Telegram / Email]
    end

    %% ============================
    %% CONEXIONES ENTRE BLOQUES
    %% ============================

    %% BOT -> WEB
    B5 --> W0
    B5 --> W1
    B5 --> W2
    B6 --> W2
    B1 --> W3
    B3 --> W3

    %% BOT -> MONITORIZACIÓN
    B6 --> M0
    M0 --> M1
    M1 --> W0

    %% ALERTAS
    M2 --> N0
    M0 --> M2
```

## Directorios
- DespliegueDocker
  - .yaml
  - PostgreSQL_bbdd
- TradingBots
  - Data
    - *.txt
  - Procesado
    - utils_procesado.py
    - visualizacion.ipynb
  - Train
    - utils_train.py
    - entrenamiento.ipynb
    - entrenamiento.py
  - Logs
  - Optimizacion
    - utils_optimizacion.py
    - optimizacion.ipynb
    - optimizacion.py
- Web
  - package.json
  - src

## Organización de todo el proyecto

- **TradingBots**
  - **Datos**
    - Datos de los precios OHLCV de los diferentes activos, en principio 4 (libra inglesa, crudo, e-mini sp500 y oro) tengo 8 archivos txt históricos con los datos día a día y minuto a minuto.
    - Datos del brocker o ficticios para testing, comisiones, spread, etc.
    - Logs e historial de funcionamiento de los bots.
    - Datos sobre los contratos, unidades por contrato, horas en las que se puede negociar, fluctuación mínima y fechas de expiración.
    - Los datos OHLCV quizás en txt, csv o parquet u otro formato.
    - Las métricas en postgreSQL.
  - **Procesado, preprocesado y postprocesado**
    - EDA y obtenición de los parámetros a usar para entrenar los modelos.
    - El EDA lo voy a reutilizar para la visualización de los bots.
    - Para obtener los parámetros voy a usar swingcharts usando candle sticks a partir de datos OHLCV día a día y minuto a minuto.
    - Tengo pensado probar distintos métodos para obtener los datos para los swingcharts
      - Zigzag
      - etc
    - También hay que sacar métricas de los modelos cuando entrenan y de los resultados que dan, tanto métricas de ML como métricas de trading.
  - **Predicción**
    - Mi idea es diseñar un bot para cada activo y quizás un bot para todos los activos y que las decisiones se tomen de manera conjunta o jerárquicamente.
    - Voy a probar con distintas estratégias con los bots.
      1. Que el bot decida si el precio va a subir, mantenerse o bajar, con esta información, usando reglas predefinidas se va a decidir si comprar o vender en short o long.
      2. Que el bot decida si ir long o short.
      3. Que un primer bot prediga cuanto va a variar el precio y un segundo bot decida qué acción tomar.
    - En principio la idea es dejar que los contratos expiren pero es posible que use o reglas o un bot para decidir si vender antes de tiempo.
    - Por otra parte, tengo que probar los modelos y optimizar los hiperparámetros de los modelos.
    - Además de probar con modelos de aprendizaje supervisado también quiero pobar con modelos de aprendizaje no supervisado para detectar tendendicas en los precios y quizás probar modelos de aprendizaje por refuerzo y por último de aprendizaje profundo.
    - Tengo que crear funciones para llamar a stop loss en caso de que los precios bajen demasiado y tenga contratos, también puedo considerar la posibilidad de añadir take profit.
  - **Optimización**
    - Una vez obtenido bots que tengan buenas ganancias, mi idea es optimizar a asignación de dinero a los bots.
    - He pensado en dos opciones.
      1. Usar Pyomo para optimizar ganancias teniendo en cuenta diferentes parámetros como ganancias o pérdidas o la volatilidad del mercado en un momento concreto, la optimización se puede hacer día a día o hora o hora o como se desee.
      2. Entrenar de alguna manera un modelo de ML para que tome esta decisión, seguramente un modelo de aprendizaje por refuerzo.
- **Web**
  - Página web donde visualizar toda la información relativa a los bots.
  - Voy a usar React junto a yarn o npm, por decidir.
  - El diseño lo voy a hacer primero en Canva.
  - Voy a hacer 4 páginas.
    - Página de inicio: Información del estado de los bots, si están tradeadon o no y en qué modo. Además, resumen de los trades del día/semana/mes/total. También, información de cuantos contratos de futuros tengo y el valor y el tipo.
    - Página de seguimiento de los bots: Usando plotly u otro visualizador, mostrar los swing charts de los distintos futuros y marcar con líneas verticales los momentos donde el bot ha realizado acciones incluyendo el precio, apalancamiento, fecha y la predicción.
    - Página de backtesting y modo offline: Poder elegir algún modelo y probar a ver como se comporta por ejemplo con otros datos. Todavía no lo tengo claro.
    - Página para añadir más datos: Controla a ver si los datos que se van recibiendo se vana añadiendo a las diferentes bases de datos y diferentes tablas. Tampoco lo tengo claro.
- **Docker**
  - En caso de que todo vaya bien procederé a desplegar todo el proyecto en Docker para que sea más fácil replicarlo.
  - Habrá dos partes.
    1. Lo primordial es conseguir que todo lo que he hecho hasta el momento funcione correctamente en Docker como si se estuviese corriendo de normal.
    2. Una vez conseguido esto y si hay tiempo, desplegar otros contenedores, por ejemplo grafana para ver que todo vaya bien o alertmanager para recibir información en Whatsapp solo el rendimiento de los bots, por ejemplo un reporte diario de como han ido o alertas por si alguno de los bots ha tenido que parar.
    3. También tengo pensado usar mlflow para reenrenar los modelos o entrenar otros nuevos y para guardar los datos y sacar métricas diarias.

---

# Diario
- **21/01/2026**
  Comienzo del proyecto de una vez por todas, después de atrasarlo desde el comienzo de las clases.
  Defino en líneas generales como tengo pensado dividir el trabajo y el código.

---

# Introducción TFG
> El presente proyecto tiene como objetivo el desarrollo e implementación de un sistema basado en técnicas de Machine Learning orientado al análisis y detección de patrones en “Swing Charts”, una herramienta ampliamente utilizada en el análisis técnico de los mercados financieros. A partir de datos históricos y en tiempo real, el sistema generará representaciones de “Swing Charts” a partir de “Bar Charts” y aplicará algoritmos de aprendizaje supervisado y no supervisado para identificar configuraciones recurrentes, como dobles máximos, dobles mínimos o rupturas de tendencia.
> 
> El desarrollo se llevará a cabo principalmente en Python, utilizando librerías como pandas, scikit-learn y matplotlib, además de entornos de conexión con plataformas de trading como Interactive Brokers API (IBKR). El proyecto busca evaluar la capacidad predictiva de los modelos entrenados y su utilidad en la generación de señales de compra o venta dentro de una estrategia de trading algorítmico. Con ello, se pretende aportar un enfoque innovador que combine la solidez del análisis técnico clásico con la adaptabilidad de las técnicas modernas de inteligencia artificial. 

---

# Asignaturas de las que he sacado conocimientos para el proyecto

:heavy_check_mark: Mucho conocimiento
:heavy_minus_sign: Algo de conocimiento
:heavy_multiplication_x: Nada de conocimiento

- Primero
  - Primer cuatrimestre
    - Álgebra
    - Cálculo
    - Fundamentos de Procesado de Datos
    - :heavy_check_mark: Programación
    - Desarrollo de Habilidades Profesionales
    - Introducción a la Ingeniería de Datos
  - Segundo cuatrimestre
    - :heavy_check_mark: Bases de Datos Relacionales y Datos Estructurados
    - Modelos Matemáticos y Matemática Discreta
    - Optimización
    - Señales y Sistemas
    - Sistemas de Adquisición de Datos

- Segundo
  - Primer cuatrimestre
    - Bases de Datos No Relacionales y Distribuidas
    - Uso Profesional de la Lengua Inglesa
    - Probabilidad y Señales Aleatorias
    - :heavy_check_mark: Programación para Big Data
    - Redes y Servicios de Comunicaciones
  - Segundo cuatrimestre
    - Fundamentos de Gestión Empresarial
    - Inferencia Estadística y Series Temporales
    - Redes de Sensores
    - Teoría de la Información
    - Sistemas de Comunicaciones para Ingeniería de Datos
    - Tecnologías Web

- Tercero
  - Primer cuatrimestre
    - Análisis de Señal
    - :heavy_check_mark: Aprendizaje Automático
    - Arquitecturas de Procesado Masivo de Datos
    - :heavy_check_mark: Computación en la Nube
    - Desarrollo Profesional del Ingeniero de Datos
    - Emprendimiento y Modelos de Negocio
  - Segundo cuatrimestre
    - :heavy_check_mark: Análisis y Visualización de Datos
    - Aplicaciones Sectoriales
    - :heavy_check_mark: Ingeniería Big Data en la Nube
    - Procesado Avanzado de Señales y Datos
    - :heavy_check_mark: Técnicas de Soporte a la Decisión

- Cuarto
  - Primer cuatrimestre
    - :heavy_check_mark: Proyectos de Ingeniería de Datos y Sistemas
    - Ciberseguridad y Protección de Datos
    - Gestión de Proyectos
    - Marco Ético y Legal
    - :heavy_check_mark: Ingeniería Web
  - Segundo cuatrimestre
    - Herramientas para la Computación y Visualización
    - Tecnologías de la Información Geoespacial
    - Electrónica de Consumo

## Contenido exacto de he aplicado de cada asignatura

- Programación
  - Python
- Bases de Datos Relacionales y Datos Estructurados
  - SQL
- Programación para Big Data
  - Sklearn
- Aprendizaje Automático
- Computación en la Nube
  - Docker
- Análisis y Visualización de Datos
  - Plotly
- Ingeniería Big Data en la Nube
- Técnicas de Soporte a la Decisión
  - Pyomo
- Proyectos de Ingeniería de Datos y Sistemas
- Ingeniería Web
  - React