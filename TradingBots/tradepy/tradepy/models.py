import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from arch import arch_model

def linear_regression_model(df):
    # Regresion lineal
    df = df.copy()
    X = df.drop(columns=["close"])
    Y = df["close"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    return model

def logistic_regression_model(df):
    # Regresion logistica
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df.drop(columns=["close", "target"])
    Y = df["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    preccision = accuracy_score(Y_test, Y_pred)
    matriz_confusion = confusion_matrix(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, model.predict(X_test)[:, 1])

    print(f"Accuracy: {preccision}")
    print(f"Confusion Matrix:\n{matriz_confusion}")
    print(f"ROC AUC Score: {roc_auc}")
    return model

def arima_model(df):
    # ARIMA
    df = df.copy()
    df.reset_index()
    dates = df["date"]
    close = df["close"]
    close.index = range(len(close))
    model = ARIMA(close, order=(5, 1, 2))
    model_fit = model.fit()

    print(model_fit.summary())

    in_sample_pred = model_fit.predict(start=0, end=len(close)-1)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, close, label='Actual')
    plt.plot(dates, in_sample_pred, label='Predicted', alpha=0.7)
    plt.title('ARIMA In-Sample Prediction') 
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    return model_fit

def garch_model(df):
    # GARCH
    df = df.copy()
    df['returns'] = df['close'].pct_change().dropna()*100
    model = arch_model(df['returns'].dropna(), vol='Garch', p=1, q=1)
    model_fit = model.fit()
    
    print(model_fit.summary())
    
    in_sample_pred = model_fit.conditional_volatility
    
    oos_pred = model_fit.forecast(horizon=10)
    forecast_variance = oos_pred.variance.iloc[-1]
    forecast_volatility_oos = np.sqrt(forecast_variance)
    
    df["historical_volatility"] = df['returns'].rolling(window=20).std()
    
    future_dates = pd.date_range(start=df['datetime'].iloc[-1] + pd.Timedelta(minutes=5), periods=10, freq='5min')
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['historical_volatility'], label='Historical Volatility')
    plt.plot(future_dates, forecast_volatility_oos.values, label='Forecasted Volatility', marker='o')
    
    plt.plot(df['datetime'], in_sample_pred, label='In-Sample Volatility', alpha=0.7)
    plt.title('GARCH Volatility Forecast')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()  
    return model_fit

def sarima_model(df):
    # SARIMA
    df = df.copy()
    df.reset_index()
    dates = df["date"]
    close = df["close"]
    close.index = range(len(close))
    model = SARIMAX(close, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    
    print(model_fit.summary())
    
    in_sample_pred = model_fit.predict(start=0, end=len(close)-1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, close, label='Actual')
    plt.plot(dates, in_sample_pred, label='Predicted', alpha=0.7)
    plt.title('SARIMA In-Sample Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    return model_fit

def decision_tree_model(df):
    # Decision Tree Classifier
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df.drop(columns=["close", "target"])
    Y = df["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
    return model

def random_forest_model(df):
    # Random forests
    # df["SMA_10"]
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df.drop(columns=["close", "target"])
    Y = df["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid={
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }, cv=3, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, Y_train)

    print(f"Best Hyperparameters: {grid_search.best_params_}")

    best_rf = grid_search.best_estimator_
    Y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
    return best_rf

def lstm_model(df):
    # LSTM
    df = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(df.drop(columns=["close"]))
    X_train = []
    Y_train = []
    window_size = 60
    for i in range(window_size, len(X_scaled)):
        X_train.append(X_scaled[i-window_size:i, 0])
        Y_train.append(X_scaled[i, 0])
        
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, Y_train, epochs=50, batch_size=32)
    
    df_test = df[-window_size:].copy()
    df_total = pd.concat([df, df_test], axis=0)
    inputs = df_total[len(df_total) - len(df_test) - window_size:]['close'].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    X_test = []
    for i in range(window_size, len(inputs)):
        X_test.append(inputs[i-window_size:i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['close'], label='Actual Price')
    plt.plot(df_test['datetime'], predicted_price, label='Predicted Price', alpha=0.7)
    plt.title('LSTM Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    rmse = np.sqrt(mean_squared_error(df_test['close'], predicted_price))
    mae = mean_absolute_error(df_test['close'], predicted_price)
    mape = mean_absolute_percentage_error(df_test['close'], predicted_price)
    
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    return model

def kmeans_clustering(df, n_clusters=3):
    # Clustering
    # Kmeans
    df = df.copy()
    X = df.drop(columns=["close"])
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    sns.scatterplot(data=df, x='rsi', y='adx', hue='cluster', palette='Set1')
    plt.title('Clustering de Indicadores Técnicos')
    plt.xlabel('RSI')
    plt.ylabel('ADX')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['close'], label='Close Price')
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['datetime'], cluster_data['close'], label=f'Cluster {cluster}', alpha=0.6)
    plt.title('Price Movement by Cluster')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    return kmeans

def dbscan_clustering(df, eps=0.5, min_samples=5):
    # DBSCAN
    df = df.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=["close"]))
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)
    
    plt.figure(figsize=(12, 6))
    noise = df[df['dbscan_cluster'] == -1]
    clusters = df[df['dbscan_cluster'] != -1]
    
    plt.scatter(noise['datetime'], noise['close'], label='Noise', color='red', alpha=0.6)
    plt.scatter(clusters['datetime'], clusters['close'], c=clusters['dbscan_cluster'], cmap='Set1', label='Clusters', alpha=0.6)
    plt.title('DBSCAN Clustering of Price Movements')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    return dbscan