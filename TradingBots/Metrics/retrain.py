import argparse
import asyncio
import pandas as pd
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from ib_async import *
import logging
import joblib  # ← para guardar el scaler

# ==================== CONFIGURACIÓN ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SYMBOLS = ['GC', 'CL', 'ES', '6B']

TIMEFRAME_DOWNLOAD = '1 min'
DURACION_HISTORICO = '1 Y'
RESAMPLE_TO = '30T'                   # Cambia a '60T' si tus modelos usan 1 hora

EPOCHS = 5
BATCH_SIZE = 64
SEQUENCE_LENGTH = 60
TRAIN_SPLIT = 0.8
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

# ==================== MODELO LSTM ====================
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return self.sigmoid(out)

# ==================== DESCARGA ====================
async def download_data(symbol: str, expiration: str = '') -> pd.DataFrame:
    ib = IB()
    await ib.connect('127.0.0.1', 7497, clientId=999, readonly=True)

    contract = Future(symbol=symbol, lastTradeDateOrContractMonth=expiration,
                      exchange='CME', currency='USD')
    contracts = await ib.qualifyContracts(contract)
    if not contracts:
        raise ValueError(f"Contrato {symbol} no encontrado")
    contract = contracts[0]

    logger.info(f"Descargando {DURACION_HISTORICO} de {symbol}...")
    bars = await ib.reqHistoricalData(
        contract=contract, endDateTime='', durationStr=DURACION_HISTORICO,
        barSizeSetting=TIMEFRAME_DOWNLOAD, whatToShow='TRADES',
        useRTH=True, formatDate=1, keepUpToDate=False
    )
    await ib.disconnect()

    df = util.df(bars)
    df['datetime'] = pd.to_datetime(df['date'])
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# ==================== PREPROCESAMIENTO ====================
def prepare_data(df: pd.DataFrame, resample_to: str = RESAMPLE_TO):
    df_resampled = df.resample(resample_to).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

    df_resampled['return'] = df_resampled['close'].pct_change()
    df_resampled['atr'] = calculate_atr(df_resampled)
    df_resampled.dropna(inplace=True)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_resampled)

    X, y = [], []
    for i in range(len(scaled) - SEQUENCE_LENGTH):
        X.append(scaled[i:i + SEQUENCE_LENGTH])
        next_close = scaled[i + SEQUENCE_LENGTH, 3]
        prev_close = scaled[i + SEQUENCE_LENGTH - 1, 3]
        y.append(1 if next_close > prev_close else 0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    X_tensor = torch.tensor(X).to(DEVICE)
    y_tensor = torch.tensor(y).to(DEVICE)

    return X_tensor, y_tensor, scaler

def calculate_atr(df: pd.DataFrame, period: int = 14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ==================== FINE-TUNING ====================
def fine_tune_model(model_path: str, X_train, y_train, X_val, y_val):
    model = LSTMModel(input_size=5).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.train()
    logger.info(f"Modelo cargado → fine-tuning en {DEVICE}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f}")

    return model

# ==================== MAIN ====================
async def retrain_symbol(symbol: str):
    try:
        df = await download_data(symbol)
        X, y, scaler = prepare_data(df)

        split = int(len(X) * TRAIN_SPLIT)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        logger.info(f"{symbol} → {len(X)} secuencias generadas")

        # Rutas
        model_path = MODEL_DIR / f"lstm_{symbol.lower()}.pth"
        scaler_path = MODEL_DIR / f"lstm_{symbol.lower()}_scaler.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró {model_path}")

        # Fine-tuning
        new_model = fine_tune_model(str(model_path), X_train, y_train, X_val, y_val)

        # Guardar modelo + scaler
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        new_model_path = MODEL_DIR / f"lstm_{symbol.lower()}_{timestamp}.pth"

        torch.save(new_model.state_dict(), new_model_path)
        joblib.dump(scaler, scaler_path)                     # ← scaler guardado aquí

        logger.info(f"✅ Modelo guardado: {new_model_path}")
        logger.info(f"✅ Scaler guardado: {scaler_path}")

        # Sobrescribir versión production (para que el bot lo lea directamente)
        torch.save(new_model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)                     # ← scaler production
        logger.info(f"   → Versión production actualizada (modelo + scaler)")

    except Exception as e:
        logger.error(f"❌ Error reentrenando {symbol}: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Reentrenamiento semanal PyTorch + scaler")
    parser.add_argument('--symbol', type=str, help='Solo un símbolo o todos')
    args = parser.parse_args()

    symbols_to_train = [args.symbol.upper()] if args.symbol else SYMBOLS

    for symbol in symbols_to_train:
        logger.info(f"=== REENTRENANDO {symbol} ===")
        await retrain_symbol(symbol)
        await asyncio.sleep(5)

    logger.info("🎉 Reentrenamiento COMPLETO. Modelos y scalers listos para el bot.")

if __name__ == "__main__":
    asyncio.run(main())
