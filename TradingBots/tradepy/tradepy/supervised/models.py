import torch.nn as nn
import torch

# class GoldLSTM_L1_Move(nn.Module):
#     """Nivel 1: ¿Hay movimiento? → 0=HOLD, 1=ACCIÓN"""
#     def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_dim, hidden_dim, num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
#         # Attention con query aprendida
#         self.attn_q = nn.Linear(hidden_dim, hidden_dim)
#         self.attn_k = nn.Linear(hidden_dim, hidden_dim)
#         self.attn_v = nn.Linear(hidden_dim, 1)

#         self.norm = nn.LayerNorm(hidden_dim)
#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, 64),
#             nn.LayerNorm(64),   # LayerNorm en vez de BatchNorm1d → funciona con cualquier batch size
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.LayerNorm(32),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, output_dim)
#         )

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         q = torch.tanh(self.attn_q(lstm_out))
#         k = torch.tanh(self.attn_k(lstm_out))
#         attn_w = torch.softmax(self.attn_v(q * k), dim=1)
#         x = (attn_w * lstm_out).sum(dim=1)
#         x = self.norm(x)
#         return self.head(x)


# class GoldLSTM_L2_Dir(nn.Module):
#     """Nivel 2: ¿Qué dirección? → 0=SELL, 1=BUY"""
#     def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_dim, hidden_dim, num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
#         self.attn_q = nn.Linear(hidden_dim, hidden_dim)
#         self.attn_k = nn.Linear(hidden_dim, hidden_dim)
#         self.attn_v = nn.Linear(hidden_dim, 1)

#         self.norm = nn.LayerNorm(hidden_dim)
#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, 64),
#             nn.LayerNorm(64),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.LayerNorm(32),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, output_dim)
#         )

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         q = torch.tanh(self.attn_q(lstm_out))
#         k = torch.tanh(self.attn_k(lstm_out))
#         attn_w = torch.softmax(self.attn_v(q * k), dim=1)
#         x = (attn_w * lstm_out).sum(dim=1)
#         x = self.norm(x)
#         return self.head(x)


# class GoldGRU_L1_Move(nn.Module):
#     """Nivel 1: ¿Hay movimiento? → 0=HOLD, 1=ACCIÓN"""
#     def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
#         super().__init__()
#         self.gru = nn.GRU(
#             input_dim, hidden_dim, num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
#         self.attn_q = nn.Linear(hidden_dim, hidden_dim)
#         self.attn_k = nn.Linear(hidden_dim, hidden_dim)
#         self.attn_v = nn.Linear(hidden_dim, 1)

#         self.norm = nn.LayerNorm(hidden_dim)
#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, 64),
#             nn.LayerNorm(64),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.LayerNorm(32),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, output_dim)
#         )

#     def forward(self, x):
#         gru_out, _ = self.gru(x)
#         q = torch.tanh(self.attn_q(gru_out))
#         k = torch.tanh(self.attn_k(gru_out))
#         attn_w = torch.softmax(self.attn_v(q * k), dim=1)
#         x = (attn_w * gru_out).sum(dim=1)
#         x = self.norm(x)
#         return self.head(x)


# class GoldGRU_L2_Dir(nn.Module):
#     """Nivel 2: ¿Qué dirección? → 0=SELL, 1=BUY"""
#     def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
#         super().__init__()
#         self.gru = nn.GRU(
#             input_dim, hidden_dim, num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
#         self.attn_q = nn.Linear(hidden_dim, hidden_dim)
#         self.attn_k = nn.Linear(hidden_dim, hidden_dim)
#         self.attn_v = nn.Linear(hidden_dim, 1)

#         self.norm = nn.LayerNorm(hidden_dim)
#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, 64),
#             nn.LayerNorm(64),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.LayerNorm(32),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, output_dim)
#         )

#     def forward(self, x):
#         gru_out, _ = self.gru(x)
#         q = torch.tanh(self.attn_q(gru_out))
#         k = torch.tanh(self.attn_k(gru_out))
#         attn_w = torch.softmax(self.attn_v(q * k), dim=1)
#         x = (attn_w * gru_out).sum(dim=1)
#         x = self.norm(x)
#         return self.head(x)

class GoldLSTM_L1_Move(nn.Module):
    """Nivel 1: ¿Hay movimiento? → 0=HOLD, 1=ACCIÓN"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attn = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)
        x = (attn_w * lstm_out).sum(dim=1)
        x = self.norm(x)
        return self.head(x)


class GoldLSTM_L2_Dir(nn.Module):
    """Nivel 2: ¿Qué dirección? → 0=SELL, 1=BUY"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attn = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            # nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)
        x = (attn_w * lstm_out).sum(dim=1)
        x = self.norm(x)
        return self.head(x)
    
class GoldGRU_L1_Move(nn.Module):
    """Nivel 1: ¿Hay movimiento? → 0=HOLD, 1=ACCIÓN"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attn = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w = torch.softmax(self.attn(gru_out), dim=1)
        x = (attn_w * gru_out).sum(dim=1)
        x = self.norm(x)
        return self.head(x)


class GoldGRU_L2_Dir(nn.Module):
    """Nivel 2: ¿Qué dirección? → 0=SELL, 1=BUY"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attn = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            # Sin BatchNorm1d — igual que L2 LSTM
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w = torch.softmax(self.attn(gru_out), dim=1)
        x = (attn_w * gru_out).sum(dim=1)
        x = self.norm(x)
        return self.head(x)