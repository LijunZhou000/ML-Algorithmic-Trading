import torch.nn as nn
import torch
class GoldLSTM_Triple_Pro(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        lstm_out, (hn, _) = self.lstm(x)

        # Última capa
        x = hn[-1]

        # Normalización (MUY importante en series temporales)
        x = self.norm(x)

        return self.head(x)  # logits

class GoldLSTM_Triple_Pro_v2(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False # Si pones True, recuerda ajustar dimensiones
        )
        self.attn = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64), # BatchNorm aquí ayuda a la convergencia rápida
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )

    # def forward(self, x):
    #     # x shape: (batch, seq_len, input_dim)
    #     lstm_out, _ = self.lstm(x)

    #     # Extraemos el último paso temporal de la última capa: (batch, hidden_dim)
    #     # Esto es más seguro que hn[-1]
    #     last_step = lstm_out[:, -1, :]

    #     x = self.norm(last_step)
        
    #     return self.head(x) # Salida: Logits para CrossEntropyLoss
    def forward(self, x):
        lstm_out, _ = self.lstm(x)          # (B, T, H)

        # Attention: aprender qué timesteps importan más
        attn_w = torch.softmax(
            self.attn(lstm_out), dim=1      # (B, T, 1)
        )
        last_step = (attn_w * lstm_out).sum(dim=1)  # (B, H)
        x = self.norm(last_step)
        return self.head(x)
    
class GoldGRU_Triple_Pro(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)  # logits
        )

    def forward(self, x):
        gru_out, hn = self.gru(x)

        # Última capa GRU
        x = hn[-1]

        # Normalización temporal
        x = self.norm(x)

        return self.head(x)  # logits
    
# class GoldGRU_Triple_Pro_v2(nn.Module):
#     def __init__(self, input_dim, output_dim=3, hidden_dim=128, num_layers=2, dropout=0.3):
#         super().__init__()

#         self.gru = nn.GRU( # <-- Cambiamos LSTM por GRU
#             input_dim,
#             hidden_dim,
#             num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )

#         self.norm = nn.LayerNorm(hidden_dim)
#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, 64),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, output_dim)
#         )

#     def forward(self, x):
#         # La GRU solo devuelve (output, hn), no hay (hn, cn)
#         gru_out, _ = self.gru(x)

#         # Extraemos el último step
#         x = gru_out[:, -1, :]
#         x = self.norm(x)

#         return self.head(x)
class GoldGRU_Triple_Pro_v2(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attn = nn.Linear(hidden_dim, 1)        # ← añadido
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)                            # (B, T, H)
        attn_w = torch.softmax(self.attn(gru_out), dim=1)  # (B, T, 1)
        x = (attn_w * gru_out).sum(dim=1)                  # (B, H)
        x = self.norm(x)
        return self.head(x)
    
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