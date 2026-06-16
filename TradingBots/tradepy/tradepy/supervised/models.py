import torch.nn as nn
import torch
import math

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
    
# ============================================================================
# LSTM MODELS
# ============================================================================
 
class BasicLSTM_L1_Move(nn.Module):
    """Nivel 1: ¿Hay movimiento? → 0=HOLD, 1=ACCIÓN (Clasificación binaria)"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
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
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Usar el último hidden state
        x = h_n[-1]
        return self.head(x)
 
 
class BasicLSTM_L2_Dir(nn.Module):
    """Nivel 2: ¿Qué dirección? → 0=SELL, 1=BUY (Clasificación binaria)"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
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
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        return self.head(x)
 
 
class BasicLSTM_L3_Regression(nn.Module):
    """Nivel 3: Predicción de retorno logarítmico a X velas (Regresión)"""
    def __init__(self, input_dim, output_dim=1, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
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
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        return self.head(x)
 
 
# ============================================================================
# GRU MODELS
# ============================================================================
 
class BasicGRU_L1_Move(nn.Module):
    """Nivel 1: ¿Hay movimiento? → 0=HOLD, 1=ACCIÓN (Clasificación binaria)"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
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
        gru_out, h_n = self.gru(x)
        # Usar el último hidden state
        x = h_n[-1]
        return self.head(x)
 
 
class BasicGRU_L2_Dir(nn.Module):
    """Nivel 2: ¿Qué dirección? → 0=SELL, 1=BUY (Clasificación binaria)"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
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
        gru_out, h_n = self.gru(x)
        x = h_n[-1]
        return self.head(x)
 
 
class BasicGRU_L3_Regression(nn.Module):
    """Nivel 3: Predicción de retorno logarítmico a X velas (Regresión)"""
    def __init__(self, input_dim, output_dim=1, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
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
        gru_out, h_n = self.gru(x)
        x = h_n[-1]
        return self.head(x)
    
class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention: múltiples perspectivas simultáneamente"""
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim debe ser divisible por num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            attn_out: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, hidden_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (batch, num_heads, seq_len, seq_len)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, head_dim)
        context = context.view(batch_size, seq_len, self.hidden_dim)
        
        # Final output projection
        output = self.out_proj(context)
        return output
 
 
class SqueezeExcitation(nn.Module):
    """SE Block: Atención a nivel de canales (features)"""
    
    def __init__(self, hidden_dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, max(1, hidden_dim // reduction))
        self.fc2 = nn.Linear(max(1, hidden_dim // reduction), hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            weighted: (batch, seq_len, hidden_dim)
        """
        # Global average pooling sobre seq_len
        squeeze = x.mean(dim=1)  # (batch, hidden_dim)
        
        # FC layers (excitation)
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)  # (batch, hidden_dim)
        
        # Reshape y multiplicar
        excitation = excitation.unsqueeze(1)  # (batch, 1, hidden_dim)
        return x * excitation
 
 
class PositionalEncoding(nn.Module):
    """Position Embeddings: marca temporal (hour of day)"""
    
    def __init__(self, hidden_dim, max_len=1440):  # 1440 min = 24 horas
        super().__init__()
        
        # Crear tabla de positional encodings
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                            (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, hidden_dim)
    
    def forward(self, x, positions=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            positions: (batch, seq_len) posiciones opcionales
        Returns:
            x + positional_encoding
        """
        if positions is None:
            # Usar posiciones secuenciales
            return x + self.pe[:, :x.size(1), :]
        else:
            # Usar posiciones específicas (ej: hora del día)
            pe = self.pe[0].gather(0, positions.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            return x + pe
 
 
# ============================================================================
# ADVANCED LSTM MODELS
# ============================================================================
 
class AdvancedLSTM_L1_Move(nn.Module):
    """
    Nivel 1 Advanced: ¿Hay movimiento?
    Componentes: BiLSTM, Multi-Head Attention, Pre-Norm, Residual, SE, Pos Encoding
    """
    
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2, 
                 num_heads=4, dropout=0.3, bidirectional=False):
        super().__init__()
        
        # Position embeddings
        self.pos_enc = PositionalEncoding(input_dim)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Pre-norm style: LayerNorm antes de cada componente
        self.norm1 = nn.LayerNorm(lstm_output_dim)
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(lstm_output_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(lstm_output_dim)
        
        # Squeeze-Excitation
        self.se = SqueezeExcitation(lstm_output_dim)
        
        self.norm3 = nn.LayerNorm(lstm_output_dim)
        
        # Head
        self.head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x, positions=None):
        # Position encoding
        x = self.pos_enc(x, positions)
        
        # Pre-norm LSTM + residual
        x_norm = self.norm1(x)
        lstm_out, _ = self.lstm(x_norm)
        x = x + lstm_out  # Residual (si shapes compatibles)
        
        # Pre-norm Attention + residual
        x_norm = self.norm2(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out  # Residual
        
        # Pre-norm SE + residual
        x_norm = self.norm3(x)
        se_out = self.se(x_norm)
        x = x + se_out  # Residual
        
        # Usar último hidden state
        x = x[:, -1, :]
        
        return self.head(x)
 
 
class AdvancedLSTM_L2_Dir(nn.Module):
    """Nivel 2 Advanced: ¿Qué dirección?"""
    
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2,
                 num_heads=4, dropout=0.3, bidirectional=False):
        super().__init__()
        
        self.pos_enc = PositionalEncoding(input_dim)
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.norm1 = nn.LayerNorm(lstm_output_dim)
        self.attention = MultiHeadAttention(lstm_output_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(lstm_output_dim)
        self.se = SqueezeExcitation(lstm_output_dim)
        self.norm3 = nn.LayerNorm(lstm_output_dim)
        
        self.head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x, positions=None):
        x = self.pos_enc(x, positions)
        
        x_norm = self.norm1(x)
        lstm_out, _ = self.lstm(x_norm)
        x = x + lstm_out
        
        x_norm = self.norm2(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out
        
        x_norm = self.norm3(x)
        se_out = self.se(x_norm)
        x = x + se_out
        
        x = x[:, -1, :]
        return self.head(x)
 
 
class AdvancedLSTM_L3_Regression(nn.Module):
    """Nivel 3 Advanced: Predicción de retorno (regresión)"""
    
    def __init__(self, input_dim, output_dim=1, hidden_dim=128, num_layers=2,
                 num_heads=4, dropout=0.3, bidirectional=False):
        super().__init__()
        
        self.pos_enc = PositionalEncoding(input_dim)
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.norm1 = nn.LayerNorm(lstm_output_dim)
        self.attention = MultiHeadAttention(lstm_output_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(lstm_output_dim)
        self.se = SqueezeExcitation(lstm_output_dim)
        self.norm3 = nn.LayerNorm(lstm_output_dim)
        
        self.head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x, positions=None):
        x = self.pos_enc(x, positions)
        
        x_norm = self.norm1(x)
        lstm_out, _ = self.lstm(x_norm)
        x = x + lstm_out
        
        x_norm = self.norm2(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out
        
        x_norm = self.norm3(x)
        se_out = self.se(x_norm)
        x = x + se_out
        
        x = x[:, -1, :]
        return self.head(x)
 
 
# ============================================================================
# ADVANCED GRU MODELS
# ============================================================================
 
class AdvancedGRU_L1_Move(nn.Module):
    """Nivel 1 Advanced GRU: ¿Hay movimiento?"""
    
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2,
                 num_heads=4, dropout=0.3, bidirectional=False):
        super().__init__()
        
        self.pos_enc = PositionalEncoding(input_dim)
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.norm1 = nn.LayerNorm(gru_output_dim)
        self.attention = MultiHeadAttention(gru_output_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(gru_output_dim)
        self.se = SqueezeExcitation(gru_output_dim)
        self.norm3 = nn.LayerNorm(gru_output_dim)
        
        self.head = nn.Sequential(
            nn.Linear(gru_output_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x, positions=None):
        x = self.pos_enc(x, positions)
        
        x_norm = self.norm1(x)
        gru_out, _ = self.gru(x_norm)
        x = x + gru_out
        
        x_norm = self.norm2(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out
        
        x_norm = self.norm3(x)
        se_out = self.se(x_norm)
        x = x + se_out
        
        x = x[:, -1, :]
        return self.head(x)
 
 
class AdvancedGRU_L2_Dir(nn.Module):
    """Nivel 2 Advanced GRU: ¿Qué dirección?"""
    
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, num_layers=2,
                 num_heads=4, dropout=0.3, bidirectional=False):
        super().__init__()
        
        self.pos_enc = PositionalEncoding(input_dim)
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.norm1 = nn.LayerNorm(gru_output_dim)
        self.attention = MultiHeadAttention(gru_output_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(gru_output_dim)
        self.se = SqueezeExcitation(gru_output_dim)
        self.norm3 = nn.LayerNorm(gru_output_dim)
        
        self.head = nn.Sequential(
            nn.Linear(gru_output_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x, positions=None):
        x = self.pos_enc(x, positions)
        
        x_norm = self.norm1(x)
        gru_out, _ = self.gru(x_norm)
        x = x + gru_out
        
        x_norm = self.norm2(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out
        
        x_norm = self.norm3(x)
        se_out = self.se(x_norm)
        x = x + se_out
        
        x = x[:, -1, :]
        return self.head(x)
 
 
class AdvancedGRU_L3_Regression(nn.Module):
    """Nivel 3 Advanced GRU: Predicción de retorno"""
    
    def __init__(self, input_dim, output_dim=1, hidden_dim=128, num_layers=2,
                 num_heads=4, dropout=0.3, bidirectional=False):
        super().__init__()
        
        self.pos_enc = PositionalEncoding(input_dim)
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.norm1 = nn.LayerNorm(gru_output_dim)
        self.attention = MultiHeadAttention(gru_output_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(gru_output_dim)
        self.se = SqueezeExcitation(gru_output_dim)
        self.norm3 = nn.LayerNorm(gru_output_dim)
        
        self.head = nn.Sequential(
            nn.Linear(gru_output_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x, positions=None):
        x = self.pos_enc(x, positions)
        
        x_norm = self.norm1(x)
        gru_out, _ = self.gru(x_norm)
        x = x + gru_out
        
        x_norm = self.norm2(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out
        
        x_norm = self.norm3(x)
        se_out = self.se(x_norm)
        x = x + se_out
        
        x = x[:, -1, :]
        return self.head(x)