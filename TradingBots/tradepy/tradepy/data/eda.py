"""
plot_cleaning.py
----------------
Plots para diagnóstico y limpieza de datos históricos de futuros.

6 plots:
    1. Histograma de velas por hora
    2. Heatmap de velas por día y hora
    3. Timeline de gaps entre velas consecutivas
    4. Distribución de volumen con outliers marcados
    5. Precio (close) completo a lo largo del tiempo
    6. Boxplot de close por año

Uso:
    from tradepy.data.plot_cleaning import plot_cleaning

    df, spec = load_future('ES')
    plot_cleaning(df, ticker='ES', show=True, save=False)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from tradepy.paths import DATA_DIR

log = logging.getLogger(__name__)

# ── Estilo ────────────────────────────────────────────────────────────────────
DAYS_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
PALETTE    = {
    'primary':   '#1a508b',
    'accent':    '#e84545',
    'secondary': '#f5a623',
    'neutral':   '#6c757d',
    'bg':        '#f8f9fa',
}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara una copia del df con columnas auxiliares para los plots."""
    d = df.copy()
    if 'datetime' in d.columns:
        d['datetime'] = pd.to_datetime(d['datetime'])
        d = d.set_index('datetime')
    else:
        d.index = pd.to_datetime(d.index)

    d['hour']     = d.index.hour
    d['day_name'] = d.index.day_name()
    d['year']     = d.index.year
    d['gap_mins'] = d.index.to_series().diff().dt.total_seconds().div(60)
    if 'datetime_utc' in d.columns:
        d["hour_utc"] = pd.to_datetime(d["datetime_utc"]).dt.hour

    return d


# ── Plots individuales ────────────────────────────────────────────────────────

def _plot_hourly_bars(ax: plt.Axes, d: pd.DataFrame, column = "hour") -> None:
    """1. Histograma: número de velas por hora."""
    counts = d.groupby(column).size()
    ax.bar(counts.index, counts.values, color=PALETTE['primary'], alpha=0.8, width=0.7)
    ax.set_title("Velas por hora", fontweight='bold')
    ax.set_xlabel("Hora UTC")
    ax.set_ylabel("Nº velas")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    # Marcar horas con muy pocas velas (< 10% de la mediana)
    threshold = counts.median() * 0.1
    sparse = counts[counts < threshold]
    for h in sparse.index:
        ax.axvline(h, color=PALETTE['accent'], linewidth=1.2, linestyle='--', alpha=0.7)


def _plot_heatmap(ax: plt.Axes, d: pd.DataFrame, column = "hour") -> None:
    """2. Heatmap: velas por día de semana y hora."""
    hmap = (
        d.groupby(['day_name', column])
        .size()
        .unstack(fill_value=0)
        .reindex([day for day in DAYS_ORDER if day in d['day_name'].unique()])
    )
    sns.heatmap(
        hmap, ax=ax, cmap='YlOrRd',
        linewidths=0.3, linecolor='white',
        cbar_kws={'label': 'Nº velas'},
        annot=False,
    )
    ax.set_title("Densidad de velas por día/hora", fontweight='bold')
    ax.set_xlabel("Hora UTC")
    ax.set_ylabel("")


def _plot_gaps(ax: plt.Axes, d: pd.DataFrame) -> None:
    """3. Timeline de gaps entre velas consecutivas (en minutos)."""
    gaps = d['gap_mins'].dropna()

    # Solo gaps > 1 min (saltos reales, no el gap de 1 min normal)
    anomalous = gaps[gaps > 2]

    ax.scatter(
        anomalous.index, anomalous.values,
        color=PALETTE['accent'], s=6, alpha=0.6, linewidths=0,
    )
    ax.set_title("Gaps entre velas consecutivas (> 2 min)", fontweight='bold')
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Minutos")
    ax.set_yscale('log')

    # Líneas de referencia
    for ref, label, color in [
        (60,   '1h',  PALETTE['secondary']),
        (1440, '1d',  PALETTE['neutral']),
        (4320, '3d',  PALETTE['accent']),
    ]:
        ax.axhline(ref, linestyle='--', linewidth=0.9, color=color, alpha=0.8, label=label)
    ax.legend(fontsize=8)


def _plot_volume(ax: plt.Axes, d: pd.DataFrame) -> None:
    """4. Distribución de volumen con percentiles 99 y 99.9 marcados."""
    vol = d['volume'].dropna()
    p99  = vol.quantile(0.99)
    p999 = vol.quantile(0.999)

    ax.hist(
        vol.clip(upper=p999 * 1.1),
        bins=80, color=PALETTE['primary'], alpha=0.75, edgecolor='none',
    )
    ax.axvline(p99,  color=PALETTE['secondary'], linewidth=1.5,
               linestyle='--', label=f'P99  = {p99:,.0f}')
    ax.axvline(p999, color=PALETTE['accent'],    linewidth=1.5,
               linestyle='--', label=f'P99.9 = {p999:,.0f}')
    ax.set_title("Distribución de volumen", fontweight='bold')
    ax.set_xlabel("Volumen")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=8)


def _plot_price(ax: plt.Axes, d: pd.DataFrame) -> None:
    """5. Precio (close) completo a lo largo del tiempo."""
    ax.plot(d.index, d['close'], color=PALETTE['primary'], linewidth=0.6, alpha=0.9)

    # Detectar saltos de precio > 5 desviaciones estándar
    ret   = d['close'].pct_change().abs()
    spikes = ret[ret > ret.mean() + 5 * ret.std()]
    ax.scatter(
        spikes.index, d.loc[spikes.index, 'close'],
        color=PALETTE['accent'], s=12, zorder=5, label=f'{len(spikes)} spikes',
    )
    ax.set_title("Precio (close) completo — spikes marcados", fontweight='bold')
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    if len(spikes):
        ax.legend(fontsize=8)


def _plot_yearly_boxplot(ax: plt.Axes, d: pd.DataFrame) -> None:
    """6. Boxplot de close por año — detecta datos fuera de rango por periodo."""
    years  = sorted(d['year'].unique())
    data   = [d.loc[d['year'] == y, 'close'].dropna().values for y in years]

    bp = ax.boxplot(
        data,
        labels=years,
        patch_artist=True,
        medianprops=dict(color=PALETTE['accent'], linewidth=1.5),
        boxprops=dict(facecolor=PALETTE['primary'], alpha=0.4),
        whiskerprops=dict(color=PALETTE['neutral']),
        capprops=dict(color=PALETTE['neutral']),
        flierprops=dict(marker='.', color=PALETTE['accent'], alpha=0.3, markersize=3),
    )
    ax.set_title("Distribución de precio por año", fontweight='bold')
    ax.set_xlabel("Año")
    ax.set_ylabel("Close")
    ax.tick_params(axis='x', rotation=45)

def _plot_sessions(ax, d: pd.DataFrame, gap_threshold=60):
    """
    Visualiza cambios de sesión detectados vs gaps reales
    """
    gap_mins = d.index.to_series().diff().dt.total_seconds() / 60
    gaps = gap_mins.dropna()

    # Detectar sesiones
    session_change = gaps > gap_threshold

    ax.plot(gaps.index, gaps.values, color='gray', alpha=0.4, linewidth=0.8)

    # Marcar gaps grandes
    ax.scatter(
        gaps.index[session_change],
        gaps.values[session_change],
        color='red',
        s=10,
        label='Session change (gap)'
    )

    ax.set_yscale('log')
    ax.set_title(f"Detección de sesiones (gap > {gap_threshold} min)", fontweight='bold')
    ax.set_ylabel("Gap (min)")
    ax.legend()
    
# ── Función principal ─────────────────────────────────────────────────────────

log = logging.getLogger(__name__)


def plot_cleaning(
    df: pd.DataFrame,
    ticker: str = "Future",
    show: bool = True,
    save: bool = False,
) -> None:
    """
    Genera plots de diagnóstico de limpieza de datos.

    Incluye:
        - Velas por hora
        - Heatmap día/hora
        - Precio completo
        - Gaps
        - Sesiones detectadas
        - Volumen
        - Boxplot anual
    """

    d = _prep(df)

    fig = plt.figure(figsize=(18, 28))
    fig.suptitle(
        f"Diagnóstico de limpieza — {ticker.upper()}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        5, 2,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        top=0.95,
        bottom=0.05,
        left=0.07,
        right=0.97,
    )

    # ── Row 0
    _plot_hourly_bars(fig.add_subplot(gs[0, 0]), d)
    _plot_heatmap(fig.add_subplot(gs[0, 1]), d)

    # ── Row 1
    _plot_price(fig.add_subplot(gs[1, :]), d)

    # ── Row 2
    _plot_gaps(fig.add_subplot(gs[2, :]), d)

    # ── Row 3
    _plot_sessions(fig.add_subplot(gs[3, :]), d)

    # ── Row 4
    _plot_volume(fig.add_subplot(gs[4, 0]), d)
    _plot_yearly_boxplot(fig.add_subplot(gs[4, 1]), d)

    if save:
        out = DATA_DIR / f"{ticker.lower()}_cleaning_diag.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info(f"[{ticker}] Guardado en {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        
def plot_comparation_utc(
    df: pd.DataFrame,
    ticker: str = "Future",
    show: bool = True,
    save: bool = False,
) -> None:
    
    d = _prep(df)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Diagnóstico de limpieza — {ticker.upper()}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        top=0.95,
        bottom=0.05,
        left=0.07,
        right=0.97,
    )

    # ── Row 0
    _plot_hourly_bars(fig.add_subplot(gs[0, 0]), d)
    _plot_heatmap(fig.add_subplot(gs[0, 1]), d)
    
    # ── Row 1
    _plot_hourly_bars(fig.add_subplot(gs[1, 0]), d, column = "hour_utc")
    _plot_heatmap(fig.add_subplot(gs[1, 1]), d, column = "hour_utc")
