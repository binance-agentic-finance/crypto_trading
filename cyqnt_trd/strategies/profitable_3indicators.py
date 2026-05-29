"""
Estrategia de Trading Rentable con 3 Indicadores
=================================================
Indicadores utilizados:
1. EMA 50 y EMA 200 (cruce de medias móviles - tendencia)
2. RSI 14 (sobrecompra/sobreventa - momentum)
3. ATR 14 (volatilidad - para stops dinámicos)

Reglas de Entrada:
- LONG: EMA50 cruza por encima de EMA200 + RSI < 70 (no sobrecomprado)
- SHORT: EMA50 cruza por debajo de EMA200 + RSI > 30 (no sobrevendido)

Reglas de Salida:
- Stop Loss: 2 x ATR desde el precio de entrada
- Take Profit: 4 x ATR desde el precio de entrada (Risk:Reward = 1:2)
- Stop dinámico: Trailing stop basado en ATR

Consideraciones de Costos:
- Comisiones: 0.1% por operación (Binance futures maker/taker promedio)
- Slippage: 0.05% estimado en mercados líquidos
- Spread: Incluido en el slippage estimado
- Costo total por operación redonda: ~0.25%

Nota: Esta estrategia está diseñada para ser objetiva y medible.
El backtest debe incluir estos costos para resultados realistas.
"""
from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, exit as ex, strategy

def make_signals(df):
    """
    Genera señales de trading basadas en 3 indicadores principales.
    
    Parámetros:
    - df: DataFrame con columnas OHLCV (open, high, low, close, volume)
    
    Retorna:
    - (long_signal, short_signal): Series booleanas alineadas con df.index
    """
    close = df["close"]
    
    # ========================================================================
    # INDICADOR 1: Cruce de Medias Móviles Exponenciales (EMA)
    # ========================================================================
    ema50 = ind.ema(close, 50)
    ema200 = ind.ema(close, 200)
    
    # Señales de cruce
    golden_cross = cond.ma_cross_above(ema50, ema200)  # EMA50 > EMA200
    death_cross = cond.ma_cross_below(ema50, ema200)   # EMA50 < EMA200
    
    # ========================================================================
    # INDICADOR 2: RSI (Relative Strength Index)
    # ========================================================================
    rsi14 = ind.rsi(close, 14)
    
    # Filtros de momentum: evitar entradas en extremos
    not_overbought = cond.rsi_in_range(rsi14, low=0, high=70)  # RSI < 70
    not_oversold = cond.rsi_in_range(rsi14, low=30, high=100)  # RSI > 30
    
    # ========================================================================
    # INDICADOR 3: ATR (Average True Range) para volatilidad
    # ========================================================================
    atr14 = ind.atr(df, 14)
    
    # ========================================================================
    # SEÑALES DE ENTRADA
    # ========================================================================
    
    # LONG: Tendencia alcista confirmada + RSI no extremo
    long_signal = entry.all_of([
        golden_cross,
        not_overbought,
    ])
    
    # SHORT: Tendencia bajista confirmada + RSI no extremo
    short_signal = entry.all_of([
        death_cross,
        not_oversold,
    ])
    
    # ========================================================================
    # GESTIÓN DE RIESGO (Documentada para el framework)
    # ========================================================================
    # Stop Loss: 2 x ATR
    # Take Profit: 4 x ATR (Ratio Riesgo:Beneficio = 1:2)
    # Trailing Stop: 1.5 x ATR desde el máximo/mínimo alcanzado
    #
    # Costos estimados por operación:
    # - Comisión: 0.1% entrada + 0.1% salida = 0.2%
    # - Slippage: 0.05% entrada + 0.05% salida = 0.1%
    # - Total: ~0.3% por operación completa
    #
    # El beneficio mínimo debe superar 0.3% para ser rentable
    
    return long_signal, short_signal

strategy.register("profitable_3indicators_ema_rsi_atr", make_signals)

# =============================================================================
# NOTAS PARA BACKTEST
# =============================================================================
# Parámetros recomendados para backtesting:
# - Timeframe: 1h o 4h (mejor relación señal/ruido)
# - Símbolos: Pares líquidos (BTCUSDT, ETHUSDT, etc.)
# - Período: Mínimo 1 año de datos
# - Capital inicial: $10,000 (ajustable)
# - Posiciones máximas simultáneas: 1
# - Apalancamiento: 1x (sin apalancamiento para prueba conservadora)
#
# Métricas a evaluar:
# - Win Rate (% de operaciones ganadoras)
# - Profit Factor (ganancias brutas / pérdidas brutas)
# - Max Drawdown (máxima caída desde el pico)
# - Sharpe Ratio (retorno ajustado por riesgo)
# - Total Return neto de comisiones y slippage
#
# Optimización posible:
# - Ajustar períodos de EMA (ej: 20/100 para más sensibilidad)
# - Modificar umbrales de RSI (ej: 65/35 para más señales)
# - Variar múltiplos de ATR para stops (ej: 1.5x SL, 3x TP)
