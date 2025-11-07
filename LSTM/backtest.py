# backtest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmm_model import HiddenMarkovModelIndicator
import ccxt
import time
      
def download_ohlcv_by_dates(symbol: str, timeframe: str, start_date: str, end_date: str, exchange: ccxt.Exchange):
    all_ohlcv = []

    # convertir les dates en timestamp ms
    since = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

    while since < end_ts:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 30*60*1000  # avancer de 30 min
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df


# def prepare_hmm_df(df: pd.DataFrame, hmm: HiddenMarkovModelIndicator):
#     df_ind = hmm.run(df)
    
#     # s'assurer que l'index est DatetimeIndex
#     if not isinstance(df_ind.index, pd.DatetimeIndex):
#         df_ind.index = pd.to_datetime(df_ind.index, errors='coerce')
    
#     # width_days pour plotting
#     med_step = df_ind.index.to_series().diff().dropna().median()
#     width_days = (med_step / pd.Timedelta(days=1)) * 0.9 if not pd.isna(med_step) else 0.02
    
#     # colonnes pour plotting
#     df_ind['top_prob_value'] = df_ind.get('prob_top', 0.0) * 100
#     df_ind['bottom_prob_value'] = df_ind.get('prob_bottom', 0.0) * 100
    
#     dominant_is_top = df_ind['top_prob_value'] >= df_ind['bottom_prob_value']
#     df_ind['red_height'] = df_ind['top_prob_value'].where(dominant_is_top, 0.0)
#     df_ind['green_height'] = df_ind['bottom_prob_value'].where(~dominant_is_top, 0.0)
    
#     # outlier_line fallback
#     if 'outlier_line' not in df_ind.columns or df_ind['outlier_line'].isna().all():
#         df_ind['signal_strength'] = df_ind[['top_prob_value', 'bottom_prob_value']].max(axis=1)
#         baseline = df_ind['signal_strength'].rolling(hmm.outlier_smoothing).mean()
#         signal_std = df_ind['signal_strength'].rolling(hmm.outlier_smoothing).std()
#         df_ind['outlier_line'] = (baseline + signal_std * hmm.outlier_sensitivity).ewm(span=10).mean()
    
#     df_ind['buy_signal']  = df_ind['bottom_prob_value'] > df_ind['outlier_line']
#     df_ind['sell_signal'] = df_ind['top_prob_value']    > df_ind['outlier_line']

    
#     return df_ind, width_days

def prepare_hmm_df(df: pd.DataFrame, hmm: HiddenMarkovModelIndicator):
    df_ind = hmm.run(df)
    
    # s'assurer que l'index est DatetimeIndex
    if not isinstance(df_ind.index, pd.DatetimeIndex):
        df_ind.index = pd.to_datetime(df_ind.index, errors='coerce')
    
    # width_days pour plotting
    med_step = df_ind.index.to_series().diff().dropna().median()
    width_days = (med_step / pd.Timedelta(days=1)) * 0.9 if not pd.isna(med_step) else 0.02
    
    # Colonnes de probabilité
    df_ind['top_prob_value'] = df_ind.get('prob_top', 0.0) * 100
    df_ind['bottom_prob_value'] = df_ind.get('prob_bottom', 0.0) * 100

    # Probabilité dominante
    dominant_is_top = df_ind['top_prob_value'] >= df_ind['bottom_prob_value']
    df_ind['red_height'] = df_ind['top_prob_value'].where(dominant_is_top, 0.0)
    df_ind['green_height'] = df_ind['bottom_prob_value'].where(~dominant_is_top, 0.0)

    # Calcul du outlier_line si manquant
    if 'outlier_line' not in df_ind.columns or df_ind['outlier_line'].isna().all():
        df_ind['signal_strength'] = df_ind[['top_prob_value', 'bottom_prob_value']].max(axis=1)
        baseline = df_ind['signal_strength'].rolling(hmm.outlier_smoothing).mean()
        signal_std = df_ind['signal_strength'].rolling(hmm.outlier_smoothing).std()
        df_ind['outlier_line'] = (baseline + signal_std * hmm.outlier_sensitivity).ewm(span=10).mean()

    # === LOGIQUE IDENTIQUE AU BACKTEST ===
    df_ind['top_cross_down'] = (
        (df_ind['top_prob_value'].shift(1) <= df_ind['outlier_line'].shift(1)) &
        (df_ind['top_prob_value'] > df_ind['outlier_line'])
    )

    df_ind['bottom_cross_down'] = (
        (df_ind['bottom_prob_value'].shift(1) <= df_ind['outlier_line'].shift(1)) &
        (df_ind['bottom_prob_value'] > df_ind['outlier_line'])
    )

    # Pour compatibilité avec le reste du code live :
    df_ind['buy_signal'] = df_ind['bottom_cross_down']
    df_ind['sell_signal'] = df_ind['top_cross_down']

    return df_ind, width_days

# ------------------- 3. Backtest HMM -------------------
import pandas as pd
import numpy as np
import logging

def backtest_hmm_no_lookahead_sl(df_ind: pd.DataFrame,
                              initial_cash: float = 10_000,
                              slippage: float = 0.0,
                              stop_loss_pct: float = None,
                              stop_loss_points: float = None,
                              trailing_stop_pct: float = None,
                              trailing_stop_points: float = None,
                              allow_close_highlow_fallback: bool = True,
                              return_trades: bool = False) -> pd.DataFrame:
    """
    Backtest HMM sans lookahead, avec stop loss (fixe ou en %) et trailing stop optionnel.
    - stop_loss_pct: ex. 0.02 pour 2% (stop = entry * (1 - 0.02) pour un long).
    - stop_loss_points: stop absolu en unités de prix (ex: 5 points).
    - trailing_stop_pct / trailing_stop_points : si fournis, stop suiveur (monte mais ne descend pas).
    - allow_close_highlow_fallback: si True et pas de high/low, on teste avec min(open,close)/max(open,close).
    - return_trades: si True renvoie (df_ind, trades_list) sinon renvoie df_ind.
    """
    # validations simples
    if stop_loss_pct is not None and stop_loss_points is not None:
        raise ValueError("Choisir stop_loss_pct OU stop_loss_points, pas les deux.")
    if trailing_stop_pct is not None and trailing_stop_points is not None:
        raise ValueError("Choisir trailing_stop_pct OU trailing_stop_points, pas les deux.")

  #  df = df_ind.copy().reset_index(drop=False)  # travail en positionnel plus simple
    df = df_ind.copy().reset_index(drop=True)
    n = len(df)
    cash = initial_cash
    btc = 0.0
    equity_curve = []
    entry_price = None    # prix d'entrée (après slippage)
    stop_price = None
    trades = []
    log = logging.getLogger(__name__)

    # helper pour calculer stop à partir d'un prix d'entrée
    def compute_initial_stop(entry_px):
        if stop_loss_pct is not None:
            return entry_px * (1.0 - float(stop_loss_pct))
        elif stop_loss_points is not None:
            return entry_px - float(stop_loss_points)
        else:
            return None

    def compute_trailing_from_price(px):
        if trailing_stop_pct is not None:
            return px * (1.0 - float(trailing_stop_pct))
        elif trailing_stop_points is not None:
            return px - float(trailing_stop_points)
        else:
            return None

    for i in range(n):
        # valorisation à la close de la barre i
        price_close = df['close'].iloc[i]
        equity_curve.append(cash + btc * price_close)

        # --- 1) Vérification des stops pour position existante pendant la barre i ---
        if btc > 0 and stop_price is not None:
            # récupère high/low si possible
            high_i = df['high'].iloc[i] if 'high' in df.columns else np.nan
            low_i  = df['low'].iloc[i]  if 'low'  in df.columns else np.nan

            used_bounds = True
            if np.isnan(high_i) or np.isnan(low_i):
                if allow_close_highlow_fallback:
                    # fallback: approx avec open/close
                    high_i = max(df['open'].iloc[i], df['close'].iloc[i])
                    low_i  = min(df['open'].iloc[i], df['close'].iloc[i])
                else:
                    used_bounds = False

            if used_bounds and (low_i <= stop_price <= high_i):
                # stop déclenché pendant la barre i
                px_exec = stop_price * (1.0 - slippage)  # vente long
                cash = btc * px_exec
                trades.append({
                    'time': df_ind.index[i + 1],
                    'side': 'stop_sell',
                    'price': stop_price,
                    'qty': btc,
                    'exec_price_after_slippage': px_exec,
                    'bar_index': i
                })
                log.debug("Stop hit on bar %d: stop=%.6f exec=%.6f qty=%.6f", i, stop_price, px_exec, btc)
                btc = 0.0
                entry_price = None
                stop_price = None
                # après stop on continue normalement (les signaux de la barre i seront exécutés à open i+1)

            else:
                # si position toujours ouverte et trailing activé, on peut remonter le stop à la clôture de la barre i
                if trailing_stop_pct is not None or trailing_stop_points is not None:
                    trailing_candidate = compute_trailing_from_price(price_close)
                    if trailing_candidate is not None:
                        # pour long, ne remonte que si trailing_candidate > stop_price
                        if stop_price is None or trailing_candidate > stop_price:
                            old = stop_price
                            stop_price = trailing_candidate
                            if old is None:
                                log.debug("Trailing stop initialisé à %.6f (close=%.6f) on bar %d", stop_price, price_close, i)
                            else:
                                log.debug("Trailing stop bumped %.6f -> %.6f on bar %d", old, stop_price, i)

        # --- 2) Pas d'exécution possible pour la dernière ligne (pas d'open suivant) ---
        if i >= n - 1:
            continue

        # --- 3) Execution des signaux calculés sur la barre i, au open de la barre i+1 ---
        price_exec = df['open'].iloc[i + 1]
        if np.isnan(price_exec):
            continue

        # BUY signal -> entrer au open[i+1]
        if df.get('buy_signal', pd.Series(False)).iloc[i] and cash > 0:
            px_entry = price_exec * (1.0 + slippage)
            btc = cash / px_entry
            cash = 0.0
            entry_price = px_entry
            stop_price = compute_initial_stop(entry_price)
            trades.append({
                'time': df_ind.index[i + 1],
                'side': 'buy',
                'price': price_exec,
                'qty': btc,
                'exec_price_after_slippage': px_entry,
                'bar_index': i + 1
            })
            log.debug("Entry BUY at bar %d open=%.6f entry_after_slippage=%.6f qty=%.6f stop=%.6f",
                      i + 1, price_exec, px_entry, btc, stop_price)

            # vérifie si stop est touché dans la même barre (i+1)
            high_n = df['high'].iloc[i + 1] if 'high' in df.columns else np.nan
            low_n  = df['low'].iloc[i + 1]  if 'low'  in df.columns else np.nan
            used_bounds = True
            if np.isnan(high_n) or np.isnan(low_n):
                if allow_close_highlow_fallback:
                    high_n = max(df['open'].iloc[i + 1], df['close'].iloc[i + 1])
                    low_n  = min(df['open'].iloc[i + 1], df['close'].iloc[i + 1])
                else:
                    used_bounds = False

            if stop_price is not None and used_bounds and (low_n <= stop_price <= high_n):
                # stop déclenché dans la même barre que l'entrée -> sortie au stop
                px_stop_exec = stop_price * (1.0 - slippage)
                cash = btc * px_stop_exec
                trades.append({
                    'time': df_ind.index[i + 1],
                    'side': 'stop_sell_after_entry',
                    'price': stop_price,
                    'qty': btc,
                    'exec_price_after_slippage': px_stop_exec,
                    'bar_index': i + 1
                })
                log.debug("Stop hit same-bar after entry on bar %d: stop=%.6f exec=%.6f qty=%.6f",
                          i + 1, stop_price, px_stop_exec, btc)
                btc = 0.0
                entry_price = None
                stop_price = None
            else:
                # si on reste en position, on peut déjà appliquer trailing sur la base du close de i+1
                if (trailing_stop_pct is not None or trailing_stop_points is not None):
                    trailing_candidate = compute_trailing_from_price(df['close'].iloc[i + 1])
                    if trailing_candidate is not None and (stop_price is None or trailing_candidate > stop_price):
                        old = stop_price
                        stop_price = trailing_candidate
                        log.debug("Trailing stop set/updated after entry: %.6f (old=%.6f) on bar %d",
                                  stop_price, old, i + 1)

        # SELL signal -> fermer au open[i+1] si position existante
        elif df.get('sell_signal', pd.Series(False)).iloc[i] and btc > 0:
            px_close = price_exec * (1.0 - slippage)
            cash = btc * px_close
            trades.append({
                'time': df_ind.index[i + 1],
                'side': 'sell',
                'price': price_exec,
                'qty': btc,
                'exec_price_after_slippage': px_close,
                'bar_index': i + 1
            })
            log.debug("Signal SELL executed at bar %d open=%.6f exec_after_slippage=%.6f qty=%.6f",
                      i + 1, price_exec, px_close, btc)
            btc = 0.0
            entry_price = None
            stop_price = None

    # fin boucle

    df_ind = df_ind.copy()
    df_ind['equity'] = equity_curve

    # Normalisation identique à ton code original (conservée)
    first_exec_price = df_ind['open'].iloc[1] if len(df_ind) > 1 else df_ind['close'].iloc[0]
    df_ind['equity_norm'] = df_ind['equity'] / df_ind['equity'].iloc[0] * first_exec_price

    buy_and_hold_btc = initial_cash / first_exec_price
    df_ind['buy_hold_norm'] = buy_and_hold_btc * df_ind['close']

    
    if return_trades:
        return df_ind, trades
    return df_ind
        




def backtest_hmm_no_lookahead(df_ind: pd.DataFrame,
                              initial_cash: float = 10_000,
                              slippage: float = 0.0) -> pd.DataFrame:
    """
    Backtest HMM simple :
    - Signaux calculés sur la barre i
    - Exécution à l'open de la barre i+1 (pour faire les conditiosn reelles)
    - Slippage exprimé en pourcentage (0.001 = 0.1%)
    """
    df_ind = df_ind.copy()

    # Signaux
    # df_ind['buy_signal']  = df_ind['bottom_prob_value'] > df_ind['outlier_line']
    # df_ind['sell_signal'] = df_ind['top_prob_value']    > df_ind['outlier_line']

    cash = initial_cash
    btc  = 0.0 
    equity_curve = []
    entry_price = None

    for i in range(len(df_ind)):
        # --- Valorisation courante (toujours à la close i pour l’équité) ---
        price_close = df_ind['close'].iloc[i]
        equity_curve.append(cash + btc * price_close)

        # --- Exécution éventuelle : next bar open ---
        # pas de trade possible sur la dernière ligne
        if i >= len(df_ind) - 1:
            continue

        price_exec = df_ind['open'].iloc[i + 1]     # prix exécution = open de la prochaine barre
        if np.isnan(price_exec):
            continue

        if df_ind['buy_signal'].iloc[i] and cash > 0:
            px = price_exec * (1 + slippage)
            btc = cash / px
            cash = 0.0

        elif df_ind['sell_signal'].iloc[i] and btc > 0:
            px = price_exec * (1 - slippage)
            cash = btc * px
            btc = 0.0

    df_ind['equity'] = equity_curve

    # Normalisation identique à ton code original
    first_exec_price = df_ind['open'].iloc[1] if len(df_ind) > 1 else df_ind['close'].iloc[0]
    df_ind['equity_norm'] = df_ind['equity'] / df_ind['equity'].iloc[0] * first_exec_price

    # Buy & Hold normalisé (entrée au même premier prix d’exécution)
    buy_and_hold_btc = initial_cash / first_exec_price
    df_ind['buy_hold_norm'] = buy_and_hold_btc * df_ind['close']

    return df_ind

def backtest_hmm_nn_combined(df_ind: pd.DataFrame,
                             initial_cash: float = 10_000,
                             slippage: float = 0.01,
                             threshold_buy: float = 0.45,
                             threshold_sell: float = 0.45) -> pd.DataFrame:
    
    df_ind = df_ind.copy()
    
    # --- Signaux HMM ---

    df_ind['top_cross_down'] = (
        (df_ind['top_prob_value'].shift(1) <= df_ind['outlier_line'].shift(1)) &
        (df_ind['top_prob_value'] > df_ind['outlier_line'])
    )

    df_ind['bottom_cross_down'] = (
        (df_ind['bottom_prob_value'].shift(1) <= df_ind['outlier_line'].shift(1)) &
        (df_ind['bottom_prob_value'] > df_ind['outlier_line'])
    )

    # Pour compatibilité avec le reste du code live :
    df_ind['buy_signal'] = df_ind['bottom_cross_down']
    df_ind['sell_signal'] = df_ind['top_cross_down']
    # df_ind['buy_signal']  = df_ind['bottom_prob_value'] > df_ind['outlier_line']
    # df_ind['sell_signal'] = df_ind['top_prob_value']    > df_ind['outlier_line']
    
    # --- Signaux combinés HMM + NN ---
    df_ind['combined_buy']  = (df_ind['buy_signal']  & (df_ind['pred_prob_up'] > threshold_buy)).astype(bool)
    df_ind['combined_sell'] = (df_ind['sell_signal'] & (df_ind['pred_prob_up'] < threshold_sell)).astype(bool)
    
    # --- Backtest ---
    cash = initial_cash
    btc = 0.0
    equity_curve = []


    for i in range(len(df_ind)):
        price_close = df_ind['close'].iloc[i]
        equity_curve.append(cash + btc * price_close)

        if i >= len(df_ind) - 1:
            continue

        price_exec = df_ind['open'].iloc[i + 1]
        if np.isnan(price_exec):
            continue
        # Exécution des trades
        if df_ind['combined_buy'].iloc[i] and cash > 0:
            px = price_exec * (1 + slippage)
            btc = cash / px
            cash = 0.0
        elif df_ind['combined_sell'].iloc[i] and btc > 0:
            px = price_exec * (1 - slippage)
            cash = btc * px
            btc = 0.0

    df_ind['equity'] = equity_curve

    # Normalisation
    first_exec_price = df_ind['open'].iloc[1] if len(df_ind) > 1 else df_ind['close'].iloc[0]
    df_ind['equity_norm'] = df_ind['equity'] / df_ind['equity'].iloc[0] * first_exec_price
    buy_and_hold_btc = initial_cash / first_exec_price
    df_ind['buy_hold_norm'] = buy_and_hold_btc * df_ind['close']

    return df_ind

import numpy as np
import pandas as pd

def backtest_hmm_nn_leveraged(df_ind: pd.DataFrame,
                                        initial_cash: float = 10_000,
                                        threshold_buy: float = 0.45,
                                        threshold_sell: float = 0.45,
                                        pnl_per_1pct: float = 200.0) -> pd.DataFrame:
    """
    Backtest combinant signaux HMM et NN avec effet de levier simulé (long only).
    Chaque variation de ±1 % du prix engendre un gain ou une perte fixe (pnl_per_1pct).
    """

    df_ind = df_ind.copy()

    # --- Signaux HMM ---
    df_ind['top_cross_down'] = (
        (df_ind['top_prob_value'].shift(1) <= df_ind['outlier_line'].shift(1)) &
        (df_ind['top_prob_value'] > df_ind['outlier_line'])
    )
    df_ind['bottom_cross_down'] = (
        (df_ind['bottom_prob_value'].shift(1) <= df_ind['outlier_line'].shift(1)) &
        (df_ind['bottom_prob_value'] > df_ind['outlier_line'])
    )

    df_ind['buy_signal'] = df_ind['bottom_cross_down']
    df_ind['sell_signal'] = df_ind['top_cross_down']

    # --- Signaux combinés HMM + NN ---
    df_ind['combined_buy']  = (df_ind['buy_signal']  & (df_ind['pred_prob_up'] > threshold_buy)).astype(bool)
    df_ind['combined_sell'] = (df_ind['sell_signal'] & (df_ind['pred_prob_up'] < threshold_sell)).astype(bool)

    # --- Backtest effet de levier (long only) ---
    cash = initial_cash
    position_open = False
    entry_price = None
    equity_curve = []

    for i in range(len(df_ind)):
        price = df_ind['close'].iloc[i]

        # Calcul du P&L latent
        if position_open and entry_price is not None:
            pct_change = (price - entry_price) / entry_price
            pnl = (pct_change / 0.01) * pnl_per_1pct
            equity_curve.append(cash + pnl)
        else:
            equity_curve.append(cash)

        # Gestion des signaux
        if i >= len(df_ind) - 1:
            continue

        # Signal d'achat → ouvrir une position si aucune ouverte
        if df_ind['combined_buy'].iloc[i] and not position_open:
            position_open = True
            entry_price = price

        # Signal de vente → fermer la position si ouverte
        elif df_ind['combined_sell'].iloc[i] and position_open:
            # On fige le P&L au moment de la vente
            pct_change = (price - entry_price) / entry_price
            pnl = (pct_change / 0.01) * pnl_per_1pct
            cash += pnl
            position_open = False
            entry_price = None

    df_ind['equity'] = equity_curve

    # Normalisation pour affichage
    df_ind['equity_norm'] = df_ind['equity'] / df_ind['equity'].iloc[0] * df_ind['close'].iloc[0]
    df_ind['buy_hold_norm'] = (initial_cash / df_ind['close'].iloc[0]) * df_ind['close']

    return df_ind

# ------------------- 4. Tracer -------------------
def plot_hmm(df_ind, symbol, timeframe, days, width_days):
    x = mdates.date2num(df_ind.index.to_pydatetime()) #o())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8), gridspec_kw={'height_ratios':[2,1]})
    
    # axe prix + equity
    ax1.plot(df_ind.index, df_ind['close'], linewidth=1.0, label='VIX index Close', color='black')
    ax1.plot(df_ind.index, df_ind['equity_norm'], color='blue', label='HMM Strategy Equity')
    if 'top_cross_down' in df_ind.columns:
        cd_top = df_ind.index[df_ind['top_cross_down'] == True]
        if len(cd_top): ax2.scatter(cd_top, df_ind.loc[cd_top, 'top_prob_value'], marker='v', s=30, color='red', label='Top cross-down')
    if 'bottom_cross_down' in df_ind.columns:
        cd_bot = df_ind.index[df_ind['bottom_cross_down'] == True]
        if len(cd_bot): ax2.scatter(cd_bot, df_ind.loc[cd_bot, 'bottom_prob_value'], marker='^', s=30, color='green', label='Bottom cross-down')
    ax1.scatter(df_ind.index[df_ind['buy_signal']], df_ind.loc[df_ind.index[df_ind['buy_signal']], 'close'], color='green', marker='^', s=60, label='Buy')
    ax1.scatter(df_ind.index[df_ind['sell_signal']], df_ind.loc[df_ind.index[df_ind['sell_signal']], 'close'], color='red', marker='v', s=60, label='Sell')
    ax1.set_title(f'{symbol} {timeframe} —  {days} days — HMM signals')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # axe probabilités
    ax2.bar(x, df_ind['green_height'], width=width_days, alpha=0.25, label='Prob bottom (green)')
    ax2.bar(x, df_ind['red_height'], width=width_days, alpha=0.25, label='Prob top (red)')
    ax2.plot(df_ind.index, df_ind['outlier_line'], linestyle='--', linewidth=1.0, label='Outlier line')
    
    ax2.set_ylim(0, max(100, float(np.nanmax(df_ind[['top_prob_value','bottom_prob_value']].values))*1.1))
    ax2.set_ylabel('Probability (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', ncol=2)
    
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    plt.show()
    
def plot_drawdown(df_ind, equity_col='equity', title='Drawdown'):
    equity = df_ind[equity_col].values
    roll_max = np.maximum.accumulate(equity)
    drawdown = (roll_max - equity) / roll_max * 100  # en %
    
    plt.figure(figsize=(14,4))
    plt.fill_between(df_ind.index, drawdown, 0, color='red', alpha=0.4)
    plt.plot(df_ind.index, drawdown, color='red', linewidth=1)
    plt.title(title)
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return drawdown

def plot_market_states(df_ind):
    plt.figure(figsize=(14,3))
    colors = {"top":"red", "normal":"gray", "bottom":"green"}
    plt.scatter(df_ind.index, [1]*len(df_ind), 
                c=df_ind['market_state'].map(colors), 
                s=10, marker="s")
    plt.title("Market states (Top / Normal / Bottom)")
    plt.yticks([])
    plt.grid(True, alpha=0.3)
    plt.show()





def evaluate_strategy(df_ind, equity_col='equity', trading_days=252):
    equity = df_ind[equity_col].values
    returns = np.diff(equity) / equity[:-1]  # rendements journaliers
    total_return = (equity[-1] / equity[0]) - 1

    # CAGR (rendement annualisé)
    n_years = (df_ind.index[-1] - df_ind.index[0]).days / 365.25
    cagr = (equity[-1] / equity[0])**(1/n_years) - 1

    # Volatilité annualisée
    vol_annual = np.std(returns) * np.sqrt(trading_days)

    # Sharpe ratio (taux sans risque = 4 % annuel)
    sharpe = (cagr - 0.04) / vol_annual if vol_annual > 0 else np.nan

    # Drawdown
    roll_max = np.maximum.accumulate(equity)
    drawdown = (roll_max - equity) / roll_max
    max_dd = drawdown.max()
    avg_dd = drawdown.mean()

    buy_signals = df_ind.get('buy_signal', pd.Series()).sum() if 'buy_signal' in df_ind.columns else 0
    sell_signals = df_ind.get('sell_signal', pd.Series()).sum() if 'sell_signal' in df_ind.columns else 0


    return {
        "total_return": total_return,
        "rendement annuel": cagr,
        "volatilitey annual": vol_annual,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "Buy Signals": buy_signals,
        "Sell Signals": sell_signals
    }

def trade_distribution(df_ind, price_col='open'):
    """
    Retourne un DataFrame avec tous les trades et leurs stats:
      - start_time, end_time
      - duration (en bars)
      - entry_price, exit_price
      - pct_return
    Exécution au prochain open pour éviter le look-ahead.
    """
    trades = []
    position = 0  # 0 = pas de position, 1 = long
    entry_price = None
    entry_time = None
    
    n = len(df_ind)
    
    for i in range(n - 1):  # stop à n-1 car on trade à l'open suivant
        exec_price = df_ind[price_col].iloc[i + 1]  # prix d'exécution au prochain open
        exec_time = df_ind.index[i + 1]
        
        # ouverture trade
        if df_ind['buy_signal'].iloc[i] and position == 0:
            position = 1
            entry_price = exec_price
            entry_time = exec_time
        
        # fermeture trade
        elif df_ind['sell_signal'].iloc[i] and position == 1:
            exit_price = exec_price
            exit_time = exec_time
            pct_return = (exit_price - entry_price) / entry_price * 100
            
            trades.append({
                'start_time': entry_time,
                'end_time': exit_time,
                'duration_bars': (i + 1 - df_ind.index.get_loc(entry_time)),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pct_return': pct_return
            })
            
            position = 0
            entry_price = None
            entry_time = None
    
    trades_df = pd.DataFrame(trades)
    return trades_df


def plot_daily_returns(df_ind, equity_col='equity', title='Daily Returns (%)'):
    """
    Calcule et trace les retours quotidiens de la stratégie.

    Paramètres :
    - df_ind : DataFrame contenant la colonne equity ou equity_norm
    - equity_col : colonne de valorisation de la stratégie
    - title : titre du graphique
    """
    # Calcul des retours quotidiens en %
    daily_ret = df_ind[equity_col].pct_change() * 100
    daily_ret = daily_ret.fillna(0)

    # Tracé
    plt.figure(figsize=(14,4))
    plt.bar(df_ind.index, daily_ret, color='darkblue', alpha=0.9, width=1)
    plt.axhline(0, color='black', linewidth=1)
    plt.title(title)
    plt.ylabel("Return (%)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return daily_ret
