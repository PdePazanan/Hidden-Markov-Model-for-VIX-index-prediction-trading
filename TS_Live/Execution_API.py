# Execution_API.py
import time
import requests
import pandas as pd
from tqdm import tqdm
from hmm_model import HiddenMarkovModelIndicator
from datetime import datetime, timedelta, timezone, UTC
# === TradeStation API Credentials ===
CLIENT_ID = "3bE5Y8Qua6OHghAGOHmK0JQ0aMYz8QgC"
CLIENT_SECRET = "SviP-D-LEapA9Ou1fjlQTK8gZZ_t998C5jMlzWfVJ_A7iUNLIEh8qyr7QyjUfCp3"
REFRESH_TOKEN = "L1EZpMoAXU4pe0XOD-iCKFj9IBhokalIFCWztWQACraKv"
TOKEN_FILE = "tokens.json"
ACCOUNT_ID="SIM903218F"

# ====== Data Params ======
DEFAULT_SYMBOL = "$VIX.X"
DEFAULT_INTERVAL = 5   # minutes
DEFAULT_UNIT = "Minute"
DEFAULT_LOOKBACK_DAYS = 20
CHUNK_SIZE = 300        # nb de barres max par requête

# ====== Trading Parameters ======
TRADE_QTY="1"
FUTURE = "VXX25"
BASE_URL = "https://sim-api.tradestation.com/v3"


def refresh_access_token(client_id, client_secret, refresh_token):
    """Obtient un access_token à partir du refresh_token"""
    url = "https://signin.tradestation.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
      #  "redirect_uri": "https://sim-api.tradestation.com"
    }

    response = requests.post(url, data=data)
    if response.status_code != 200:
        raise Exception(f"Erreur refresh token: {response.status_code}, {response.text}")

    token_info = response.json()
    return token_info["access_token"]




#=== Fetch Bars with Pagination ===
def fetch_bars(symbol,
               access_token,
               unit="Minute",
               interval=15,
               start_utc=None,
               end_utc=None,
               chunk_size=CHUNK_SIZE):
    """
    Télécharge les barres OHLCV de TradeStation avec pagination.
    Retourne un DataFrame indexé par datetime UTC.
    """

    url = f"{BASE_URL}/marketdata/barcharts/{symbol}"
    headers = {"Authorization": f"Bearer {access_token}"}
    all_bars = []

    if end_utc is None:
        end_utc = datetime.now(timezone.utc)
    if start_utc is None:
        start_utc = end_utc - timedelta(days=30)

    total_span = (end_utc - start_utc).total_seconds() / 3600

    cursor = end_utc
    i = 0
    print(f" Téléchargement des barres pour {symbol} ({interval}{unit})")
    print(f" Période demandée : {start_utc:%Y-%m-%d %H:%M} → {end_utc:%Y-%m-%d %H:%M}\n")
    progress = tqdm(total=total_span, desc=f"Téléchargement {symbol}", unit="h")

    while cursor > start_utc:
        params = {
            "unit": unit,
            "interval": interval,
            "barsback": chunk_size,
            "startdate": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        i += 1
        print(f"🔹 [{i}] Requête API : {params['startdate']} ...", flush=True)

        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 504:
            print(f"Timeout 504 à {cursor}, on réduit le chunk_size.")
            chunk_size = max(1000, chunk_size // 2)
            continue
        r.raise_for_status()
        data = r.json()

        bars = data.get("Bars") or data.get("bars") or []
        if not bars:
            break

        chunk = []
        for b in bars:
            dt = pd.to_datetime(b["TimeStamp"], utc=True)
            chunk.append({
                "datetime": dt,
                "open": float(b["Open"]),
                "high": float(b["High"]),
                "low": float(b["Low"]),
                "close": float(b["Close"]),
                "volume": int(b.get("Volume", 0)),
            })
        all_bars.extend(chunk)

        # nouveau curseur = plus ancienne barre - 1 seconde
        cursor = min(c["datetime"] for c in chunk) - timedelta(seconds=1)

        covered = (end_utc - cursor).total_seconds() / 3600
        progress.n = min(covered, total_span)
        progress.refresh()

        if len(chunk) < chunk_size:
            break
        
    progress.close()

    df = pd.DataFrame(all_bars)
    if df.empty:
        return df

    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    return df

def fetch_recent_bars(symbol,
                      access_token,
                      unit="Minute",
                      interval=5,
                      barsback=100):
    """
    Récupère uniquement les dernières barres récentes (live update).
    Utilise barsback sans startdate pour ne pas surcharger l'API.
    """
    url = f"{BASE_URL}/marketdata/barcharts/{symbol}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "unit": unit,
        "interval": interval,
        "barsback": barsback,
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Erreur HTTP fetch_recent_bars: {e}")
        return pd.DataFrame()

    data = r.json()
    bars = data.get("Bars") or data.get("bars") or []
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "datetime": pd.to_datetime(b["TimeStamp"], utc=True),
        "open": float(b["Open"]),
        "high": float(b["High"]),
        "low": float(b["Low"]),
        "close": float(b["Close"]),
        "volume": int(b.get("Volume", 0)),
    } for b in bars])

    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    return df


# fonction qui marchait de base mais bloque a cause du vix 
def fetch_bars_init(symbol,
               access_token,
               unit="Minute",
               interval=5,
               start_utc=None,
               end_utc=None,
               chunk_size=CHUNK_SIZE):

    url = f"{BASE_URL}/marketdata/barcharts/{symbol}"
    headers = {"Authorization": f"Bearer {access_token}"}
    all_bars = []

    if end_utc is None:
        end_utc = datetime.now(timezone.utc)
    if start_utc is None:
        start_utc = end_utc - timedelta(days=30)

    cursor = end_utc

    while cursor > start_utc:
        params = {
            "unit": unit,
            "interval": interval,
            "barsback": chunk_size,
            "startdate": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 504:
            print(f"⚠️ Timeout 504 à {cursor}.")
            chunk_size = max(1000, chunk_size // 2)
            continue
        r.raise_for_status()
        data = r.json()

        bars = data.get("Bars") or data.get("bars") or []
        if not bars:
            break

        chunk = []
        for b in bars:
            dt = pd.to_datetime(b["TimeStamp"], utc=True)
            chunk.append({
                "datetime": dt,
                "open": float(b["Open"]),
                "high": float(b["High"]),
                "low": float(b["Low"]),
                "close": float(b["Close"]),
                "volume": int(b.get("Volume", 0)),
            })
        all_bars.extend(chunk)

        # nouveau curseur = plus ancienne barre - 1 seconde
        cursor = min(c["datetime"] for c in chunk) - timedelta(seconds=1)

        if len(chunk) < chunk_size:
            break

    df = pd.DataFrame(all_bars)
    if df.empty:
        return df

    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    return df



def get_quote(access_token, symbol):
    """Récupère le dernier bid/ask pour un symbole futures"""
    url = f"{BASE_URL}/data/quote/{symbol}"
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Erreur quote: {resp.status_code} -> {resp.text}")
    data = resp.json()
    
    quote = data[0]
    bid = float(quote["Bid"])
    ask = float(quote["Ask"])
    last = float(quote["Last"])
    return bid, ask,last

def buy_position(access_token, account_id, symbol, side, qty):
    bid, ask,last = get_quote(access_token, symbol)
    print(bid,ask)

    url = f"{BASE_URL}/orderexecution/orders"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "AccountID": account_id,
        "Symbol": symbol,
        "Quantity": qty,
        "OrderType": "Limit",
        "LimitPrice": str(bid),  # Ex: "20.5"
        "TradeAction":"Buy",
        "TimeInForce": {"Duration": "Day"},
        "Route": "Intelligent_Future"
    }

    resp = requests.post(url, json=payload, headers=headers)
    print("<<< Réponse :", resp.status_code, resp.text)
    if resp.status_code not in (200, 201):
        raise Exception(f"Erreur API: {resp.status_code} -> {resp.text}")
    return resp.json()

def sell_position(access_token, account_id, symbol, side, qty, limit_price=None):
    bid,ask, last=get_quote(access_token,symbol)
    limit_price = bid

    url = f"{BASE_URL}/orderexecution/orders"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "AccountID": account_id,
        "Symbol": symbol,
        "Quantity": qty,
        "OrderType": "Limit",
        "LimitPrice": str(limit_price),
        "TradeAction": "Sell", 
        "TimeInForce": {"Duration": "Day"},
        "Route": "Intelligent_Future"
    }
    resp = requests.post(url, json=payload, headers=headers)
    print("<<< Réponse :", resp.status_code, resp.text)
    if resp.status_code not in (200, 201):
        raise Exception(f"Erreur API: {resp.status_code} -> {resp.text}")
    return resp.json()



def get_positions(access_token, account_id):
    """
    get the details of the opened positions of the tradestation account
    """
    url = f"{BASE_URL}/brokerage/accounts/{account_id}/positions"
    headers = {"Authorization": f"Bearer {access_token}"  } #  "Content-Type": "application/json"
    resp = requests.request("GET", url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Erreur récupération positions: {resp.status_code} -> {resp.text}")
    positions = resp.json()  # liste de positions
    return positions



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
