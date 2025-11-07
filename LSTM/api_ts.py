# api_ts.py
import time
import sys
from tqdm import tqdm
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone, UTC
from config import CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN, SIM_MODE, CHUNK_SIZE, DEFAULT_SYMBOL,DEFAULT_UNIT,DEFAULT_INTERVAL,DEFAULT_LOOKBACK_DAYS

# === API endpoints ===
SIM_BASE = "https://sim-api.tradestation.com/v3"
LIVE_BASE = "https://api.tradestation.com/v3"
TOKEN_URL = "https://signin.tradestation.com/oauth/token"

# Choix de la base en fonction du mode
BASE_URL = SIM_BASE 

SYMBOLTOD="$VVIX.X"


# === Refresh Access Token ===
def refresh_access_token(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, refresh_token=REFRESH_TOKEN):
    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }
    r = requests.post(TOKEN_URL, data=data, timeout=15)
    r.raise_for_status()
    tokens = r.json()
    print("Token response:", tokens)
    return tokens["access_token"], tokens.get("refresh_token", refresh_token), tokens["expires_in"]



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

# if __name__ == "__main__":
#     print("🔑 Test du refresh access token...")
#     try:
#         access_token, new_refresh, expires_in = refresh_access_token()
#         print("✅ Access token obtenu :", access_token[:40] + "...")
#         print("⌛ Expiration dans :", expires_in, "secondes")
#         print("🔄 Nouveau refresh token :", new_refresh[:40] + "...")
        
#     except Exception as e:
#         print("❌ Erreur lors du refresh :", e)


#===========telecharger les donnees==========#
# start_date = (datetime.utcnow() - timedelta(days=DEFAULT_LOOKBACK_DAYS)).replace(tzinfo=timezone.utc)
# end_date = datetime.utcnow().replace(tzinfo=timezone.utc)
# print(f"Téléchargement des barres pour {DEFAULT_SYMBOL} du {start_date} au {end_date}...")
# df_bars = fetch_bars(
#         symbol=DEFAULT_SYMBOL,
#         access_token=access_token,
#         unit=DEFAULT_UNIT,
#         interval=DEFAULT_INTERVAL,
#         start_utc=start_date,
#         end_utc=end_date,
#         chunk_size=CHUNK_SIZE
#     )




# print("Récupération du token...")
# access_token, _, expires_in = refresh_access_token(
#     CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN
# )
# print(f"Token OK (expires in {expires_in}s)")

# start_date = datetime.now(UTC) - timedelta(days=DEFAULT_LOOKBACK_DAYS)
# end_date   = datetime.now(UTC)
# print(f"Téléchargement des barres pour {SYMBOLTOD} du {start_date} au {end_date}...")
# df_raw = fetch_bars(
#         symbol=SYMBOLTOD,
#         access_token=access_token,
#         unit=DEFAULT_UNIT,
#         interval=DEFAULT_INTERVAL,
#         start_utc=start_date,
#         end_utc=end_date,
#         chunk_size=CHUNK_SIZE
# )

# # Vérifie si le DataFrame contient bien des données
# if not df_raw.empty:
#     # Nom du fichier CSV dynamique selon le symbole et la période
#     filename = f"{SYMBOLTOD}_{DEFAULT_INTERVAL}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv"

#     # Sauvegarde en UTF-8 avec index (datetime)
#     df_raw.to_csv(filename, index=True, encoding='utf-8')

#     print(f"✅ Données sauvegardées dans le fichier : {filename}")
# else:
#     print("⚠️ Aucun résultat à sauvegarder.")
