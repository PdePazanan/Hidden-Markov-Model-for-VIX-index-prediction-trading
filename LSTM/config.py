# config.py

# === TradeStation API Credentials ===
CLIENT_ID = "xxxxxxxxxxxxxxxxx"
CLIENT_SECRET = "xxxxxxxxxxxxxxxxxxxxxxxx"
REFRESH_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
TOKEN_FILE = "tokens.json"
ACCOUNT_ID="xxxxxxxxxxxxxxx"


# === Mode ===
# True = Simulation API (paper trading)
# False = Production API (live trading)
SIM_MODE = True

# === Data Params ===
DEFAULT_SYMBOL = "$VIX.X"
#DEFAULT_SYMBOL ="VXV25"
DEFAULT_INTERVAL =120   # minutes
DEFAULT_UNIT = "Minute"
DEFAULT_LOOKBACK_DAYS =4000
CHUNK_SIZE = 3000        # nb de barres max par requête

# === Trading Params ===
INITIAL_CASH = 100
SLIPPAGE = 0.001          # 0.1%
QTY="1"


FUTURE = "VXX25"
