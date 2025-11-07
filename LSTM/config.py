# config.py

# === TradeStation API Credentials ===
CLIENT_ID = "3bE5Y8Qua6OHghAGOHmK0JQ0aMYz8QgC"
CLIENT_SECRET = "SviP-D-LEapA9Ou1fjlQTK8gZZ_t998C5jMlzWfVJ_A7iUNLIEh8qyr7QyjUfCp3"
REFRESH_TOKEN = "L1EZpMoAXU4pe0XOD-iCKFj9IBhokalIFCWztWQACraKv"
TOKEN_FILE = "tokens.json"
ACCOUNT_ID="SIM903218F"


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