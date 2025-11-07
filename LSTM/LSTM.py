import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, UTC
from config import CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN, SIM_MODE, DEFAULT_SYMBOL, DEFAULT_UNIT, DEFAULT_INTERVAL, DEFAULT_LOOKBACK_DAYS, CHUNK_SIZE, INITIAL_CASH, SLIPPAGE
from api_ts import refresh_access_token, fetch_bars
from hmm_model import HiddenMarkovModelIndicator
from backtest import *
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration générale
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 30
BATCH_SIZE = 128
LR = 1e-3
N_EPOCHS = 300


print("=== TradeStation HMM Backtest ===")

# 1) Authentification et récupération token
print("Récupération du token...")
access_token, _, expires_in = refresh_access_token(
    CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN
)
print(f"Token OK (expires in {expires_in}s)")


# ----- Charger les données de marché -----
# start_date = (datetime.utcnow() - timedelta(days=DEFAULT_LOOKBACK_DAYS)).replace(tzinfo=timezone.utc)
# end_date = datetime.utcnow().replace(tzinfo=timezone.utc)

start_date = datetime.now(UTC) - timedelta(days=DEFAULT_LOOKBACK_DAYS)
end_date   = datetime.now(UTC)
print(f"Téléchargement des barres pour {DEFAULT_SYMBOL} du {start_date} au {end_date}...")
df_raw = fetch_bars(
        symbol=DEFAULT_SYMBOL,
        access_token=access_token,
        unit=DEFAULT_UNIT,
        interval=DEFAULT_INTERVAL,
        start_utc=start_date,
        end_utc=end_date,
        chunk_size=CHUNK_SIZE
)

# train_end = int(len(df_raw ) * 0.5)
# valid_end = int(len(df_raw ) * 0.7)

# df_train_raw = df_raw .iloc[:train_end]
# df_valid_raw = df_raw .iloc[train_end:valid_end]
# df_test_raw  = df_raw .iloc[valid_end:]


# #======= AJOUT DU VVIX ======================================================
print("Téléchargement du VVIX...")
vvix_df = fetch_bars(
    symbol="$VVIX.X",
    access_token=access_token,
    unit=DEFAULT_UNIT,
    interval=DEFAULT_INTERVAL,
    start_utc=start_date,
    end_utc=end_date,
    chunk_size=CHUNK_SIZE
)

# --- Synchronisation temporelle propre ---

# ==== Pour le VIX ====
df_raw = df_raw.copy()

if isinstance(df_raw.index, pd.DatetimeIndex):
    df_raw = df_raw.reset_index()
    if "index" in df_raw.columns:
        df_raw = df_raw.rename(columns={"index": "datetime"})
    elif "date" in df_raw.columns:
        df_raw = df_raw.rename(columns={"date": "datetime"})

if "datetime" not in df_raw.columns:
    raise ValueError(f"'datetime' column not found in df_raw, colonnes = {df_raw.columns.tolist()}")

df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
df_raw = df_raw.drop_duplicates(subset=["datetime"])
df_raw = df_raw.sort_values("datetime")


# ==== Pour le VVIX ====
vvix_df = vvix_df.copy()

if isinstance(vvix_df.index, pd.DatetimeIndex):
    vvix_df = vvix_df.reset_index()
    if "index" in vvix_df.columns:
        vvix_df = vvix_df.rename(columns={"index": "datetime"})
    elif "date" in vvix_df.columns:
        vvix_df = vvix_df.rename(columns={"date": "datetime"})

if "datetime" not in vvix_df.columns:
    raise ValueError(f"'datetime' column not found in vvix_df, colonnes = {vvix_df.columns.tolist()}")

vvix_df["datetime"] = pd.to_datetime(vvix_df["datetime"])
vvix_df = vvix_df.drop_duplicates(subset=["datetime"])
vvix_df = vvix_df.sort_values("datetime")

# 🔹 Merge propre avec tolérance (VVIX aligné au plus proche)
df_merged = pd.merge_asof(
    df_raw,
    vvix_df[["datetime", "close"]].rename(columns={"close": "vvix_close"}),
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("1h")
)
df_merged.dropna(subset=["vvix_close"], inplace=True)

# === Calcul des features VVIX ===
df_merged["vvix_return"] = df_merged["vvix_close"].pct_change()
df_merged["vvix_volatility"] = df_merged["vvix_return"].rolling(10).std()
df_merged["vvix_trend"] = df_merged["vvix_close"].diff()
df_merged["corr_vix_vvix"] = df_merged["close"].rolling(30).corr(df_merged["vvix_close"])
df_merged.dropna(inplace=True)

# === Split cohérent ===
train_end = int(len(df_merged) * 0.5)
valid_end = int(len(df_merged) * 0.7)

df_train_raw = df_merged.iloc[:train_end]
df_valid_raw = df_merged.iloc[train_end:valid_end]
df_test_raw  = df_merged.iloc[valid_end:]


#====================== FIN DE PREPARATION DU VVIX==================================== 


# ----- Calcul des signaux HMM -----

hmm = HiddenMarkovModelIndicator( 
        lookback_period=111,    
        learning_rate=0.2776267,
        outlier_smoothing=11,  
        outlier_sensitivity=3.1861692,
        sensitivity=0.75,
        min_signal_gap=10
    )

def enrich_with_hmm(df):
    df_out, _ = prepare_hmm_df(df, hmm)
    df_out["future_return"] = df_out["close"].shift(-10)/df_out["close"] - 1    # predire un mouvement a 10 jours futurs pour entrainer
    df_out.dropna(inplace=True)
    df_out["future_return_binary"] = (df_out["future_return"] > 0).astype(float)
    return df_out

def enrich_features_only(df): # on utilise pas le futur car on est en test live
    df_out, _ = prepare_hmm_df(df, hmm)
    return df_out


df_train = enrich_with_hmm(df_train_raw)
df_valid = enrich_with_hmm(df_valid_raw)  
df_test  = enrich_features_only(df_test_raw)

# Features supplémentaires
#extra_feats = ["log_return","return_magnitude","price_velocity","price_acceleration","recent_volatility"]

# === Normalisation des features VVIX ===
vvix_cols = ["vvix_close", "vvix_return", "vvix_volatility", "vvix_trend", "corr_vix_vvix"]

for col in vvix_cols:
    mean = df_train[col].mean()
    std = df_train[col].std()
    for df in [df_train, df_valid, df_test]:
        df[col] = (df[col] - mean) / std


# === Liste complète des features ===
extra_feats = [
    "log_return", "return_magnitude", "price_velocity", "price_acceleration", "recent_volatility",
    "vvix_close", "vvix_return", "vvix_volatility", "vvix_trend", "corr_vix_vvix"
]


class MarketHMMData(Dataset):
    def __init__(self, market_df, target_col=None, seq_len=30, extra_features=None):
        self.seq_len = seq_len
        self.features = market_df[extra_features].values.astype("float32")
        self.hmm_probs = market_df[['prob_top','prob_bottom','prob_normal']].values.astype("float32")
        self.has_target = target_col is not None
        if self.has_target:
            self.targets = market_df[target_col].values.astype("float32")

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        seq = self.features[idx:idx+self.seq_len]
        hmm_last = self.hmm_probs[idx+self.seq_len]
        if self.has_target:
            y = self.targets[idx+self.seq_len]
            return {"seq": torch.tensor(seq), "hmm": torch.tensor(hmm_last), "y": torch.tensor(y)}
        else:
            return {"seq": torch.tensor(seq), "hmm": torch.tensor(hmm_last)}




class HMMFusionNet(nn.Module):
    """
    2 couches LSTM + MLP plus profond 3 couches
    """
    def __init__(self, n_features, lstm_hidden=128, lstm_layers=2,
                 mlp_hidden=(256,128,64), output_size=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout  # dropout inter-couches LSTM
        )
        self.mlp = nn.Sequential(
            
            nn.Linear(lstm_hidden + 3, mlp_hidden[0]),  #layer 1, 256
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(mlp_hidden[0], mlp_hidden[1]),   #layer 2, 128
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(mlp_hidden[1], mlp_hidden[2]),  #layer 3, 64
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(mlp_hidden[2], output_size),
            nn.Sigmoid()   # pour sortie [0,1]
        )

    def forward(self, seq_input, hmm_vec):
        out, _ = self.lstm(seq_input)
        last = out[:, -1, :]
        x = torch.cat([last, hmm_vec], dim=1)
        return self.mlp(x).squeeze(-1)

def train_model(model, train_loader, valid_loader, n_epochs, lr, device):
    model.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCELoss()

    train_losses = []
    valeur_losses   = []

    for epoch in range(1, n_epochs+1):
        model.train()
        running = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}"):
            seq = batch["seq"].to(device)
            hmm = batch["hmm"].to(device)
            y   = batch["y"].to(device)

            opt.zero_grad()
            out = model(seq, hmm)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * seq.size(0)

        train_loss = running/len(train_loader.dataset)
        train_losses.append(train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                seq = batch["seq"].to(device)
                hmm = batch["hmm"].to(device)
                y   = batch["y"].to(device)
                out = model(seq, hmm)
                val_loss += loss_fn(out, y).item() * seq.size(0)
        valeur_loss = val_loss/len(valid_loader.dataset)
        valeur_losses.append(valeur_loss)

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {running/len(train_loader.dataset):.4f} | "
              f"Val Loss: {val_loss/len(valid_loader.dataset):.4f}")
        
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valeur_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy")
    plt.title("Loss evolution")
    plt.legend()
    plt.grid(True)
    plt.close()  # <-- au lieu de plt.show()
    # plt.show()

    return train_losses, valeur_losses


#======================= ENTRAINEMENT ET TEST DU MODELE ===========================

train_dataset = MarketHMMData(df_train, "future_return_binary", SEQ_LEN, extra_feats)
valid_dataset = MarketHMMData(df_valid, "future_return_binary", SEQ_LEN, extra_feats)
test_dataset = MarketHMMData(df_test, target_col=None, seq_len=SEQ_LEN, extra_features=extra_feats)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader  = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)




#========== POUR UN SEUL RUN ===================================================
# model = HMMFusionNet(n_features=len(extra_feats))
# train_model(model, train_loader, valid_loader, N_EPOCHS, LR, DEVICE)



# model.eval()
# probs = []
# with torch.no_grad():
#     for batch in test_loader:
#         seq = batch["seq"].to(DEVICE)
#         hmm = batch["hmm"].to(DEVICE)
#         p   = model(seq, hmm).cpu().numpy()
#         probs.extend(p)

# # Aligner avec le DataFrame test
# df_pred = df_test.iloc[SEQ_LEN:].copy()
# df_pred["pred_prob_up"] = probs
# df_pred.index = pd.to_datetime(df_pred.index)

# # Backtest combiné HMM + NN
# df2 = backtest_hmm_nn_combined(
#     df_pred, initial_cash=10000, slippage=0.001,
#     threshold_buy=0.3, threshold_sell=0.3
# )

# # ---- PREDICTIONS SUR TRAIN / VALID / TEST ----
# def predict_dataset(model, loader, device):
#     model.eval()
#     probs = []
#     with torch.no_grad():
#         for batch in loader:
#             seq = batch["seq"].to(device)
#             hmm = batch["hmm"].to(device)
#             p   = model(seq, hmm).cpu().numpy()
#             probs.extend(p)
#     return np.array(probs)

# # Prédictions
# probs_train = predict_dataset(model, train_loader, DEVICE)
# probs_valid = predict_dataset(model, valid_loader, DEVICE)
# probs_test  = predict_dataset(model, test_loader, DEVICE)

# # Aligner avec les bons DataFrames
# df_train_pred = df_train.iloc[SEQ_LEN:].copy()
# df_valid_pred = df_valid.iloc[SEQ_LEN:].copy()
# df_test_pred  = df_test.iloc[SEQ_LEN:].copy()

# df_train_pred["pred_prob_up"] = probs_train
# df_valid_pred["pred_prob_up"] = probs_valid
# df_test_pred["pred_prob_up"]  = probs_test

# df_all_pred = pd.concat([
#     df_train_pred.assign(dataset="train"),
#     df_valid_pred.assign(dataset="valid"),
#     df_test_pred.assign(dataset="test")
# ])



# df_bt_train = backtest_hmm_nn_combined(df_train_pred, initial_cash=10000, slippage=0.001)
# df_bt_valid = backtest_hmm_nn_combined(df_valid_pred, initial_cash=10000, slippage=0.001)
# df_bt_test  = backtest_hmm_nn_combined(df_test_pred,  initial_cash=10000, slippage=0.001)


# plt.figure(figsize=(16,6))
# plt.plot(df_all_pred.index, df_all_pred["close"], color="black", label="Close Price")
# plt.plot(df_bt_train.index, df_bt_train['equity_norm'], color='green', label='Train Equity')
# plt.plot(df_bt_valid.index, df_bt_valid['equity_norm'], color='gold', label='Validation Equity')
# plt.plot(df_bt_test.index,  df_bt_test['equity_norm'],  color='blue', label='Test Equity')

# # Délimiter visuellement les périodes
# plt.axvspan(df_train_pred.index[0], df_train_pred.index[-1], color="green", alpha=0.1, label="Train")
# plt.axvspan(df_valid_pred.index[0], df_valid_pred.index[-1], color="yellow", alpha=0.1, label="Validation")
# plt.axvspan(df_test_pred.index[0], df_test_pred.index[-1], color="red", alpha=0.1, label="Test")

# plt.title(f"Évolution de l'Equity - Train / Validation / Test ({DEFAULT_SYMBOL})")
# plt.xlabel("Date")
# plt.ylabel("Équity normalisée")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()


# #======================= TRACER======================
# plt.figure(figsize=(16,6))
# plt.plot(df2.index, df2['close'], linewidth=1.0, label='VIX index Close', color='black')
# plt.plot(df2.index, df2['equity_norm'], color='blue', label='HMM+NN Strategy Equity')

# # Signaux HMM
# plt.scatter(df2.index[df2['buy_signal']==1], df2['close'][df2['buy_signal']==1], marker='^', color='green', label='HMM Buy', s=50)
# plt.scatter(df2.index[df2['sell_signal']==1], df2['close'][df2['sell_signal']==1], marker='v', color='red', label='HMM Sell', s=50)

# # Signaux combinés HMM + NN
# plt.scatter(df2.index[df2['combined_buy']==1], df2['close'][df2['combined_buy']==1], marker='^', color='lime', label='NN Confirmed Buy', s=70)
# plt.scatter(df2.index[df2['combined_sell']==1], df2['close'][df2['combined_sell']==1], marker='v', color='orange', label='NN Confirmed Sell', s=70)


# plt.legend()
# plt.grid()
# plt.show()
# drawdown = plot_drawdown(df2, equity_col='equity_norm', title="Drawdowns of HMM strategy")
# metrics = evaluate_strategy(df2, equity_col='equity_norm')

# for k, v in metrics.items():
#     if k == "sharpe" or "buy_signals" or "sell_signals":                     # ← Sharpe en ratio pur
#         print(f"{k}: {v:.2f}")
#     else:                                  # ← autres valeurs en %
#         print(f"{k}: {v*100:.2f}%")

#==================================================================================================================






#======== entrainement pour N runs ========

# sauvegarder des versions "propres" et jamais modifiées
df_train_base = df_train.copy(deep=True)
df_valid_base = df_valid.copy(deep=True)
df_test_base  = df_test.copy(deep=True)


N_RUNS = 20  # Nombre d'entraînements
results = []

# === Boucle d'entraînement ===
for run in range(1, N_RUNS + 1):
    print(f"\n🚀 === Entraînement {run}/{N_RUNS} ===")


    # 1️⃣ Nouveau modèle à chaque itération
    model = HMMFusionNet(n_features=len(extra_feats))

    # 2️⃣ Entraînement
    #train_model(model, train_loader, valid_loader, N_EPOCHS, LR, DEVICE)
    train_losses, val_losses = train_model(model, train_loader, valid_loader, N_EPOCHS, LR, DEVICE)
    

    # 3️⃣ Fonction de prédiction
    def predict_dataset(model, loader, device):
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                seq = batch["seq"].to(device)
                hmm = batch["hmm"].to(device)
                p   = model(seq, hmm).cpu().numpy()
                probs.extend(p)
        return np.array(probs)

    # 4️⃣ Prédictions sur les 3 datasets
    probs_train = predict_dataset(model, train_loader, DEVICE)
    probs_valid = predict_dataset(model, valid_loader, DEVICE)
    probs_test  = predict_dataset(model, test_loader, DEVICE)
#==============modif pour vvix=============================
    df_train_run = df_train_base.copy(deep=True)            #
    df_valid_run = df_valid_base.copy(deep=True)            #
    df_test_run  = df_test_base.copy(deep=True)             #
                                                            #
    df_train_pred = df_train_run.iloc[SEQ_LEN:].copy()      #
    df_valid_pred = df_valid_run.iloc[SEQ_LEN:].copy()      #
    df_test_pred  = df_test_run.iloc[SEQ_LEN:].copy()       #
#===========================================================
    # df_train_pred = df_train.iloc[SEQ_LEN:].copy()
    # df_valid_pred = df_valid.iloc[SEQ_LEN:].copy()
    # df_test_pred  = df_test.iloc[SEQ_LEN:].copy()
 
    df_train_pred["pred_prob_up"] = probs_train
    df_valid_pred["pred_prob_up"] = probs_valid
    df_test_pred["pred_prob_up"]  = probs_test

    # for df in [df_train_pred, df_valid_pred, df_test_pred]:
    #     if "datetime" in df.columns:
    #         df = df.copy()
    #         df["datetime"] = pd.to_datetime(df["datetime"])
    #         df.set_index("datetime", inplace=True)


    # Ensuite concaténer sans réinitialiser l’index
    df_all_pred = pd.concat([
        df_train_pred.assign(dataset="train"),
        df_valid_pred.assign(dataset="valid"),
        df_test_pred.assign(dataset="test")
    ])

    df_all_pred["datetime"] = pd.to_datetime(df_all_pred["datetime"])
    df_all_pred = df_all_pred.set_index("datetime").sort_index()




    # 5️⃣ Backtest sur train / valid / test
    df_bt_train = backtest_hmm_nn_combined(df_train_pred, initial_cash=10000, slippage=0.01)
    df_bt_valid = backtest_hmm_nn_combined(df_valid_pred, initial_cash=10000, slippage=0.01)
    df_bt_test  = backtest_hmm_nn_combined(df_test_pred,  initial_cash=10000, slippage=0.01)

    # df_bt_train = backtest_hmm_nn_leveraged(df_train_pred, initial_cash=10000)
    # df_bt_valid = backtest_hmm_nn_leveraged(df_valid_pred, initial_cash=10000)
    # df_bt_test  = backtest_hmm_nn_leveraged(df_test_pred,  initial_cash=10000)


#=================================================================================================
    # --- Vérification et restauration de l'index datetime ---                                  #
    # -------------------------                                                                 #
    # Restaurer l'index datetime
    # -------------------------
    def restore_bt_index_from_pred(df_bt, df_pred, datetime_col="datetime"):
        # 1) obtenir les timestamps de référence depuis df_pred
        if datetime_col in df_pred.columns:
            times = pd.to_datetime(df_pred[datetime_col].values)
        elif isinstance(df_pred.index, pd.DatetimeIndex):
            times = df_pred.index.values
        else:
            # fallback : utiliser un range date basé sur la longueur de df_pred
            times = pd.date_range(end=pd.Timestamp.now(), periods=len(df_pred), freq="T").values

        # 2) assigner une portion correspondant à la longueur de df_bt
        if len(df_bt) == len(times):
            df_bt.index = pd.to_datetime(times)
        elif len(df_bt) < len(times):
            df_bt.index = pd.to_datetime(times[-len(df_bt):])
        else:
            # df_bt longer -> générer un DatetimeIndex couvrant la même plage
            start = pd.to_datetime(times[0])
            end   = pd.to_datetime(times[-1])
            df_bt.index = pd.date_range(start=start, end=end, periods=len(df_bt))

        return df_bt

    # Appliquer (faites pareil pour train/valid si besoin)
    df_bt_train = restore_bt_index_from_pred(df_bt_train, df_train_pred, datetime_col="datetime")
    df_bt_valid = restore_bt_index_from_pred(df_bt_valid, df_valid_pred, datetime_col="datetime")

  
    df_bt_test = restore_bt_index_from_pred(df_bt_test, df_test_pred, datetime_col="datetime")

    # debug rapide pour vérifier                                                                    #
    print("df_bt_test index type:", type(df_bt_test.index))                                         #
    print("df_bt_test range:", df_bt_test.index[0], "→", df_bt_test.index[-1])                      #
#===================================================================================================
    # 6️⃣ Sauvegarde des résultats
    metrics = evaluate_strategy(df_bt_test, equity_col='equity_norm')
    
    results.append({
        "run": run,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "df_bt_train": df_bt_train,
        "df_bt_valid": df_bt_valid,
        "df_bt_test": df_bt_test,
        "metrics": metrics,
        "model": model.state_dict() 
    })

# ✅ === FIN DE LA BOUCLE ===
print(f"\n✅ {N_RUNS} entraînements terminés !\n")






#================== TRACER DES RESULTATS ==================================


# === Affichage global des courbes de loss après tous les runs ===
plt.figure(figsize=(10,5))
for r in results:
    plt.plot(r["train_losses"], alpha=0.6, label=f"Run {r['run']} (train)")
plt.title("Training Losses over multiple runs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
for r in results:
    plt.plot(r["val_losses"], alpha=0.6, label=f"Run {r['run']} (val)")
plt.title("Validation Losses over multiple runs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()


# === Tracé 1 : 20 equity curves (phase test uniquement) ===
plt.figure(figsize=(16,6))
for r in results:
    plt.plot(df_bt_test.index, df_bt_test['close'], linewidth=1.0, label='VIX index Close', color='black')
    plt.plot(r["df_bt_test"].index, r["df_bt_test"]['equity_norm'], alpha=0.7, label=f'Run {r["run"]}')
plt.title(f"Évolution of the Equity (period of Test) on {N_RUNS} training - {DEFAULT_SYMBOL}")
plt.xlabel("Date")
plt.ylabel("Équity curve normalized")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()


# === Tracé 2 : 20 courbes train / validation / test combinées ===
plt.figure(figsize=(16,6))
for r in results:
    #plt.plot(df_raw.index, df_all_pred["close"], color="black", label="Close Price")
    plt.plot(df_all_pred.index, df_all_pred["close"], color="black", label="Close Price")

    # plt.plot(df_bt_train.index, df_bt_train['close'], linewidth=1.0, label='VIX index Close', color='black')
    # plt.plot(df_bt_valid.index, df_bt_valid['close'], linewidth=1.0, label='VIX index Close', color='black')
    # plt.plot(df_bt_test.index, df_bt_test['close'], linewidth=1.0, label='VIX index Close', color='black')
    
    plt.plot(r["df_bt_train"].index, r["df_bt_train"]['equity_norm'], color='green', alpha=0.8,linewidth=1)
    plt.plot(r["df_bt_valid"].index, r["df_bt_valid"]['equity_norm'], color='gold', alpha=0.8,linewidth=1)
    plt.plot(r["df_bt_test"].index,  r["df_bt_test"]['equity_norm'],  color='blue', alpha=0.8,linewidth=1)

plt.axvspan(df_bt_train.index[SEQ_LEN], df_bt_train.index[-1], color="green", alpha=0.1, label="Train")
plt.axvspan(df_bt_valid.index[SEQ_LEN], df_bt_valid.index[-1], color="yellow", alpha=0.1, label="Validation")
plt.axvspan(df_bt_test.index[SEQ_LEN],  df_bt_test.index[-1],  color="red", alpha=0.1, label="Test")

# plt.axvspan(df_train.index[SEQ_LEN], df_train.index[-1], color="green", alpha=0.1, label="Train")
# plt.axvspan(df_valid.index[SEQ_LEN], df_valid.index[-1], color="yellow", alpha=0.1, label="Validation")
# plt.axvspan(df_test.index[SEQ_LEN],  df_test.index[-1],  color="red", alpha=0.1, label="Test")

plt.title(f"Équity curves on {N_RUNS} runs - Train / Validation / Test ({DEFAULT_SYMBOL})")
plt.xlabel("Date")
plt.ylabel("Équity curve normalized")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# ====================================Model with best profit ===========================
best_run = max(results, key=lambda r: r["metrics"].get("total_return", -999))
print(f"\n🏆 Meilleur modèle, best profit: Run {best_run['run']}")
print("=== Metrics du meilleur modèle (phase test) ===")
for k, v in best_run["metrics"].items():
    print(f"{k}: {v:.4f}")
#======================================================================================


# ========================== Model with less drawdown==================================
best_run_drawdowns = min(results, key=lambda r: r["metrics"].get("max_drawdown", 999))

print(f"\n🏆 Modèle avec le drawdown minimal : Run {best_run_drawdowns['run']}")
print("=== Metrics du modèle avec le moins de drawdowns (phase test) ===")
for k, v in best_run_drawdowns["metrics"].items():
    print(f"{k}: {v:.4f}")
#======================================================================================


# ====== Model with best sharpe ratio, optimal to balance between risk and return======
best_run_sharpe = max(results, key=lambda r: r["metrics"].get("sharpe_ratio", -999))
print(f"\n🏆 Meilleur modèle, best profit: Run {best_run_sharpe['run']}")
print("=== Metrics du modèle avec le meilleur sharpe ratio (phase test) ===")
for k, v in best_run_sharpe["metrics"].items():
    print(f"{k}: {v:.4f}")
#======================================================================================

# === Meilleur modèle par profit ===
best_model_path_profit = f"best_model_profit_run_{best_run['run']}.pth"
torch.save(best_run["model"], best_model_path_profit)

# === Meilleur modèle par drawdown ===
best_model_path_drawdown = f"best_model_drawdown_run_{best_run_drawdowns['run']}.pth"
torch.save(best_run_drawdowns["model"], best_model_path_drawdown)

# === Meilleur modèle par Sharpe ===
best_model_path_sharpe = f"best_model_sharpe_run_{best_run_sharpe['run']}.pth"
torch.save(best_run_sharpe["model"], best_model_path_sharpe)

print("✅ Modèles sauvegardés :")
print(f" - Profit     : {best_model_path_profit}")
print(f" - Drawdown   : {best_model_path_drawdown}")
print(f" - Sharpe     : {best_model_path_sharpe}")



# === Tracé 1 :Model with best profit ===
df_best = best_run["df_bt_test"]

plt.figure(figsize=(16,6))
plt.plot(df_best.index, df_best['close'], linewidth=1.0, label='Close', color='black')
plt.plot(df_best.index, df_best['equity_norm'], color='blue', label='HMM+NN Equity')

plt.scatter(df_best.index[df_best['buy_signal']==1], df_best['close'][df_best['buy_signal']==1],
            marker='^', color='green', s=50, label='HMM Buy')
plt.scatter(df_best.index[df_best['sell_signal']==1], df_best['close'][df_best['sell_signal']==1],
            marker='v', color='red', s=50, label='HMM Sell')

plt.scatter(df_best.index[df_best['combined_buy']==1], df_best['close'][df_best['combined_buy']==1],
            marker='^', color='lime', s=70, label='NN Confirmed Buy')
plt.scatter(df_best.index[df_best['combined_sell']==1], df_best['close'][df_best['combined_sell']==1],
            marker='v', color='orange', s=70, label='NN Confirmed Sell')

plt.title(f"Best Run, with the best profit #{best_run['run']} - Equity and Signals ({DEFAULT_SYMBOL})")
plt.xlabel("Date")
plt.ylabel("Équity / Price")
plt.grid()
plt.legend()
plt.show()



# === Tracé 2 :Model with less drawdown===
df_best_drawdowns = best_run_drawdowns["df_bt_test"]
plt.figure(figsize=(16,6))
plt.plot(df_best_drawdowns.index, df_best_drawdowns['close'], linewidth=1.0, label='Close', color='black')
plt.plot(df_best_drawdowns.index, df_best_drawdowns['equity_norm'], color='blue', label='HMM+NN Equity')

plt.scatter(df_best_drawdowns.index[df_best_drawdowns['buy_signal']==1], df_best_drawdowns['close'][df_best_drawdowns['buy_signal']==1],
            marker='^', color='green', s=50, label='HMM Buy')
plt.scatter(df_best_drawdowns.index[df_best_drawdowns['sell_signal']==1], df_best_drawdowns['close'][df_best_drawdowns['sell_signal']==1],
            marker='v', color='red', s=50, label='HMM Sell')

plt.scatter(df_best_drawdowns.index[df_best_drawdowns['combined_buy']==1], df_best_drawdowns['close'][df_best_drawdowns['combined_buy']==1],
            marker='^', color='lime', s=70, label='NN Confirmed Buy')
plt.scatter(df_best_drawdowns.index[df_best_drawdowns['combined_sell']==1], df_best_drawdowns['close'][df_best_drawdowns['combined_sell']==1],
            marker='v', color='orange', s=70, label='NN Confirmed Sell')

plt.title(f"Best Run with the less drawdowns #{best_run['run']} - Equity and Signals ({DEFAULT_SYMBOL})")
plt.xlabel("Date")
plt.ylabel("Équity / Price")
plt.grid()
plt.legend()
plt.show()



# === Tracé 3 :Model with best sharpe ratio ===
df_best_sharpe = best_run_sharpe["df_bt_test"]
plt.figure(figsize=(16,6))
plt.plot(df_best_sharpe.index, df_best_sharpe['close'], linewidth=1.0, label='Close', color='black')
plt.plot(df_best_sharpe.index, df_best_sharpe['equity_norm'], color='blue', label='HMM+NN Equity')

plt.scatter(df_best_sharpe.index[df_best_sharpe['buy_signal']==1], df_best_sharpe['close'][df_best_sharpe['buy_signal']==1],
            marker='^', color='green', s=50, label='HMM Buy')
plt.scatter(df_best_sharpe.index[df_best_sharpe['sell_signal']==1], df_best_sharpe['close'][df_best_sharpe['sell_signal']==1],
            marker='v', color='red', s=50, label='HMM Sell')

plt.scatter(df_best_sharpe.index[df_best_sharpe['combined_buy']==1], df_best_sharpe['close'][df_best_sharpe['combined_buy']==1],
            marker='^', color='lime', s=70, label='NN Confirmed Buy')
plt.scatter(df_best_sharpe.index[df_best_sharpe['combined_sell']==1], df_best_sharpe['close'][df_best_sharpe['combined_sell']==1],
            marker='v', color='orange', s=70, label='NN Confirmed Sell')

plt.title(f"Best Run with the best sharpe ratio #{best_run['run']} - Equity and Signals ({DEFAULT_SYMBOL})")
plt.xlabel("Date")
plt.ylabel("Équity / Price")
plt.grid()
plt.legend()
plt.show()