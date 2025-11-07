### this file is a live trading algorithm to link with a TradeStation account, you need to fill out your API informations on the Execution_API file ###
### I have made other files for other broker and trading platform, you can contact me if you want ###

import time
import requests
import pandas as pd
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone, UTC
from Execution_API import fetch_bars
from Execution_API import *
from hmm_model import HiddenMarkovModelIndicator

def live_hmm_trader():
    print("======== HMM Live Trader for VX25 ========")

    # 1. Authentification
    access_token = refresh_access_token(CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN)
    
    token_acquired_at = datetime.now(UTC)
    print(f"[{datetime.now()}] Token received ✔️ , expire in 1200s")
    # 2. Télécharger les 100 derniers jours
    start_date = datetime.now(UTC) - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    end_date   = datetime.now(UTC)
   # df = fetch_bars_init(DEFAULT_SYMBOL, access_token, DEFAULT_UNIT, DEFAULT_INTERVAL, start_date, end_date, CHUNK_SIZE)
    df = fetch_bars_init(FUTURE, access_token, DEFAULT_UNIT, DEFAULT_INTERVAL, start_date, end_date, CHUNK_SIZE)
    print(f"{len(df)} bars download for initialization")
    print("  /$$                                 /$$ /$$                     ")
    print(" | $$                                | $$|__/                     ")
    print(" | $$        /$$$$$$   /$$$$$$   /$$$$$$$ /$$ /$$$$$$$   /$$$$$$  ")
    print(" | $$       /$$__  $$ |____  $$ /$$__  $$| $$| $$__  $$ /$$__  $$ ")
    print(" | $$      | $$  \ $$  /$$$$$$$| $$  | $$| $$| $$  \ $$| $$  \ $$ ")
    print(" | $$      | $$  | $$ /$$__  $$| $$  | $$| $$| $$  | $$| $$  | $$ ")
    print(" | $$$$$$$$|  $$$$$$/|  $$$$$$$|  $$$$$$$| $$| $$  | $$|  $$$$$$$ ")
    print(" |________/ \______/  \_______/ \_______/|__/|__/  |__/ \____  $$ ")
    print("                                                        /$$  \ $$ ")
    print("                                                       |  $$$$$$/ ")
    print("                                                        \______/  ")
    print("===================== launching of the HMM =======================")

    # 3. Créer le modèle HMM (5 min optimal)
    # hmm = HiddenMarkovModelIndicator(
    #     lookback_period=70,
    #     learning_rate=0.2776267,
    #     outlier_smoothing=19,
    #     outlier_sensitivity=3.1861692,
    #     sensitivity=0.75,
    #     min_signal_gap=20
    # )
    hmm = HiddenMarkovModelIndicator(  # optimal for 15mn timeframe
        lookback_period=111,
        learning_rate=0.2776267,
        outlier_smoothing=11,  
        outlier_sensitivity=3.1861692,
        sensitivity=0.75,
        min_signal_gap=8
    )

    # 4. Préparer les données et initialiser le graphique
    df_ind, width_days = prepare_hmm_df(df, hmm)
    plt.ion()
    fig, (ax_price, ax_prob) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
    plt.show(block=False)
    

    last_signal = None  # pour éviter doubles envois
    last_buy_bar_index = None
    last_sell_bar_index = None

    def update_plot(df_ind,width_days):
        try:
            ax_price.clear()
            ax_prob.clear()

            # --- Nettoyage / alignement des données ---
            valid_rows = df_ind.dropna(subset=['close', 'green_height', 'red_height', 'outlier_line']).sort_index()
            if valid_rows.empty:
                print(" Aucun point valide à afficher (toutes les valeurs NaN).")
                return

            # Sécurisation de la conversion temps -> float matplotlib
            x = mdates.date2num(valid_rows.index.to_pydatetime())
            green = valid_rows['green_height'].to_numpy()
            red = valid_rows['red_height'].to_numpy()
            outlier = valid_rows['outlier_line'].to_numpy()

            # Alignement strict des tailles
            min_len = min(len(x), len(green), len(red), len(outlier))
            if len(x) != len(green):
                print(f" Taille incohérente détectée : x={len(x)}, green={len(green)} — correction automatique")
            x, green, red, outlier = x[:min_len], green[:min_len], red[:min_len], outlier[:min_len]

            # === PRIX ===
            ax_price.plot(valid_rows.index[:min_len], valid_rows['close'].iloc[:min_len], color='black', label='VX25 Close')
            buy_idx = valid_rows.index[valid_rows['buy_signal']]
            sell_idx = valid_rows.index[valid_rows['sell_signal']]
            ax_price.scatter(buy_idx, valid_rows.loc[buy_idx, 'close'], color='green', marker='^', s=60, label='Buy')
            ax_price.scatter(sell_idx, valid_rows.loc[sell_idx, 'close'], color='red', marker='v', s=60, label='Sell')
            ax_price.legend(loc="upper left")
            ax_price.grid(True, alpha=0.3)
            ax_price.set_title(f"{DEFAULT_SYMBOL} Live — HMM Strategy")

            # === PROBABILITÉS ===
            ax_prob.bar(x, green, width=width_days, alpha=0.3, color='green', label='Bottom prob')
            ax_prob.bar(x, red, width=width_days, alpha=0.3, color='red', label='Top prob')
            ax_prob.plot(valid_rows.index[:min_len], outlier, '--', color='blue', linewidth=1)
            ax_prob.legend(loc="upper left")
            ax_prob.grid(True, alpha=0.3)
            ax_prob.set_ylabel("Probability (%)")

            plt.tight_layout()
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.pause(0.1)

        except Exception as e:
            print(f"Erreur live (update_plot): {e}")

    update_plot(df_ind,width_days)

    print("Launching of the live trading...")
    current_position = 0
    current_position_sell=0


    # 5. Boucle live infinie
    while True:
        try:
            # Vérifie l'heure de la dernière barre
            latest_time = df.index[-2]
            now = datetime.now(UTC) 
            if (now - token_acquired_at).total_seconds() > 1000:  # we refresh the token before it expire
                access_token= refresh_access_token(CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN)
                token_acquired_at = datetime.now(UTC) 
                print(" New token received ✔️ , expire in 1200s") 

           
            print(f"\nLast update time : {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            bid, ask, last = get_quote(access_token, DEFAULT_SYMBOL)
            print(f"last price VIX index : ${last}")
            bid1, ask1, last1 = get_quote(access_token, FUTURE)
            print(f"last prices VXV25 : \n bid : ${bid1} , ask : ${ask1}")

            print("=============================================")

            new_start = latest_time + timedelta(seconds=1)
            new_end = now

          #  df_new = fetch_bars_init(DEFAULT_SYMBOL, access_token, DEFAULT_UNIT, DEFAULT_INTERVAL, new_start, new_end, CHUNK_SIZE)
            df_new = fetch_bars_init(FUTURE, access_token, DEFAULT_UNIT, DEFAULT_INTERVAL, new_start, new_end, CHUNK_SIZE)
            
            if not df_new.empty:
                # Fusionne sans doublons
                df_new = df_new[~df_new.index.isin(df.index)]
                df = pd.concat([df, df_new]).sort_index()

                #df = pd.concat([df, df_new]).drop_duplicates().sort_index()
                #df = df.iloc[-4000:]  # garder taille max
            else:
                print("⚠️ Aucun nouveau tick reçu.")


            # if df_new.empty:
            #     print("⚠️ Aucune nouvelle barre disponible, on attend 60s...")
            #     time.sleep(60)
            #     continue
            # # on garde une fenetre de 4000 barres, a chaque nouvelle on supprime la plus ancienne
            # df_new = df_new[~df_new.index.isin(df.index)]
            # df = pd.concat([df, df_new]).sort_index()


            if len(df) > 4000:  # environ 100 jours de barres 5m
                df = df.iloc[-4000:]

            df_ind, width_days = prepare_hmm_df(df, hmm)
            last_bar_time = df.index[-1]
            last_row = df_ind.iloc[-1]  # avant la dernière barre complète
            close_px = last_row['close']
         

# Logic of the strategy :
#           buy --> buy : We open a long position
#           buy --> sell: We close the previous long positions
#           sell -->sell: We open a short position
#           sell --> buy: We close the previous short position and we open a long position

            #  BUY SIGNAL
            if last_row['buy_signal']:
                if last_buy_bar_index != last_bar_time:
                    print(f" BUY signal detected at {close_px:.2f} ({last_bar_time})")
                    buy_position(
                        access_token=access_token,
                        account_id=ACCOUNT_ID,
                        symbol=FUTURE,
                        side="Buy",
                        qty=TRADE_QTY+current_position_sell
                    )
                    print(f" Buy order sent at {close_px:.2f}.")
                    current_position_sell=0
                    current_position += 1
                    last_signal = 'buy'
                    last_buy_bar_index = last_bar_time  # mémorise la barre du signal
                else:
                    print("🟢 Buy signal déjà exécuté sur cette barre.")

            #  SELL SIGNAL TO CLOSE POSITION
            if last_row['sell_signal'] and last_signal == 'buy':
                if last_sell_bar_index != last_bar_time:
                    print(f" SELL signal to close detected at {close_px:.2f} ({last_bar_time})")
                    position=get_positions(access_token, ACCOUNT_ID)
                    if not position.get("Positions"):
                        print("no open positions")
                    else :
                        print("Open positions :")
                        for pos in position["Positions"]:
                            quantity=pos.get("Quantity","N/A")
                            longshort = pos.get("LongShort", "N/A")
                            print(f" {quantity} {longshort} will be closed")
                        if longshort=="Long" :
                            sell_position(
                                access_token=access_token,
                                account_id=ACCOUNT_ID,
                                symbol=FUTURE,
                                side="Sell",
                                qty=quantity
                            )
                            print(f" Sell order sent at {close_px:.2f}.")
                            last_signal = 'sell'
                            current_position_sell = 0
                            last_sell_bar_index = last_bar_time  # mémorise la barre du signal
            
             # SELL SIGNAL 
            if last_row['sell_signal']  and last_signal != 'buy':
                if last_sell_bar_index != last_bar_time:
                    print(f" SELL signal detected at {close_px:.2f} ({last_bar_time})")
                    sell_position(
                        access_token=access_token,
                        account_id=ACCOUNT_ID,
                        symbol=FUTURE,
                        side="Sell",
                        qty=TRADE_QTY
                    )
                    print(f" Sell order sent at {close_px:.2f}.")
                    last_signal = 'sell'
                    current_position_sell += 1
                    last_sell_bar_index = last_bar_time  # mémorise la barre du signal



                else:
                    print("🔴 Sell signal déjà exécuté sur cette barre.")
            else:
                print("No new signals detected.")

            print("last signal : ", last_signal)
            print(f"len(index)={len(df_ind.index)}, len(green)={len(df_ind['green_height'])}, len(outlier)={len(df_ind['outlier_line'])}")
            update_plot(df_ind,width_days)

            # recupere les positions ouvertes
            position=get_positions(access_token, ACCOUNT_ID)
            if not position.get("Positions"):
                print("no open positions")
            else :
                print("Open positions :")
                for pos in position["Positions"]:
                    symbol = pos.get("Symbol", "N/A")
                    quantity=pos.get("Quantity","N/A")
                    time_str=pos.get('Timestamp', None)
                    open_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) if time_str else None
                    price=pos.get('AveragePrice', 'N/A')
                    longshort = pos.get("LongShort", "N/A")
                    unrealized_pl = pos.get("UnrealizedProfitLoss", 0)
                    print(f"  {symbol} | {longshort} opened the {open_time} at {price} | \n  Quantity :{quantity} |Unrealized P/L : $ {unrealized_pl}")
                    print("current position: ",quantity)
                
            time.sleep(20)  # check all the 20 secondes

        except Exception as e:
            print("Erreur live:", e)
            time.sleep(60)


if __name__ == "__main__":
    live_hmm_trader()




