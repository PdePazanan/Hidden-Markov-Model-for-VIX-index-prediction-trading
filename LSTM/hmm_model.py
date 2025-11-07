
import numpy as np
import pandas as pd
from collections import deque

class HiddenMarkovModelIndicator:
    def __init__(self, lookback_period=50, learning_rate=0.2,
                 outlier_smoothing=20, outlier_sensitivity=3.0,
                 sensitivity=0.75, min_signal_gap=10, show_probabilities=True):
        self.lookback_period = lookback_period
        self.learning_rate = learning_rate
        self.outlier_smoothing = outlier_smoothing
        self.outlier_sensitivity = outlier_sensitivity
        self.sensitivity = sensitivity
        self.min_signal_gap = min_signal_gap
        self.show_probabilities = show_probabilities

        self.normal_to_normal = 0.385              #proba de passer de l'etat normal a normal    # simu optuna 2007-2025 pour VIX en daily 
        self.normal_to_top = 0.1045      #0.075     # etat normal a etat haut
        self.normal_to_bottom = 0.51   #0.075
        self.top_to_normal = 0.5575       #0.6
        self.top_to_top = 0.38495           #0.35
        self.top_to_bottom = 0.06       #0.05
        self.bottom_to_normal = 0.79417   #0.6
        self.bottom_to_top = 0.05375     #0.05
        self.bottom_to_bottom = 0.16   #0.35

        # self.normal_to_normal = 0.56415              #proba de passer de l'etat normal a normal    # simu optuna pour 15min
        # self.normal_to_top = 0.327536      #0.075     # etat normal a etat haut
        # self.normal_to_bottom = 0.10835   #0.075
        # self.top_to_normal = 0.238185       #0.6
        # self.top_to_top = 0.38459           #0.35
        # self.top_to_bottom = 0.37723       #0.05   combinaison heuristique
        # self.bottom_to_normal = 0.232936   #0.6
        # self.bottom_to_top = 0.31651     #0.05
        # self.bottom_to_bottom = 0.4564  #0.35

        self.sensitivity = 0.7
        self.min_signal_gap = 10

        # Probas initials
        self.prob_normal = 0.85
        self.prob_top = 0.075
        self.prob_bottom = 0.075

        # === Ajout dans __init__ ===
        self.prev_probs_normal = deque([self.prob_normal]*self.lookback_period, maxlen=self.lookback_period)
        self.prev_probs_top = deque([self.prob_top]*self.lookback_period, maxlen=self.lookback_period)
        self.prev_probs_bottom = deque([self.prob_bottom]*self.lookback_period, maxlen=self.lookback_period)

    def _normalize_price(self, series, length):
        highest = series.rolling(length).max()
        lowest = series.rolling(length).min()
        rng = highest - lowest
        return np.where(rng > 0, (series - lowest) / rng, 0.5)

    def run(self, df: pd.DataFrame):
        df = df.copy()
        # Assure colonnes attendues
        for c in ['open','high','low','close']:
            if c not in df.columns:
                raise ValueError(f"Colonne manquante: {c}")

        # === Features du marche pour observations vectorisées ===
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['return_magnitude'] = np.abs(df['log_return'])

        df['price_velocity'] = df['close'].diff()
        df['price_acceleration'] = df['price_velocity'].diff()
        df['price_jerk'] = df['price_acceleration'].diff()

        df['local_price_position'] = self._normalize_price(df['close'], self.lookback_period)
        df['local_high_position'] = self._normalize_price(df['high'], self.lookback_period)
        df['local_low_position'] = self._normalize_price(df['low'], self.lookback_period)

        df['price_gap'] = (df['open'] - df['close'].shift(1)).abs() / df['close'].shift(1)
        df['intraday_range'] = (df['high'] - df['low']) / df['close']
        df['body_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-6)

        # Volatilité récente: moyenne glissante des variations absolues en % sur 10 pas
        df['recent_volatility'] = df['close'].pct_change().abs().rolling(10).mean()
        df['rv_sma10'] = df['recent_volatility'].rolling(10).mean()
        df['rv_sma15'] = df['recent_volatility'].rolling(15).mean()
        df['rv_sma20'] = df['recent_volatility'].rolling(20).mean()

        # Moyenne prix
        df['mean_price'] = df['close'].rolling(self.lookback_period).mean()
        df['price_deviation'] = (df['close'] - df['mean_price']) / (df['mean_price'] + 1e-9)
        df['abs_deviation'] = df['price_deviation'].abs()

        # Placeholders résultats
        df['prob_normal'] = np.nan
        df['prob_top'] = np.nan
        df['prob_bottom'] = np.nan
        df['confirmed_top'] = False
        df['confirmed_bottom'] = False
        df['cumulative_pressure'] = np.nan

        # Boucles dépendantes du temps (
        up_streak = 0
        down_streak = 0
        cumulative_pressure = 0.0
        bars_since_top, bars_since_bottom = 999, 999
        market_states = [0] * len(df)  # 0=normal, 1=top, 2=bottom    #####

        # #=======Definition de cumulative pressure glissante sur lookback period==============#
        for i in range(len(df)):
            if i == 0:
                # init
                df.iat[i, df.columns.get_loc('cumulative_pressure')] = cumulative_pressure
                continue

            # Streaks
            if df['close'].iat[i] > df['close'].iat[i-1]:
                up_streak += 1
                down_streak = 0
            elif df['close'].iat[i] < df['close'].iat[i-1]:
                down_streak += 1
                up_streak = 0
            
            # === cumulative pressure glissante ===
            lookback_period=100
            daily_pressure = (df['close'].iat[i] - df['open'].iat[i]) / (df['high'].iat[i] - df['low'].iat[i] + 1e-6)
            cumulative_pressure_window = deque(maxlen=lookback_period)
            cumulative_pressure_window.append(daily_pressure)
            cumulative_pressure = np.mean(cumulative_pressure_window)
            df.iat[i, df.columns.get_loc('cumulative_pressure')] = cumulative_pressure


            if i < self.lookback_period:
                # Probabilités initiales avant fenêtrage complet
                df.iat[i, df.columns.get_loc('prob_normal')] = self.prev_probs_normal[-1]
                df.iat[i, df.columns.get_loc('prob_top')] = self.prev_probs_top[-1]
                df.iat[i, df.columns.get_loc('prob_bottom')] = self.prev_probs_bottom[-1]
                continue
           

            # Raccourcis row-like
            lp = float(df['local_price_position'].iat[i])
            pv = float(df['price_velocity'].iat[i])
            pa = float(df['price_acceleration'].iat[i])
            gap = float(df['price_gap'].iat[i])
            rv = float(df['recent_volatility'].iat[i])
            rv_sma10 = float(df['rv_sma10'].iat[i]) if not np.isnan(df['rv_sma10'].iat[i]) else rv
            rv_sma15 = float(df['rv_sma15'].iat[i]) if not np.isnan(df['rv_sma15'].iat[i]) else rv
            rv_sma20 = float(df['rv_sma20'].iat[i]) if not np.isnan(df['rv_sma20'].iat[i]) else rv

            # === Emissions === we calculate the probability of observing the current features for the 3 states  heuristic combinaison
            normal_velocity_prob = 0.8 if abs(pv) < rv * df['close'].iat[i] else 0.3
            normal_position_prob = 0.8 if (lp > 0.2 and lp < 0.8) else 0.4
            normal_pressure_prob = 0.7 if abs(cumulative_pressure) < 0.3 else 0.3
            normal_streak_prob = 0.8 if (up_streak < 5 and down_streak < 5) else 0.2
            normal_emission = (normal_velocity_prob + normal_position_prob + normal_pressure_prob + normal_streak_prob) / 4.0

            top_position_prob = 0.9 if lp > 0.8 else (0.5 if lp > 0.6 else 0.1)
            top_acceleration_prob = 0.8 if (pa < 0 and pv > 0) else 0.3
            top_pressure_prob = 0.8 if (cumulative_pressure > 0.4 and daily_pressure < cumulative_pressure) else 0.3
            top_volatility_prob = 0.7 if rv > rv_sma10 else 0.4
            top_gap_prob = 0.6 if gap > rv else 0.4
            top_streak_prob = 0.8 if (up_streak > 3 and pa < 0) else 0.4
            top_emission = (top_position_prob*1.5 + top_acceleration_prob + top_pressure_prob + top_volatility_prob + top_gap_prob + top_streak_prob) / 6.0

            bottom_position_prob = 0.9 if lp < 0.2 else (0.6 if lp < 0.4 else 0.1)
            bottom_acceleration_prob = 0.8 if (pa > 0 and pv < 0) else 0.3
            bottom_pressure_prob = 0.8 if (cumulative_pressure < -0.4 and daily_pressure > cumulative_pressure) else 0.3
            bottom_volatility_prob = 0.7 if rv > rv_sma10 else 0.4
            bottom_gap_prob = 0.6 if gap > rv else 0.4
            bottom_streak_prob = 0.8 if (down_streak > 3 and pa > 0) else 0.4
            bottom_emission = (bottom_position_prob + bottom_acceleration_prob + bottom_pressure_prob + bottom_volatility_prob + bottom_gap_prob + bottom_streak_prob) / 6.0

            # === Normalisation pour les transitions (chaque barre) ===
            n_tot = self.normal_to_normal + self.normal_to_top + self.normal_to_bottom
            if n_tot != 1.0:
                self.normal_to_normal /= n_tot
                self.normal_to_top /= n_tot
                self.normal_to_bottom /= n_tot

            t_tot = self.top_to_normal + self.top_to_top + self.top_to_bottom
            if t_tot != 1.0:
                self.top_to_normal /= t_tot
                self.top_to_top /= t_tot
                self.top_to_bottom /= t_tot

            b_tot = self.bottom_to_normal + self.bottom_to_top + self.bottom_to_bottom
            if b_tot != 1.0:
                self.bottom_to_normal /= b_tot
                self.bottom_to_top /= b_tot
                self.bottom_to_bottom /= b_tot

            prior_normal = self.prev_probs_normal[-1] * self.normal_to_normal + \
                    self.prev_probs_top[-1] * self.top_to_normal + \
                    self.prev_probs_bottom[-1] * self.bottom_to_normal    # proba a priori d'etre dans un etat normal avant de voir la nouvelle observation

            prior_top = self.prev_probs_normal[-1] * self.normal_to_top + \
                    self.prev_probs_top[-1] * self.top_to_top + \
                    self.prev_probs_bottom[-1] * self.bottom_to_top

            prior_bottom = self.prev_probs_normal[-1] * self.normal_to_bottom + \
                    self.prev_probs_top[-1] * self.top_to_bottom + \
                    self.prev_probs_bottom[-1] * self.bottom_to_bottom


            # 2 Bayes update
            unnorm_normal = prior_normal * normal_emission
            unnorm_top = prior_top * top_emission
            unnorm_bottom = prior_bottom * bottom_emission
            total_prob = unnorm_normal + unnorm_top + unnorm_bottom
            if total_prob > 0:
                self.prob_normal = unnorm_normal / total_prob
                self.prob_top = unnorm_top / total_prob
                self.prob_bottom = unnorm_bottom / total_prob

            # 3 Mise à jour des mémoires glissantes
          
            self.prev_probs_normal.append(self.prob_normal)
            self.prev_probs_top.append(self.prob_top)
            self.prev_probs_bottom.append(self.prob_bottom)

            
            # === Détermination état dominant ===
          #  new_state = 0
          ## State 1 : etats top --> vendre
          ##state 2 : etat bottom --> acheter
          ## state 3 : etat normal --> hold
            if (self.prob_top > self.prob_normal and self.prob_top > self.prob_bottom):# and self.prob_top > self.sensitivity):
                new_state = 1
            elif (self.prob_bottom > self.prob_normal and self.prob_bottom > self.prob_top): #and self.prob_bottom > self.sensitivity):
                new_state = 2
            elif (self.prob_normal > self.prob_bottom and self.prob_normal > self.prob_top):
                new_state=0
            
           # === Signaux et espaces min ===
            
            top_signal = bottom_signal = False
            if new_state == 1 and self.prob_top > self.sensitivity and bars_since_top >= self.min_signal_gap:
                    top_signal = True
                    bars_since_top = 0
                    bars_since_bottom += 1
            elif new_state == 2 and self.prob_bottom > self.sensitivity and bars_since_bottom >= self.min_signal_gap:
                    bottom_signal = True
                    bars_since_bottom = 0
                    bars_since_top += 1
            else:
                    bars_since_top += 1
                    bars_since_bottom += 1
           
            # === Confirmations multi-facteurs ===
            confirmation_factors_top = 0
            if lp > 0.8: confirmation_factors_top += 1
            if (pa < 0 and pv > 0): confirmation_factors_top += 1
            if (rv > rv_sma15): confirmation_factors_top += 1
            if (cumulative_pressure > 0.3): confirmation_factors_top += 1

            confirmation_factors_bottom = 0
            if lp < 0.2: confirmation_factors_bottom += 1
            if (pa > 0 and pv < 0): confirmation_factors_bottom += 1
            if (rv > rv_sma15): confirmation_factors_bottom += 1
            if (cumulative_pressure < -0.3): confirmation_factors_bottom += 1

            confirmed_top = bool(top_signal and (confirmation_factors_top >= 2) and (self.prob_top > self.sensitivity))
            confirmed_bottom = bool(bottom_signal and (confirmation_factors_bottom >= 2) and (self.prob_bottom > self.sensitivity))

            market_states[i] = new_state

            # === Apprentissage adaptatif ===
            la = self.learning_rate
            if i >= 5:
                price_change_5 = (df['close'].iat[i] - df['close'].iat[i-5]) / (df['close'].iat[i-5] + 1e-12)
            else:
                price_change_5 = 0.0

            # Zones hautes/baisses
            if (lp > 0.7 and price_change_5 < -0.02):
                self.normal_to_top = min(self.normal_to_top + la * 0.001, 0.2)
                self.normal_to_normal = max(self.normal_to_normal - la * 0.0005, 0.6)
            if (lp < 0.3 and price_change_5 > 0.02):
                self.normal_to_bottom = min(self.normal_to_bottom + la * 0.001, 0.2)
                self.normal_to_normal = max(self.normal_to_normal - la * 0.0005, 0.6)

            # Volatilité élevée
            if rv > (rv_sma20 * 1.2):
                self.normal_to_top = min(self.normal_to_top + la * 0.0005, 0.2)
                self.normal_to_bottom = min(self.normal_to_bottom + la * 0.0005, 0.2)
                self.normal_to_normal = max(self.normal_to_normal - la * 0.001, 0.6)

            # Apprentissage basé sur les signaux 
            if confirmed_top:
                if i >= 5 and df['close'].iat[i-5] < df['close'].iat[i]: 
                    self.normal_to_top = min(self.normal_to_top + la * 0.02, 0.25)
                    self.top_to_normal = min(self.top_to_normal + la * 0.1, 0.9)
                    self.normal_to_normal = max(self.normal_to_normal - la * 0.01, 0.6)
                elif i >= 5 and df['close'].iat[i-5] >= df['close'].iat[i]:
                    self.normal_to_top = max(self.normal_to_top - la * 0.01, 0.05)
                    self.top_to_normal = max(self.top_to_normal - la * 0.05, 0.3)

            if confirmed_bottom:
                if i >= 5 and df['close'].iat[i-5] > df['close'].iat[i]:
                    self.normal_to_bottom = min(self.normal_to_bottom + la * 0.02, 0.25)
                    self.bottom_to_normal = min(self.bottom_to_normal + la * 0.1, 0.9)
                    self.normal_to_normal = max(self.normal_to_normal - la * 0.01, 0.6)
                elif i >= 5 and df['close'].iat[i-5] <= df['close'].iat[i]:
                    self.normal_to_bottom = max(self.normal_to_bottom - la * 0.01, 0.05)
                    self.bottom_to_normal = max(self.bottom_to_normal - la * 0.05, 0.3)

            # Écriture des sorties
            df.iat[i, df.columns.get_loc('prob_normal')] = self.prob_normal
            df.iat[i, df.columns.get_loc('prob_top')] = self.prob_top
            df.iat[i, df.columns.get_loc('prob_bottom')] = self.prob_bottom
            df.iat[i, df.columns.get_loc('confirmed_top')] = confirmed_top
            df.iat[i, df.columns.get_loc('confirmed_bottom')] = confirmed_bottom


        # écrire les états dans le dataframe et labels utiles
        df['market_state'] = pd.Series(market_states, index=df.index).astype(int)
        df['market_state_label'] = df['market_state'].map({0: 'normal', 1: 'top', 2: 'bottom'})
        # couleur utile si besoin
        df['market_state_color'] = df['market_state'].map({0: 'gray', 1: 'red', 2: 'green'})


        # === Lignes d'outlier & crossunders ===
        df['top_prob_value'] = df['prob_top'] * 100
        df['bottom_prob_value'] = df['prob_bottom'] * 100
        df['signal_strength'] = df[['top_prob_value','bottom_prob_value']].max(axis=1)
        baseline = df['signal_strength'].rolling(self.outlier_smoothing).mean()
        signal_std = df['signal_strength'].rolling(self.outlier_smoothing).std()
        baseline = df['signal_strength'].rolling(self.outlier_smoothing).mean()
        signal_std = df['signal_strength'].rolling(self.outlier_smoothing).std()
        df['outlier_line'] = (baseline + signal_std * self.outlier_sensitivity).rolling(10).mean()


        # Crossunder (comme ta.crossunder)
        df['top_cross_down'] = (df['top_prob_value'] < df['outlier_line']) & (df['top_prob_value'].shift(1) >= df['outlier_line'].shift(1))
        df['bottom_cross_down'] = (df['bottom_prob_value'] < df['outlier_line']) & (df['bottom_prob_value'].shift(1) >= df['outlier_line'].shift(1))

        return df
    
    def generate_signals(self, df: pd.DataFrame):
        """
        Génère des signaux exploitables pour backtest ou live.
        Signal = +1 (achat), -1 (vente), 0 (rien).
        On se base sur la logique buy/sell de ton ancien backtest.
        """
        df = self.run(df)  # ton HMM original calcule toutes les colonnes nécessaires

        # créer top_prob_value / bottom_prob_value si ce n'est pas fait
        if 'top_prob_value' not in df.columns:
            df['top_prob_value'] = df['prob_top'] * 100
        if 'bottom_prob_value' not in df.columns:
            df['bottom_prob_value'] = df['prob_bottom'] * 100

        # outlier_line si pas déjà calculé
        if 'outlier_line' not in df.columns:
            baseline = df[['top_prob_value','bottom_prob_value']].max(axis=1).rolling(self.outlier_smoothing).mean()
            signal_std = df[['top_prob_value','bottom_prob_value']].max(axis=1).rolling(self.outlier_smoothing).std()
            df['outlier_line'] = (baseline + signal_std * self.outlier_sensitivity).rolling(10).mean()

        # signaux buy/sell
        df['signal'] = 0
        df.loc[df['bottom_prob_value'] > df['outlier_line'], 'signal'] = 1
        df.loc[df['top_prob_value'] > df['outlier_line'], 'signal'] = -1

        return df[["open","high","low","close","signal",
                "prob_normal","prob_top","prob_bottom",
                "top_prob_value","bottom_prob_value","outlier_line"]]
