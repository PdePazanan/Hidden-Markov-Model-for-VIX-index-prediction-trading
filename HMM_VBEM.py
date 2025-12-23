# fichier: vix_bayesian_hmm.py
# VB-HMM with independent Gaussian emissions per dimension (Normal-Gamma conjugacy)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln
from tqdm import trange
import joblib

# ---------------------------
# Feature preparation
# ---------------------------
def prepare_data_multifeat(df, feature_col="close"):
    df = df.copy().sort_index()
    df = df.dropna(subset=[feature_col])
    # core features
    df['logret'] = np.log(df[feature_col] / df[feature_col].shift(1))

    roll_mean = df[feature_col].rolling(20).mean()
    roll_std  = df[feature_col].rolling(20).std()
    df['zscore'] = (df[feature_col] - roll_mean) / roll_std


#    df['zscore'] = (df[feature_col] - df[feature_col].rolling(20).mean()) / df[feature_col].rolling(20).std()
    if "high" in df.columns and "low" in df.columns:
        df["range"] = (df["high"] - df["low"]) / df[feature_col]
    else:
        df["range"] = df["logret"].abs()
    df["slope10"] = df[feature_col].diff(10) / 10.0

    df = df.dropna()
    X = df[["logret", "zscore", "range", "slope10"]].values.astype(float)
    return X, df.index, df

# ---------------------------
# VB-HMM Implementation
# ---------------------------

class VBHMM:
    """
    Variational Bayesian HMM with:
    - K hidden states
    - D-dimensional observations with independent Gaussian emissions per dimension.
    Priors:
      - pi ~ Dir(alpha0)
      - A[row] ~ Dir(alphaA0)
      - For each state k and dimension d:
            tau_kd ~ Gamma(a0, b0)  (precision)
            mu_kd | tau_kd ~ Normal(mu0, (lambda0 * tau_kd)^-1)
    VB approximates posterior q(pi), q(A), q(mu,tau) with same factorized forms.
    """
    def __init__(self, n_states=3, n_iter=100, tol=1e-4, verbose=True):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

        # priors (weakly informative defaults)
        self.alpha0 = np.ones(self.K) * 1.0         # prior for initial state (Dirichlet)
        self.alphaA0 = np.ones(self.K) * 1.0        # each row Dir prior concentration (symmetric)
        # Normal-Gamma hyperparams
        self.mu0 = 0.0
        self.lambda0 = 1e-2
        self.a0 = 1.0
        self.b0 = 1.0

        # placeholders for variational params
        self.alpha_pi = None       # Dirichlet posterior for pi
        self.alpha_A = None        # Dirichlet posterior rows for A (K x K)
        # for mu/tau: arrays (K, D)
        self.lambda_kd = None
        self.mu_kd = None
        self.a_kd = None
        self.b_kd = None

    def _init_from_data(self, X):
        N, D = X.shape
        self.N = N
        self.D = D
        K = self.K

        # initialize q(pi) and q(A)
        self.alpha_pi = self.alpha0 + np.ones(K) * (N / (K * 10.0))  # mild
        self.alpha_A = np.tile(self.alphaA0, (K, 1)) + np.random.rand(K, K)

        # initialize Normal-Gamma variational params
        self.lambda_kd = np.ones((K, D)) * (self.lambda0 + 1.0)
        self.mu_kd = np.zeros((K, D)) + np.mean(X, axis=0) + 0.01 * np.random.randn(K, D)
        self.a_kd = np.ones((K, D)) * (self.a0 + 0.5)
        self.b_kd = np.ones((K, D)) * (self.b0 + 0.5)

        # responsibilities initialization via KMeans-like soft assignment (simple)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=5, random_state=42).fit(X)
        labels = km.labels_
        gamma = np.zeros((N, K))
        for k in range(K):
            gamma[:, k] = (labels == k).astype(float)
        # smooth
        gamma = gamma + 1e-2
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        return gamma

    def _expectations(self):
        """
        Compute expected logs used in forward-backward:
        E[log pi], E[log A], E[ log N(x_t | mu_k, tau_k) ] (per time k)
        """
        K, D = self.K, self.D

        # E log pi
        E_log_pi = digamma(self.alpha_pi) - digamma(self.alpha_pi.sum())

        # E log A: each row i
        E_log_A = digamma(self.alpha_A) - digamma(self.alpha_A.sum(axis=1, keepdims=True))

        # For emissions: E[log tau] and E[tau] and E[mu^2 * tau] etc per k,d
        E_log_tau = digamma(self.a_kd) - np.log(self.b_kd)
        E_tau = self.a_kd / self.b_kd

        # compute expected log likelihood for each time n and state k:
        # E_log_N = 0.5 * sum_d [ E_log_tau_d - log(2pi) - E_tau_d * ( (x_nd - mu_kd)^2 + 1/lambda_kd ) ]
        # returns array shape (N, K)
        X = self.X  # (N, D)
        N = self.N
        E_log_N = np.zeros((N, K))
        const = 0.5 * (-np.log(2 * np.pi))
        for k in range(K):
            tmp = 0.0
            for d in range(D):
                mu_kd = self.mu_kd[k, d]
                lam_kd = self.lambda_kd[k, d]
                Elogtau = E_log_tau[k, d]
                Etau = E_tau[k, d]
                # squared error term:
                sq = (X[:, d] - mu_kd) ** 2
                # expectation includes variance term due to posterior on mu: + 1/lam_kd
                term = 0.5 * (Elogtau - np.log(2 * np.pi) - Etau * (sq + 1.0 / lam_kd))
                tmp += term
            E_log_N[:, k] = tmp
        return E_log_pi, E_log_A, E_log_N

    def _forward_backward(self, E_log_pi, E_log_A, E_log_N):
        """
        Standard forward-backward in log-domain.
        E_log_pi: (K,), E_log_A: (K,K), E_log_N: (N,K)
        Returns gamma (N,K) and xi_sum (K,K) summed over time
        """
        N, K = self.N, self.K

        # forward pass in log-space
        log_alpha = np.zeros((N, K))
        log_alpha[0] = E_log_pi + E_log_N[0]
        log_alpha[0] = log_alpha[0] - logsumexp(log_alpha[0])

        for t in range(1, N):
            # logsum over previous states i: logsum_i (log_alpha[t-1,i] + E_log_A[i,j])
            prev = log_alpha[t - 1][:, None] + E_log_A  # (K,K)
            log_alpha[t] = E_log_N[t] + logsumexp(prev, axis=0)
            log_alpha[t] = log_alpha[t] - logsumexp(log_alpha[t])

        # backward pass for gamma
        log_beta = np.zeros((N, K))
        # initialization log_beta[N-1] = 0
        for t in range(N - 2, -1, -1):
            # log_beta[t] = logsum_j ( E_log_A[i,j] + E_log_N[t+1,j] + log_beta[t+1, j] )
            temp = E_log_A + (E_log_N[t + 1] + log_beta[t + 1])[None, :]
            log_beta[t] = logsumexp(temp, axis=1)
            log_beta[t] = log_beta[t] - logsumexp(log_beta[t])

        log_gamma = log_alpha + log_beta
        # normalize
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        # xi sums
        xi_sum = np.zeros((K, K))
        for t in range(N - 1):
            # log_xi_ij ∝ log_alpha[t,i] + E_log_A[i,j] + E_log_N[t+1,j] + log_beta[t+1,j]
            log_xi = (log_alpha[t][:, None] +
                      E_log_A +
                      (E_log_N[t + 1] + log_beta[t + 1])[None, :])
            log_xi = log_xi - logsumexp(log_xi)
            xi_sum += np.exp(log_xi)

        return gamma, xi_sum

    def fit(self, X):
        """
        Fit VB-HMM to X (N,D)
        """
        self.X = np.asarray(X)
        self.N, self.D = self.X.shape
        # initialize from data -> initial responsibilities gamma
        gamma = self._init_from_data(self.X)

        prev_elbo = -np.inf
        for it in trange(self.n_iter, desc="VB-HMM"):
            # M-step: update q(pi), q(A), q(mu,tau) using gamma and xi estimates
            Nk = gamma.sum(axis=0) + 1e-8  # effective counts per state

            # update Dirichlet for initial pi (we use gamma[0] as proxy)
            self.alpha_pi = self.alpha0 + gamma[0] * 1.0

            # update alpha_A: need xi sums; compute expected xi from transitions using current E's
            # We will compute expectations below via forward-backward using current params. For first iter we have gamma; let's compute expectations given current params.
            # Compute expectations first from current gamma
             
             
            xbar = (gamma.T @ self.X) / (Nk[:, None] + 1e-12)  # (K,D)
            S_k = np.zeros((self.K, self.D))
            for k in range(self.K):
                diffs = self.X - xbar[k]
                weighted = (gamma[:, k][:, None] * (diffs ** 2))
                S_k[k] = weighted.sum(axis=0) / (Nk[k] + 1e-12)

            # Update Normal-Gamma posteriors (per state k and dimension d)
            for k in range(self.K):
                for d in range(self.D):
                    # posterior lambda, mu
                    lam_n = self.lambda0 + Nk[k]
                    mu_n = (self.lambda0 * self.mu0 + Nk[k] * xbar[k, d]) / lam_n
                    a_n = self.a0 + 0.5 * Nk[k]
                    # b_n = b0 + 0.5 * ( Nk*var + (lambda0*Nk/(lambda0+Nk))*(xbar-mu0)^2 )
                    term = (Nk[k] * (S_k[k, d])) + (self.lambda0 * Nk[k] / (self.lambda0 + Nk[k])) * ((xbar[k, d] - self.mu0) ** 2)
                    b_n = self.b0 + 0.5 * term
                    self.lambda_kd[k, d] = lam_n
                    self.mu_kd[k, d] = mu_n
                    self.a_kd[k, d] = a_n
                    self.b_kd[k, d] = b_n

            # Now compute expectations given updated q(mu,tau) and current Dirichlet alpha_A (we'll update A below using xi)
            E_log_pi, E_log_A, E_log_N = self._expectations()

            # E-step: forward-backward to compute gamma and xi_sum
            gamma, xi_sum = self._forward_backward(E_log_pi, E_log_A, E_log_N)

            # Update alpha_A using xi_sum
            # xi_sum is expected transitions counts aggregated over time
            # Each row i should add xi_sum[i,:]
            self.alpha_A = (self.alphaA0[None, :] + xi_sum + 1e-8)

            # ELBO (optional): we can compute an approximate evidence lower bound to monitor convergence (skipped heavy terms)
            # Here we do a proxy: compute expected complete log likelihood under q
            elbo = (np.sum(gamma * (E_log_pi[None, :] + E_log_N)) +
                    np.sum(xi_sum * E_log_A))
            # check convergence
            if np.abs(elbo - prev_elbo) < self.tol:
                if self.verbose:
                    print(f"\nConverged at iter {it}, ELBO delta {elbo - prev_elbo:.6f}")
                break
            prev_elbo = elbo

        # store final posteriors
        self.gamma = gamma
        self.xi_sum = xi_sum
        # compute posterior means for parameters
        self._compute_posterior_means()

    def _compute_posterior_means(self):
        # posterior means for pi and A
        self.pi_mean = (self.alpha_pi) / (self.alpha_pi.sum())
        self.A_mean = (self.alpha_A) / (self.alpha_A.sum(axis=1, keepdims=True))

        # emission posterior means:
        # E[tau] = a/b, E[mu] = mu_n
        self.E_tau = self.a_kd / self.b_kd
        self.mu_mean = self.mu_kd.copy()
        # variance per state,d approx = 1 / (E_tau)
        self.var_mean = 1.0 / (self.E_tau + 1e-12)


    def predict_filtered_online(self, X):
        X = np.asarray(X)
        N = X.shape[0]
        K = self.K

        states = []
        probs  = []

        # log params
        log_pi = np.log(self.pi_mean + 1e-12)
        log_A = np.log(self.A_mean + 1e-12)

        # first step: forward init
        x0 = X[0:1]
        E_log_N0 = np.zeros((1, K))
        for k in range(K):
            tmp = 0
            for d in range(self.D):
                mu = self.mu_mean[k, d]
                var = self.var_mean[k, d]
                tmp += -0.5 * (np.log(2 * np.pi * var) + (x0[:, d] - mu) ** 2 / var)
            E_log_N0[:, k] = tmp
        log_alpha = log_pi + E_log_N0[0]
        log_alpha -= logsumexp(log_alpha)

        probs.append(np.exp(log_alpha))
        states.append(np.argmax(log_alpha))

        # next steps: one-bar-ahead forward
        for t in range(1, N):
            xt = X[t:t+1]
            E_log_Nt = np.zeros((1, K))
            for k in range(K):
                tmp = 0
                for d in range(self.D):
                    mu = self.mu_mean[k, d]
                    var = self.var_mean[k, d]
                    tmp += -0.5 * (np.log(2 * np.pi * var) + (xt[:, d] - mu) ** 2 / var)
                E_log_Nt[:, k] = tmp

            # forward update
            prev = log_alpha[:, None] + log_A
            log_alpha = E_log_Nt[0] + logsumexp(prev, axis=0)
            log_alpha -= logsumexp(log_alpha)

            probs.append(np.exp(log_alpha))
            states.append(np.argmax(log_alpha))

        return np.array(states), np.array(probs)


    def predict_filtered(self, X):
        """
        Compute filtered probabilities p(z_t | x_1:t) using posterior mean params.
        Uses forward algorithm with emission probabilities computed with posterior means.
        """
        X = np.asarray(X)
        N = X.shape[0]
        K = self.K
        # compute emission log-likelihoods with posterior mean parameters
        E_log_N = np.zeros((N, K))
        for k in range(K):
            tmp = 0.0
            for d in range(self.D):
                mu = self.mu_mean[k, d]
                var = self.var_mean[k, d]
                # gaussian log-lik
                tmp += -0.5 * (np.log(2 * np.pi * var) + (X[:, d] - mu) ** 2 / var)
            E_log_N[:, k] = tmp
        # forward in log space
        log_alpha = np.zeros((N, K))
        log_pi = np.log(self.pi_mean + 1e-12)
        log_A = np.log(self.A_mean + 1e-12)
        log_alpha[0] = log_pi + E_log_N[0]
        log_alpha[0] = log_alpha[0] - logsumexp(log_alpha[0])
        for t in range(1, N):
            prev = log_alpha[t-1][:, None] + log_A
            log_alpha[t] = E_log_N[t] + logsumexp(prev, axis=0)
            log_alpha[t] = log_alpha[t] - logsumexp(log_alpha[t])
        filtered_probs = np.exp(log_alpha)
        states = np.argmax(filtered_probs, axis=1)
        return states, filtered_probs

# ---------------------------
# Utilities
# ---------------------------
def logsumexp(a, axis=None):
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    s = np.exp(a - a_max)
    out = a_max + np.log(np.sum(s, axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

# ---------------------------
# Performance report (copy your robust version)
# ---------------------------
def performance_report(df,
                       name="",
                       price_col="vix_close",
                       position_col="position",
                       strategy_logret_col="strategy_logret",
                       equity_col="equity_curve",
                       bars_per_year=252*78):
    df = df.copy()
    if strategy_logret_col not in df.columns:
        df[strategy_logret_col] = 0.0
    df[strategy_logret_col] = df[strategy_logret_col].fillna(0.0)
    df[equity_col] = np.exp(df[strategy_logret_col].cumsum())
    df[equity_col] = df[equity_col].fillna(method="ffill").fillna(1.0)
    total_return = df[equity_col].iloc[-1] / df[equity_col].iloc[0] - 1.0
    try:
        n_years = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 24 * 3600)
        if n_years <= 0:
            n_years = len(df) / bars_per_year
    except Exception:
        n_years = len(df) / bars_per_year
    cagr = (df[equity_col].iloc[-1] / df[equity_col].iloc[0]) ** (1 / max(n_years, 1e-9)) - 1
    period_returns = df[equity_col].pct_change().fillna(0.0)
    vol_annual = period_returns.std() * np.sqrt(bars_per_year)
    sharpe = (period_returns.mean() * bars_per_year - 0.04) / (vol_annual + 1e-12)
    roll_max = df[equity_col].cummax()
    drawdown = df[equity_col] / roll_max - 1.0
    max_dd = drawdown.min()
    avg_dd = drawdown.mean()
    pos = df[position_col].fillna(0).astype(int)
    dif = pos.diff().fillna(0.0)
    entries = list(dif[dif > 0].index)
    exits = list(dif[dif < 0].index)
    if len(entries) == 0 and pos.iloc[0] == 1:
        entries = [df.index[0]]
    paired = []
    i_entry = 0
    i_exit = 0
    while i_entry < len(entries):
        entry_idx = entries[i_entry]
        exit_idx = None
        while i_exit < len(exits):
            if exits[i_exit] > entry_idx:
                exit_idx = exits[i_exit]
                i_exit += 1
                break
            i_exit += 1
        if exit_idx is None:
            exit_idx = df.index[-1]
        paired.append((entry_idx, exit_idx))
        i_entry += 1
    trade_returns = []
    trade_durations = []
    for entry_idx, exit_idx in paired:
        try:
            p_entry = df.loc[entry_idx, price_col]
            p_exit = df.loc[exit_idx, price_col]
        except KeyError:
            p_entry = df[price_col].iloc[df.index.get_indexer([entry_idx], method="nearest")[0]]
            p_exit  = df[price_col].iloc[df.index.get_indexer([exit_idx], method="nearest")[0]]
        if p_entry is None or p_exit is None or p_entry == 0:
            continue
        tr_ret = (p_exit / p_entry) - 1.0
        trade_returns.append(tr_ret)
        dur_bars = df.index.get_loc(exit_idx) - df.index.get_loc(entry_idx)
        trade_durations.append(dur_bars)
    n_trades = len(trade_returns)
    avg_profit_per_trade = np.mean(trade_returns) if n_trades > 0 else 0.0
    win_rate = np.mean(np.array(trade_returns) > 0) if n_trades > 0 else np.nan
    avg_duration_bars = np.mean(trade_durations) if n_trades > 0 else np.nan
    wins_sum = np.sum([r for r in trade_returns if r > 0]) if n_trades > 0 else 0.0
    losses_sum = -np.sum([r for r in trade_returns if r < 0]) if n_trades > 0 else 0.0
    profit_factor = (wins_sum / (losses_sum + 1e-12)) if losses_sum > 0 else np.nan
    metrics = {
        "Total Return": total_return,
        "CAGR": cagr,
        "Ann Vol": vol_annual,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Avg Drawdown": avg_dd,
        "Trades Count": n_trades,
        "Avg Profit/Trade": avg_profit_per_trade,
        "Win Rate": win_rate,
        "Avg Duration (bars)": avg_duration_bars,
        "Profit Factor": profit_factor
    }
    print(f"\n===== PERFORMANCE REPORT: {name} =====")
    for k, v in metrics.items():
        if isinstance(v, float) or isinstance(v, np.floating):
            print(f"{k:25s}: {v:.6f}")
        else:
            print(f"{k:25s}: {v}")
    print("======================================\n")
    plt.figure(figsize=(10,4))
    plt.plot(df.index, drawdown, label="Drawdown")
    plt.fill_between(df.index, drawdown, 0, where=(drawdown<0), color="red", alpha=0.2)
    plt.title(f"Drawdown ({name}) — MaxDD: {max_dd:.3f}")
    plt.grid(True, alpha=0.3)
    plt.show()
    return metrics, paired, trade_returns



def plot_equity_and_signals(df_period, label=""):
    """
    Trace un graphique séparé contenant :
    - Equity curve
    - Prix du VIX
    - Flèches Buy/Sell (via plot_signals)
    """
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.set_title(f"Equity Curve + VIX + Signals — {label}")

    # ----------------------
    # Equity curve (left axis)
    # ----------------------
    ax1.plot(df_period.index, df_period["equity_curve"],
             linewidth=2, color="blue", label=f"Equity {label}")

    ax1.set_ylabel("Equity")
    ax1.grid(True, alpha=0.3)

    # ----------------------
    # VIX price (right axis)
    # ----------------------
    ax2 = ax1.twinx()
    ax2.plot(df_period.index, df_period["vix_close"], color="black", linewidth=1.8,
             label=f"VIX {label}")

    ax2.set_ylabel("VIX Price")

    # ----------------------
    # Add Buy/Sell signals
    # ----------------------
   # plot_signals(ax2, df_period, label_prefix=label)

    # ----------------------
    # Legends
    # ----------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.show()

# ---------------------------
# Main script: train / infer / backtest
# ---------------------------
if __name__ == "__main__":
    # load data
    df = pd.read_csv("$VIX.X_5Minute_2011_to_2025.csv", parse_dates=["datetime"], index_col="datetime")
  #  df = pd.read_csv("vix5min_2006.csv", parse_dates=["datetime"], index_col="datetime") 
   # df = pd.read_csv("EURUSD_5min_2017_2025.csv", parse_dates=["datetime"], index_col="datetime") 
    df = df.sort_index()
    split_date = pd.to_datetime("2021-01-01").tz_localize("UTC")
    #split_date = pd.to_datetime("2022-01-01").tz_localize("UTC")
    df_train, df_test = df[df.index < split_date], df[df.index >= split_date]

# #========= with 2 files =============================
#     df_train = pd.read_csv("vxx25.csv", parse_dates=["datetime"], index_col="datetime")
#     df_test  = pd.read_csv("VXZ25.csv",  parse_dates=["datetime"], index_col="datetime")

#     # Assurer un tri temporel (utile si ce n’est pas garanti)
#     df_train = df_train.sort_index()
#     df_test  = df_test.sort_index()
# #=======================================================
    # prepare features
    X_train, idx_train, df_train_full = prepare_data_multifeat(df_train, feature_col="close")
    X_test, idx_test, df_test_full = prepare_data_multifeat(df_test, feature_col="close")
    
    PNL_MULTIPLIER = 20000.0
    K = 3
    vb = VBHMM(n_states=K, n_iter=200, tol=1e-4, verbose=True)
    vb._init_from_data(X_train)  # init internal shapes
    vb.fit(X_train)


    print("Posterior means of pi:", vb.pi_mean)
    print("Posterior means of A (first rows):\n", vb.A_mean[:,:])

    # Exemple à ajouter au main
    print("\nEmission Mean (mu_mean) per State (K x D):")
    print(vb.mu_mean)
    print("\nEmission Std Dev (sqrt(var_mean)) per State (K x D):")
    print(np.sqrt(vb.var_mean))

    # inference (filtered probabilities)
    states_train, post_train = vb.predict_filtered(X_train)
    states_test, post_test = vb.predict_filtered_online(X_test)

    # build dataframes for plotting/backtest
    df_train2 = pd.DataFrame(index=idx_train)
    df_train2["state"] = states_train
    df_train2["vix_close"] = df_train_full["close"].loc[idx_train]
    df_test2 = pd.DataFrame(index=idx_test)
    df_test2["state"] = states_test
    df_test2["vix_close"] = df_test_full["close"].loc[idx_test]

    # Plot regimes & posterior probs (train+test)
    state_colors = {0: "green", 1: "orange", 2: "red"}
    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    plt.title("VIX regimes (VB-HMM) — train + test")
    plt.plot(df_train2.index, df_train2["vix_close"], color="blue", alpha=0.4, label="VIX (train)")
    plt.plot(df_test2.index, df_test2["vix_close"], color="black", alpha=0.4, label="VIX (test)")
    for st in sorted(df_train2["state"].unique()):
        mask = df_train2["state"] == st
        plt.scatter(df_train2.index[mask], df_train2["vix_close"][mask], color=state_colors[st], s=10, label=f"État {st} (train)")
    for st in sorted(df_test2["state"].unique()):
        mask = df_test2["state"] == st
        plt.scatter(df_test2.index[mask], df_test2["vix_close"][mask], color=state_colors[st], marker="x", s=30, label=f"État {st} (test)")
    plt.legend(loc="upper left")

    # plt.subplot(2,1,2)
    # for k in range(K):
    #     plt.plot(idx_train, post_train[:, k], color=state_colors[k], alpha=0.7, label=f"P(state={k}) train" if k==0 else None)
    #     plt.plot(idx_test, post_test[:, k], color=state_colors[k], alpha=0.9, linestyle="--", label=f"P(state={k}) test" if k==0 else None)
    # plt.ylim(0,1.02)
    # plt.title("Filtered probabilities per state")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    # -----------------------------------------------------
    # PLOT 2 : État Dominant (le plus probable)
    # -----------------------------------------------------
    
    # Déterminer l'état dominant (état le plus probable) pour chaque point temporel
    dominant_state_train = np.argmax(post_train, axis=1)
    dominant_state_test = np.argmax(post_test, axis=1)

    # Récupérer la probabilité de l'état dominant pour chaque point temporel
    max_prob_train = np.max(post_train, axis=1)
    max_prob_test = np.max(post_test, axis=1)
    
    plt.subplot(2,1,2)
    plt.title("Probabilité de l'État Dominant")

    # Tracer les probabilités de l'état dominant, en colorant selon l'état lui-même
    for st in range(K):
        # TRAIN SET
        mask_train = dominant_state_train == st
        if np.any(mask_train):
            # Trace uniquement les points où l'état 'st' est dominant
            plt.plot(idx_train[mask_train], max_prob_train[mask_train],
                     color=state_colors[st],
                     linestyle='-',
                     linewidth=2.5,
                     alpha=0.8,
                     label=f"Prob. État {st} (Train)" if st == 0 else None)

        # TEST SET
        mask_test = dominant_state_test == st
        if np.any(mask_test):
            plt.plot(idx_test[mask_test], max_prob_test[mask_test],
                     color=state_colors[st],
                     linestyle='--',
                     linewidth=1.5,
                     alpha=0.9,
                     label=f"Prob. État {st} (Test)" if st == 0 else None)
    
    # Ajout d'une ligne pour le seuil
    plt.axhline(0.5, color='gray', linestyle=':', linewidth=1.0, label='Seuil P=0.5')
    
    plt.ylim(0, 1.02)
    plt.ylabel("P(État Dominant)")
    plt.legend()
    plt.tight_layout()
    plt.show()




    # Backtest simple: long when state == best_state and posterior > threshold
    # Determine best_state on train by looking at mean returns when in state
    # Here as simple heuristic choose state with highest avg subsequent positive return
    df_train2["logret"] = np.log(df_train2["vix_close"] / df_train2["vix_close"].shift(1))
    avg_ret_per_state = []
    for k in range(K):
        avg_ret_per_state.append(df_train2.loc[df_train2["state"]==k, "logret"].mean())
    best_state = int(np.argmax(avg_ret_per_state))
    print("Best state by avg logret in train:", best_state)


#================= PROFIT AND LOSS IN PERCENTAGE ===============================
    # # positions using posterior > threshold
    threshold = 0.6
    df_train2["pos"] = (post_train[:, best_state] > threshold).astype(int)
    df_train2["pos"] = df_train2["pos"].shift(1).fillna(0).astype(int)
    df_train2["strategy_logret"] = df_train2["pos"] * df_train2["logret"]
   # df_train2["equity_curve"] = np.exp(df_train2["strategy_logret"].cumsum())
    

    df_test2["logret"] = np.log(df_test2["vix_close"] / df_test2["vix_close"].shift(1))
    df_test2["pos"] = (post_test[:, best_state] > threshold).astype(int)
    df_test2["pos"] = df_test2["pos"].shift(1).fillna(0).astype(int)
    df_test2["strategy_logret"] = df_test2["pos"] * df_test2["logret"]
    #df_test2["equity_curve"] = np.exp(df_test2["strategy_logret"].cumsum())
#=========================================================================================


#========================= PROFIT AND LOSS WITH LEVERAGE 1% = $200 ============================================
# #    # df_train2["logret"] = np.log(df_train2["vix_close"] / df_train2["vix_close"].shift(1))
    
    # 1. Calcul du Rendement Simple (nécessaire pour le PnL)
    df_train2["simple_ret"] = df_train2["vix_close"] / df_train2["vix_close"].shift(1) - 1.0
    df_train2["simple_ret"] = df_train2["simple_ret"].fillna(0.0) # S'assurer que le premier rendement est 0
    
    # 2. Position et Signal
    df_train2["pos"] = (post_train[:, best_state] > threshold).astype(int)
    df_train2["pos"] = df_train2["pos"].shift(1).fillna(0).astype(int)
    
    # 3. Calcul du PnL en dollars pour la période
    # PnL = Signal * Rendement Simple * Multiplicateur
    df_train2["pnl_period"] = df_train2["pos"] * df_train2["simple_ret"] * PNL_MULTIPLIER
    
    # 4. Calcul de l'Equity Curve (PnL cumulé, départ de 0)
    df_train2["equity_curve"] = df_train2["pnl_period"].cumsum()
    
    # 1. Calcul du Rendement Simple
    df_test2["simple_ret"] = df_test2["vix_close"] / df_test2["vix_close"].shift(1) - 1.0
    df_test2["simple_ret"] = df_test2["simple_ret"].fillna(0.0)
    
    # 2. Position et Signal
    df_test2["pos"] = (post_test[:, best_state] > threshold).astype(int)
    df_test2["pos"] = df_test2["pos"].shift(1).fillna(0).astype(int)
    
    # 3. Calcul du PnL en dollars pour la période
    df_test2["pnl_period"] = df_test2["pos"] * df_test2["simple_ret"] * PNL_MULTIPLIER
    
    # 4. Calcul de l'Equity Curve (PnL cumulé, départ de 0)
    df_test2["equity_curve"] = df_test2["pnl_period"].cumsum()

    df_train_report = df_train2.rename(columns={"pos":"position"})
    # strategy_logret = log(1 + simple_ret_strategy)
    df_train_report["strategy_ret"] = df_train_report["pnl_period"] / PNL_MULTIPLIER # Rendement équivalent
    df_train_report["strategy_logret"] = np.log(1 + df_train_report["strategy_ret"])
    
    # Test
    df_test_report = df_test2.rename(columns={"pos":"position"})
    df_test_report["strategy_ret"] = df_test_report["pnl_period"] / PNL_MULTIPLIER
    df_test_report["strategy_logret"] = np.log(1 + df_test_report["strategy_ret"])

#============================================================================================================



    # plot equity
    plt.figure(figsize=(12,5))
    plt.plot(df_train2.index, df_train2["equity_curve"], label="Equity Train")
    plt.plot(df_train2.index, df_train2["vix_close"], color="black", linewidth=1.8, label="VIX (train)")
    
    plt.plot(df_test2.index, df_test2["equity_curve"], label="Equity Test")
    plt.plot(df_test2.index, df_test2["vix_close"], color="black",linewidth=1.8, label="VIX (test)")
  #  plt.axvline(split_date, color="red", linestyle="--")
    plt.legend()
    plt.title("Equity curves (VB-HMM strategy, threshold {})".format(threshold))
    plt.show()

    plot_equity_and_signals(df_train2, "TRAIN")
    plot_equity_and_signals(df_test2, "TEST")


    # Perf reports
    performance_report(df_train2.rename(columns={"pos":"position"}), name="Train")
    performance_report(df_test2.rename(columns={"pos":"position"}), name="Test")

    joblib.dump(vb, "vb__future_model.pkl")
