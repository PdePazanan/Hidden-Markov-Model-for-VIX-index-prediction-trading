# Hidden Markov Model for VIX index prediction trading

![Texte alternatif](Results_graphs/daily_trading_5min_timeframe.png)

## Operation of the model
Markov chains are significantly used today in the market to benefit investors and traders. I developed here a Hidden Markov Model that would help us identify different hidden states in the market and use it to develop a trading strategy.
The model estimates three hidden states : normal trading, top formation, and bottom formation using market observations like returns, price levels, volatility, momentum... It tracks shifting market structures while being sensitive to different volatility and regime stability by dynamically updating state probabilities using Bayesian inference and adaptive transition matrices to calculate top, normal and bottom probability. 

- Our model research the 2 states top (where the market is overbought) and bottom (where the market is oversell). It is based on the asumption that the VIX index is stationnary and the model is seeking to predict potential reversal point.
The model is defined in the file hmm_model.py and for more precision the mathematics of the model are explained in the file hmm.pdf


- The second model is obtained thanks to a LSTM with a MLP that had been trained with market features and the probability from the Hidden Markov Model, The folder LSTM contain the necessary to train the model, we trained the model for the VIX with a 2-hours timeframe.
- 
The graphs results are in the folder results_graphs. I used for one model the VVIX who is an index that mesure the expected volatility of the VIX index, it is calculated using the same methodology as the VIX, but instead of relying on S&P 500 options, it uses prices from VIX options. So we have 2 models, one using the VVIX as a feature for more precision, our model try to detect patterns and correlation between VVIX and VIX, it is also combined with the HMM. The other model is just the LSTM combined with the HMM. Bellow a diagram that explained the different layers 👇 

![Texte alternatif](Results_graphs/LSTM.png)

Our model is learning with a lot of accuracy because we focus just on the VIX market and this index is oscillating between $0 and $100, that's why the model is able to detect patterns, This forward-looking approach enables us to identify regime shifts before they are reflected in price movements, giving our trades an informational advantage.

## Live application for future as a short-term trading strategy
For a short term strategy, the VIX index is available in Future in TradeStation, the important folder is TS_Live. The model is defined in the file hmm_model and the live algorithm is Live_trader.py.
To launch this live algorithm you need to complete the Execution_API with your API informations :

CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN, TOKEN_FILE, ACCOUNT_ID

The other parameters are filled for the VIX index in 5 min data.

Since the VIX index is not directy tradable, we will trade the future of the current month, that's why our strategy is based on 5 minutes timeframe because we don't want our trades to last too long. Even though the results were better using a bigger timeframe and using a long term strategy, but in reality with Future we will need to pay rollover fees and we need to sell the contract at the end of the month.

## Live application for medium-term trading strategy

Since the long term results were much better, we had better profits and less drawdowns, with out of sample data we can have an incredible 70% annual return by putting a slippage of 0.1% for every trade taken, this performance will be a little lower if the swaps for the DFC VIX are high. We can trade the VIX index with CFDs, it's not a Future contract so there is no expiry date, but there is still fees for swaps every night. Vantage market propose the VIX index as a CFD.
The best model trained with LSTM is in the folder LSTM, and give the signals combined with the HMM.
I'm working to develop a code to integrate a Live trading strategy with Vantage market were the VIX as a CFD is available, and to integrate the fees of swaps in our backtest.

### Data
You can also find intraday data for the VIX index and VVIX index since 2011, in 5 min timeframe or 2 hours timeframe. It is impossible to find these data for free on the internet. You're welcome

-
-
-

If you have suggestion or criticism i will be happy to here them. I did this for free, if you want to support me you can do it here 👇👇 

BTC wallet adress (BTC network) : bc1q5hpk5m03wu76e3psddg7my6pglkpxjry276ymx

USDC wallet adress (Base network) : 0x31A4733Cf72fc998101f011fA289aA4Dd38Ba509
