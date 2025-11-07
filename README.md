# Hidden Markov Model for VIX index prediction trading

![Texte alternatif](Results_graphs/Screenshot.png)

## Functionnment of the model

The model is using euristic emission to calculate hidden states in the VIX market. The 3 hidden states are top, normal, bottom. Our model research the 2 states top (where the market is overbought) and bottom (where the market is oversell). It is based on the asumption that the VIX index is stationnary and the model is seeking to predict potential reversal point, we use different features from the market and Bayesian inference to calculate the top, normal and bottom probability . The model is defined in the file hmm_model.py and the mathematics of the model are explained in the file hmm.pdf

The second model is a LSTM that had been trained with market features and the probability from the Hidden Markov Model, you can find the different file in the folder TradeStation

## Live application for future as a short term strategy
For a short term strategy, the VIX index is available in Future in TradeStation, the important folder is TS_Live. The model is defined in the file hmm_model and the live algorithm is Live_trader.py.
To launch this live algorithm you need to complete the Execution_API with your API informations :

CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN, TOKEN_FILE, ACCOUNT_ID

The other parameters are filled for the VIX index in 5 min data.

Since the VIX index is not directy tradable, we will trade the future of the current month, that's why our strategy is based on 5 minutes timeframe because we don't want our trades to last too long. Even though the results were better using a bigger timeframe and using a long term strategy, but in reality with Future we will need to pay rollover fees and we need to sell the contract at the end of the month.

## Live application for long term strategy

Since the long term results were much better, we had better profits and less drawdowns, we can trade the VIX index with CFDs, it's not a Future contract so there is no expiry date, but there is still fees for swaps every night. Vantage market propose the VIX index as a CFD.
The best model trained with LSTM is in the folder LSTM, and give the signals combined with the HMM.

The folder Projet_HMM is mainly a draft so it is not very important.

The folder Tradestation is important and explain the backtest of the different trategies


If you have suggestion or criticism i will be happy to here them. I did this for free, if you want to support me you can do it here 👇👇 
BTC wallet adress (BTC network) : bc1q5hpk5m03wu76e3psddg7my6pglkpxjry276ymx
