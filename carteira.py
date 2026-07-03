import pandas as pd
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov

tickers = ["EGIE3.SA", "ITSA4.SA", "BBAS3.SA", "PSSA3.SA", "TAEE11.SA"]
dados = yf.download(tickers, period="10y")['Close']
dados

retornos = mean_historical_return(dados)
retornos

# matriz de covariância
cov_matrix = sample_cov(dados)
cov_matrix

# -- Carteira 1: Sharpe Máximo -- #
ef_sharpe = EfficientFrontier(retornos, cov_matrix)
weights_sharpe = ef_sharpe.max_sharpe()
df_weights_sharpe = pd.Series(weights_sharpe)
df_weights_sharpe.plot(title="Pesos Carteira Sharpe", kind="pie", autopct='%1.1f%%', figsize=(6, 6))

ef_sharpe.portfolio_performance(verbose=True)

# -- Carteira 2: Risco Mínimo/Volatilidade Mínima -- #
ef_volat = EfficientFrontier(retornos, cov_matrix)
weights_volat = ef_volat.min_volatility()
df_weights_volat = pd.Series(weights_volat)
df_weights_volat.plot(title="Pesos Carteira Volatilidade Mínima", kind="pie", autopct='%1.1f%%', figsize=(6, 6))

ef_volat.portfolio_performance(verbose=True)