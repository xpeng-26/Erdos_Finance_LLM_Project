# AI-Driven Stock Trading using Reinforcement Learning and Large Language Model

# Team Members
[Xiangwei Peng](https://github.com/xpeng-26)

[Xiaokang Wang](https://github.com/Mathheadkang)

[Xu Zhuang](https://github.com/zxmath)

# Introduction
This project is for the [Erdös Institude](https://www.erdosinstitute.org) 2025 Spring deep learning boot camp.

This project aims to predict stock price movements by combining financial news sentiment analysis using a local LLM with deep reinforcement learning–based trading strategies. It provides an end-to-end configurable pipeline covering data ingestion, feature engineering, agent training, and performance evaluation.

## Source Code Structure
- **config/**: Configuration templates (`trade_w_llm.toml`) defining pipeline flags, data paths, and model hyperparameters.
- **src/ingestion/**: Modules for fetching and storing raw stock price and news data.
- **src/feature/**: Feature engineering drivers for technical indicators (`calculate_factors`) and AI-driven sentiment advisory (`inference_ai_sentiment_advisory`).
- **src/strategy/**: Reinforcement learning setup including Gymnasium environments and agent implementations (DDQN, PPO, A2C).
- **src/evaluation/**: Backtesting framework to evaluate agent performance against market baseline.
- **src/utils/**: Utility functions for configuration parsing, logging, and directory management.
- **Notebooks/**: Jupyter notebooks for exploratory data analysis and result visualization.


## Key Models and Frameworks

- **Technique Indicators**: TA-Lib for technical factors (e.g., RSI, MACD, Bollinger Bands).
- **Large Language Model**: 
  - Gemma-3 model for sentiment scoring and advisory (configured via `feature.llm_model`).
  - llama.cpp for local inference.
- **Reinforcement Learning Agents**:
  - **DDQN**: Double DQN implemented with PyTorch for discrete-action single-asset trading.
  - **PPO** & **A2C**: Policy-based methods from Stable Baselines3.
  - **Trading Environments**: Built with Gymnasium for single-asset (`trading-v0`) and portfolio (`trading-port-v0`) scenarios.
 
## Trading environment
We use the **net asset value (NAV)** and the **Sharpe Ratio** as our evaluation metric. The trading rules are as follows:
- We start at day 1 with NAV 1;
- Everyday our observation space will be the techinical indicators and the LLM news factors;
- At the end of the day, we will re-allocate the asset to all of the stocks. Here we allow shortselling. Mathematically, we will calculate the new weights of every stocks and normalize the sum of the weight to be between -1 and 1;
- The strategy return will be computed using the next-day market return times the weights;
- The NAV will be updated by multiplying the previous NAV with one plus the strategy return;
- The Sharpe ratio will be computed using the daily return information of the trading strategy. 

## Experiment and Results

### Data

- Run experiement on Apple, Costco, Chevron from 2022-03-01 to 2025-02-28 (2 years as training period and the last year
  as testing period). 
- Stock price: Yahoo Finance API
- News data: Alpha Ventage API

### Experiment

- Single vs. Multiple Stocks (Portfolio)
- Tech Indicators only vs. Tech Indicators + News Sentiment Score

### Results



## Conclusions

- PPO has the best performance.
- DDQN agents exhibit consistent returns with lower volatility, while PPO/A2C methods offer faster convergence during training.
- Integrating LLM-driven sentiment features doesn't predictive signals when paired with traditional technical
  indicators. This indicate more complex works is needed to extract news information.
