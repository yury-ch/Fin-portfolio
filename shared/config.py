# shared/config.py
# Shared algorithmic constants used across analysis_engine and calculation_service.

# Risk-free rate used for Sharpe ratio calculation in screening (analysis_engine)
# and in calculate_portfolio_metrics. The portfolio optimizer (optimize_portfolio)
# accepts a user-supplied value via PortfolioOptimizationRequest.risk_free and is
# not governed by this constant.
DEFAULT_RISK_FREE_RATE: float = 0.0
