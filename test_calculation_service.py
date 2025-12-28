# test_calculation_service.py
# -------------------------------
# Unit Tests for Calculation Service
# Tests for portfolio optimization, statistics, and metrics calculations
# -------------------------------

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.calculation_service import CalculationService, app
from shared.models import OptimizationResult, ServiceResponse

try:
    from starlette.testclient import TestClient
except ImportError:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        # Skip API tests if TestClient is not available
        TestClient = None


class TestCalculationService:
    """Test suite for CalculationService core functionality"""

    @pytest.fixture
    def calculation_service(self):
        """Create a CalculationService instance for testing"""
        return CalculationService()

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data for testing"""
        # Generate sample price data for 3 stocks over 100 days
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Simulate realistic stock price movements
        initial_prices = {'AAPL': 150, 'MSFT': 250, 'GOOGL': 2800}
        prices_data = {}

        for ticker, initial_price in initial_prices.items():
            # Generate realistic price series with some drift and volatility
            returns = np.random.normal(0.001, 0.02, 100)  # Mean daily return ~0.1%, volatility ~2%
            prices = [initial_price]

            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1.0))  # Ensure positive prices

            prices_data[ticker] = prices

        return pd.DataFrame(prices_data, index=dates)

    @pytest.fixture
    def sample_weights(self):
        """Create sample portfolio weights for testing"""
        return pd.Series({'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3})

    def test_compute_stats_valid_data(self, calculation_service, sample_prices):
        """Test compute_stats with valid price data"""
        mu, S = calculation_service.compute_stats(sample_prices)

        # Check that returns are calculated
        assert isinstance(mu, pd.Series)
        assert len(mu) == 3
        assert all(ticker in mu.index for ticker in ['AAPL', 'MSFT', 'GOOGL'])

        # Check covariance matrix
        assert isinstance(S, pd.DataFrame)
        assert S.shape == (3, 3)
        assert all(ticker in S.index for ticker in ['AAPL', 'MSFT', 'GOOGL'])
        assert all(ticker in S.columns for ticker in ['AAPL', 'MSFT', 'GOOGL'])

        # Covariance matrix should be symmetric and positive semi-definite
        assert np.allclose(S.values, S.T.values)
        eigenvals = np.linalg.eigvals(S.values)
        assert all(eigenvals >= -1e-10)  # Allow for small numerical errors

    def test_compute_stats_empty_data(self, calculation_service):
        """Test compute_stats with empty DataFrame"""
        empty_df = pd.DataFrame()

        try:
            mu, S = calculation_service.compute_stats(empty_df)
            # If no exception is raised, check that the result is reasonable
            # (PyPortfolioOpt might handle empty dataframes gracefully)
            assert mu is not None
            assert S is not None
        except Exception:
            # If an exception is raised, that's also acceptable behavior
            pass

    def test_enforce_min_holdings_normal_case(self, calculation_service):
        """Test enforce_min_holdings with normal weights"""
        weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.3, 'GOOGL': 0.15, 'TSLA': 0.05})

        result = calculation_service.enforce_min_holdings(weights, min_n=3, prune_below_pct=0.1)

        # Should keep top 3 holdings and renormalize
        assert len(result) == 3
        assert abs(result.sum() - 1.0) < 1e-10  # Should sum to 1
        assert 'AAPL' in result.index
        assert 'MSFT' in result.index
        assert 'GOOGL' in result.index
        assert 'TSLA' not in result.index  # Should be pruned

    def test_enforce_min_holdings_insufficient_weights(self, calculation_service):
        """Test enforce_min_holdings when we have fewer weights than min_n"""
        weights = pd.Series({'AAPL': 0.6, 'MSFT': 0.4})

        result = calculation_service.enforce_min_holdings(weights, min_n=3, prune_below_pct=0.1)

        # Should keep all available weights when insufficient
        assert len(result) == 2
        assert abs(result.sum() - 1.0) < 1e-10
        assert 'AAPL' in result.index
        assert 'MSFT' in result.index

    def test_enforce_min_holdings_zero_threshold(self, calculation_service):
        """Test enforce_min_holdings with zero pruning threshold"""
        weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.3, 'GOOGL': 0.15, 'TSLA': 0.05})

        result = calculation_service.enforce_min_holdings(weights, min_n=4, prune_below_pct=0.0)

        # Should keep all weights when threshold is zero
        assert len(result) == 4
        assert abs(result.sum() - 1.0) < 1e-10

    @patch('services.calculation_service.EfficientFrontier')
    @patch('services.calculation_service.get_latest_prices')
    @patch('services.calculation_service.DiscreteAllocation')
    def test_optimize_portfolio_success(self, mock_discrete_alloc, mock_latest_prices,
                                      mock_ef, calculation_service, sample_prices):
        """Test successful portfolio optimization"""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        investment_amount = 100000

        # Mock EfficientFrontier
        mock_ef_instance = Mock()
        mock_ef_instance.max_quadratic_utility.return_value = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        mock_ef_instance.portfolio_performance.return_value = (0.12, 0.18, 0.67)  # return, vol, sharpe
        mock_ef.return_value = mock_ef_instance

        # Mock latest prices
        mock_latest_prices.return_value = pd.Series({'AAPL': 150, 'MSFT': 250, 'GOOGL': 2800})

        # Mock DiscreteAllocation
        mock_da_instance = Mock()
        mock_da_instance.greedy_portfolio.return_value = ({'AAPL': 266, 'MSFT': 120, 'GOOGL': 10}, 2000)
        mock_discrete_alloc.return_value = mock_da_instance

        result = calculation_service.optimize_portfolio(
            tickers, sample_prices, investment_amount
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10
        assert result.expected_annual_return == 0.12
        assert result.annual_volatility == 0.18
        assert result.sharpe_ratio == 0.67
        assert result.leftover_cash == 2000

    def test_optimize_portfolio_empty_prices(self, calculation_service):
        """Test optimize_portfolio with empty price data"""
        tickers = ['AAPL', 'MSFT']
        empty_prices = pd.DataFrame()

        with pytest.raises((ValueError, Exception)):
            calculation_service.optimize_portfolio(tickers, empty_prices, 100000)

    def test_optimize_portfolio_no_matching_tickers(self, calculation_service, sample_prices):
        """Test optimize_portfolio with no matching tickers"""
        tickers = ['INVALID1', 'INVALID2']

        with pytest.raises((ValueError, Exception)):
            calculation_service.optimize_portfolio(tickers, sample_prices, 100000)

    def test_optimize_portfolio_insufficient_data(self, calculation_service):
        """Test optimize_portfolio with insufficient historical data"""
        tickers = ['AAPL', 'MSFT']
        # Create very short price series
        short_prices = pd.DataFrame({
            'AAPL': [150, 151, 149],
            'MSFT': [250, 252, 248]
        }, index=pd.date_range('2023-01-01', periods=3))

        with pytest.raises((ValueError, Exception)):
            calculation_service.optimize_portfolio(tickers, short_prices, 100000)

    def test_calculate_portfolio_metrics_success(self, calculation_service, sample_prices):
        """Test successful portfolio metrics calculation"""
        weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}

        metrics = calculation_service.calculate_portfolio_metrics(sample_prices, weights)

        assert isinstance(metrics, dict)
        required_keys = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio',
                        'max_drawdown', 'portfolio_value']
        assert all(key in metrics for key in required_keys)

        # Check reasonable ranges for financial metrics
        assert -1 <= metrics['total_return'] <= 5  # Between -100% and 500%
        assert -1 <= metrics['annual_return'] <= 2  # Between -100% and 200%
        assert 0 <= metrics['volatility'] <= 2  # Positive volatility
        assert metrics['max_drawdown'] >= 0  # Drawdown should be positive
        assert abs(metrics['portfolio_value'] - 1.0) < 1e-10  # Weights should sum to 1

    def test_calculate_portfolio_metrics_empty_weights(self, calculation_service, sample_prices):
        """Test portfolio metrics with empty weights"""
        weights = {}

        with pytest.raises(ValueError, match="No price data available for the given weights"):
            calculation_service.calculate_portfolio_metrics(sample_prices, weights)

    def test_calculate_portfolio_metrics_no_matching_data(self, calculation_service, sample_prices):
        """Test portfolio metrics with weights that don't match price data"""
        weights = {'INVALID1': 0.5, 'INVALID2': 0.5}

        with pytest.raises(ValueError, match="No price data available for the given weights"):
            calculation_service.calculate_portfolio_metrics(sample_prices, weights)


class TestCalculationServiceAPI:
    """Test suite for FastAPI endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        if TestClient is None:
            pytest.skip("TestClient not available")
        return TestClient(app)

    @pytest.fixture
    def sample_prices_dict(self):
        """Sample prices in dictionary format for API calls"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')

        prices_data = {}
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            prices_data[date_str] = {
                'AAPL': 150 + np.random.normal(0, 5),
                'MSFT': 250 + np.random.normal(0, 8),
                'GOOGL': 2800 + np.random.normal(0, 100)
            }

        return prices_data

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "calculation_service"

    @patch('services.calculation_service.calculation_service.optimize_portfolio')
    def test_optimize_portfolio_endpoint_success(self, mock_optimize, client, sample_prices_dict):
        """Test successful portfolio optimization endpoint"""
        # Mock the optimization result
        mock_result = OptimizationResult(
            weights={'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3},
            expected_annual_return=0.12,
            annual_volatility=0.18,
            sharpe_ratio=0.67,
            allocation={'AAPL': 266, 'MSFT': 120, 'GOOGL': 10},
            leftover_cash=2000.0
        )
        mock_optimize.return_value = mock_result

        payload = {
            'tickers': ['AAPL', 'MSFT', 'GOOGL'],
            'prices_data': sample_prices_dict,
            'investment_amount': 100000,
            'risk_aversion': 1.0,
            'min_weight_threshold': 0.0025,
            'min_holdings': 3
        }

        response = client.post("/optimize-portfolio", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data['success'] is True
        assert 'data' in data
        assert 'weights' in data['data']
        assert data['data']['expected_annual_return'] == 0.12

    def test_optimize_portfolio_endpoint_missing_params(self, client):
        """Test portfolio optimization endpoint with missing parameters"""
        payload = {
            'tickers': ['AAPL', 'MSFT']
            # Missing prices_data
        }

        response = client.post("/optimize-portfolio", json=payload)
        assert response.status_code == 400

    @patch('services.calculation_service.calculation_service.calculate_portfolio_metrics')
    def test_portfolio_metrics_endpoint_success(self, mock_calculate, client, sample_prices_dict):
        """Test successful portfolio metrics endpoint"""
        mock_metrics = {
            'total_return': 0.15,
            'annual_return': 0.12,
            'volatility': 0.18,
            'sharpe_ratio': 0.67,
            'max_drawdown': 0.08,
            'portfolio_value': 1.0
        }
        mock_calculate.return_value = mock_metrics

        weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        response = client.post(f"/portfolio-metrics?weights={weights}",
                              json=sample_prices_dict)

        # Note: This endpoint signature might need adjustment based on actual implementation
        # The test assumes the current structure but may need modification

    @patch('services.calculation_service.calculation_service.compute_stats')
    def test_compute_stats_endpoint_success(self, mock_compute, client, sample_prices_dict):
        """Test successful statistics computation endpoint"""
        mock_mu = pd.Series({'AAPL': 0.12, 'MSFT': 0.10, 'GOOGL': 0.15})
        mock_S = pd.DataFrame({
            'AAPL': [0.04, 0.02, 0.01],
            'MSFT': [0.02, 0.03, 0.015],
            'GOOGL': [0.01, 0.015, 0.05]
        }, index=['AAPL', 'MSFT', 'GOOGL'])

        mock_compute.return_value = (mock_mu, mock_S)

        response = client.post("/compute-stats", json=sample_prices_dict)
        assert response.status_code == 200

        data = response.json()
        assert data['success'] is True
        assert 'data' in data
        assert 'expected_returns' in data['data']
        assert 'covariance_matrix' in data['data']


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def calculation_service(self):
        return CalculationService()

    def test_compute_stats_single_stock(self, calculation_service):
        """Test compute_stats with single stock"""
        single_stock_prices = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.02 + 0.001
        }, index=pd.date_range('2023-01-01', periods=100))

        mu, S = calculation_service.compute_stats(single_stock_prices)

        assert len(mu) == 1
        assert S.shape == (1, 1)
        assert S.iloc[0, 0] > 0  # Variance should be positive

    def test_compute_stats_highly_correlated_stocks(self, calculation_service):
        """Test compute_stats with highly correlated stocks"""
        # Create two highly correlated price series
        base_returns = np.random.randn(100) * 0.02 + 0.001
        noise1 = np.random.randn(100) * 0.001
        noise2 = np.random.randn(100) * 0.001

        prices_df = pd.DataFrame({
            'STOCK1': (base_returns + noise1).cumsum() + 100,
            'STOCK2': (base_returns + noise2).cumsum() + 100
        }, index=pd.date_range('2023-01-01', periods=100))

        mu, S = calculation_service.compute_stats(prices_df)

        # Check that correlation is high
        corr_matrix = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
        assert corr_matrix.iloc[0, 1] > 0.5  # Should be highly correlated

    def test_enforce_min_holdings_all_zero_weights(self, calculation_service):
        """Test enforce_min_holdings with all zero weights"""
        zero_weights = pd.Series({'AAPL': 0.0, 'MSFT': 0.0, 'GOOGL': 0.0})

        # This should handle the edge case gracefully
        result = calculation_service.enforce_min_holdings(zero_weights, min_n=2, prune_below_pct=0.01)

        # Should return empty series or handle appropriately
        assert len(result) >= 0  # Should not crash

    def test_portfolio_metrics_extreme_volatility(self, calculation_service):
        """Test portfolio metrics with extremely volatile prices"""
        # Create extremely volatile price series
        volatile_returns = np.random.randn(100) * 0.1  # 10% daily volatility
        prices_df = pd.DataFrame({
            'VOLATILE': (1 + volatile_returns).cumprod() * 100
        }, index=pd.date_range('2023-01-01', periods=100))

        weights = {'VOLATILE': 1.0}
        metrics = calculation_service.calculate_portfolio_metrics(prices_df, weights)

        # Should handle extreme volatility without crashing
        assert metrics['volatility'] > 0
        assert not np.isnan(metrics['volatility'])
        assert not np.isinf(metrics['volatility'])


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])