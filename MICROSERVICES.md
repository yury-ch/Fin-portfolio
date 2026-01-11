# S&P 500 Portfolio Optimizer - Microservices Architecture

This document describes the refactored microservices architecture for the S&P 500 Portfolio Optimizer.

## Architecture Overview

The application has been decomposed into four services:

### 0. Ticker Service (`services/ticker_service.py`)
- **Port:** 8000
- **Responsibilities:**
  - Download the official Wikipedia S&P 500 constituents table
  - Persist cached tickers under `sp500_data/sp500_constituents.csv` (legacy Nasdaq cache auto-detected)
  - Provide diagnostics and manual refresh endpoints
- **Validators:** Run `python services/ticker_validation_service.py` monthly to compare cached vs Wikipedia constituents and save reports under `sp500_data/validation/`.
- **Endpoints:**
  - `GET /health` - Health check and cache metadata
  - `GET /sp500-tickers` - Return cached or freshly downloaded tickers
  - `POST /refresh` - Force an immediate download
- **API Documentation:** http://localhost:8000/docs

### 1. Data Service (`services/data_service.py`)
- **Port:** 8001
- **Responsibilities:**
  - Yahoo Finance API integration via yfinance
  - Data caching and persistence (Parquet files)
  - S&P 500 stock analysis
  - Rate limiting and batch processing
- **Endpoints:**
  - `GET /health` - Health check
  - `POST /stock-data` - Get stock price data
  - `POST /sp500-analysis` - Get S&P 500 analysis
  - `GET /cache-info` - Get cache status
  - `GET /sp500-tickers` - Get S&P 500 sample tickers
- **API Documentation:** http://localhost:8001/docs

### 2. Calculation Service (`services/calculation_service.py`)
- **Port:** 8002
- **Responsibilities:**
  - Portfolio optimization using PyPortfolioOpt
  - Statistical calculations (expected returns, covariance)
  - Risk metrics and portfolio performance analysis
- **Endpoints:**
  - `GET /health` - Health check
  - `POST /optimize-portfolio` - Optimize portfolio allocation
  - `POST /portfolio-metrics` - Calculate portfolio metrics
  - `POST /compute-stats` - Compute statistical measures
- **API Documentation:** http://localhost:8002/docs

### 3. Presentation Service (`services/presentation_service.py`)
- **Streamlit Application**
- **Responsibilities:**
  - User interface and interaction handling
  - Service orchestration and communication
  - Data visualization and presentation
- **Features:**
  - Service health monitoring
  - Real-time communication with backend services
  - Comprehensive portfolio analysis UI

### 4. Price Sync Service (`services/price_sync_service.py`)
- **Type:** CLI worker (run via cron/shell)
- **Responsibilities:**
  - Download Yahoo Finance price history for the S&P universe with an initial 5-year backfill
  - Persist parquet caches under `sp500_data/price_cache/` and append weekly deltas instead of re-downloading full horizons
  - Refresh the cached data asynchronously so the UI never blocks on Yahoo calls
- **Execution:**
  - Run manually with `./run-price-sync.sh`
  - Schedule via cron every Monday 23:00 CET (see instructions below)
- **Configuration:** Periods/intervals are configurable via CLI flags; defaults mirror the analyzer horizons (`1y`, `2y`, `3y` at `1d` interval).

### 5. Analysis Sync Service (`services/analysis_sync_service.py`)
- **Type:** CLI worker (run via cron/shell)
- **Responsibilities:**
  - Read cached Yahoo prices and compute derived metrics (return, volatility, Sharpe, drawdown, etc.)
  - Persist `sp500_analysis_<period>.parquet` + metadata so the web app only serves precomputed data
- **Execution:**
  - Run manually with `./run-analysis-sync.sh`
  - Trigger right after every price sync (same cron entry or via `run-price-sync.sh && run-analysis-sync.sh`)
- **Notes:** The data service no longer performs on-demand analysis; if the cache is stale/missing it instructs operators to rerun the analysis sync.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements-microservices.txt
```

### 2. Start All Services
```bash
python start-microservices.py
```

This will start:
- Ticker Service on http://localhost:8000
- Data Service on http://localhost:8001
- Calculation Service on http://localhost:8002

### 3. Start Presentation Layer
In a separate terminal:
```bash
streamlit run services/presentation_service.py
```

### 4. Access the Application
- **Web UI:** http://localhost:8501
- **Ticker Service API:** http://localhost:8000/docs
- **Data Service API:** http://localhost:8001/docs
- **Calculation Service API:** http://localhost:8002/docs

## Scheduled Price Sync

Run `./run-price-sync.sh` once per week to warm the Yahoo parquet cache. The recommended cron entry (using Central European Time) is:

```
CRON_TZ=Europe/Berlin
0 23 * * 1 /path/to/repo/run-price-sync.sh >> /var/log/price_sync.log 2>&1
5 23 * * 1 /path/to/repo/run-analysis-sync.sh >> /var/log/analysis_sync.log 2>&1
30 6 1 * * /path/to/.venv/bin/python /path/to/repo/services/ticker_validation_service.py >> /var/log/ticker_validation.log 2>&1
```

Or, to guarantee analysis immediately after price loading, chain them in a single entry:

```
CRON_TZ=Europe/Berlin
0 23 * * 1 /path/to/repo/run-price-sync.sh && /path/to/repo/run-analysis-sync.sh >> /var/log/weekly_sync.log 2>&1
```

This keeps both caches fresh every Monday at 23:00 CET/CEST and runs ticker validation on the first day of each month. Adjust the path/log destination as needed and pass `--force-refresh-tickers` to either script when you need a fresh universe download. Sample launchd plists live under `ops/launchd/`.

## Manual Service Startup

If you prefer to start services individually:

### Data Service
```bash
cd services
python data_service.py
```

### Calculation Service  
```bash
cd services
python calculation_service.py
```

### Presentation Service
```bash
streamlit run services/presentation_service.py
```

## Architecture Benefits

### Scalability
- Each service can be scaled independently
- Data service can handle multiple analysis requests
- Calculation service can be replicated for parallel optimization

### Maintainability
- Clear separation of concerns
- Each service has a single responsibility
- Easy to modify or replace individual components

### Testability
- Services can be tested in isolation
- Mock services can be created for testing
- API contracts are well-defined

### Deployability
- Services can be deployed separately
- Rolling deployments possible
- Different services can use different technologies

## Data Flow

1. **User Input** → Presentation Service
2. **Stock Selection** → Data Service (S&P 500 analysis)
3. **Price Data** → Data Service (historical prices)
4. **Optimization** → Calculation Service (portfolio optimization)
5. **Results** → Presentation Service → User

## Service Communication

Services communicate via HTTP REST APIs using:
- **Request/Response Models:** Pydantic models in `shared/models.py`
- **Error Handling:** Standardized error responses
- **Timeouts:** Configured per endpoint based on expected processing time

## Development

### Adding New Endpoints
1. Define request/response models in `shared/models.py`
2. Implement endpoint in appropriate service
3. Update client calls in presentation service
4. Test API functionality

### Extending Services
- **Data Service:** Add new data sources or analysis methods
- **Calculation Service:** Add new optimization algorithms
- **Presentation Service:** Add new UI components or visualizations

## Production Considerations

### Docker Deployment
Each service should be containerized with its own Dockerfile:
- Data service with yfinance dependencies
- Calculation service with PyPortfolioOpt
- Presentation service with Streamlit

### Load Balancing
- Multiple instances of each service behind load balancers
- Health checks for service discovery
- Circuit breakers for fault tolerance

### Database Integration
- Replace Parquet files with proper database
- Add database migration scripts
- Implement connection pooling

### Security
- Add API authentication and authorization
- Implement request rate limiting
- Use HTTPS for all communication

### Monitoring
- Add logging and metrics collection  
- Implement distributed tracing
- Set up alerts for service failures

## File Structure
```
fin-portfolio/
├── services/
│   ├── data_service.py          # Data layer microservice
│   ├── calculation_service.py   # Calculation layer microservice  
│   └── presentation_service.py  # UI layer microservice
├── shared/
│   └── models.py               # Shared data models
├── sp500_data/                 # Data cache directory
├── app.py                      # Original monolithic app
├── requirements.txt            # Original dependencies
├── requirements-microservices.txt  # Microservices dependencies
├── start-microservices.py      # Service launcher script
└── MICROSERVICES.md           # This documentation
```

## Migration from Monolith

The original monolithic application (`app.py`) is preserved and continues to work. The microservices version provides:

- **Better separation of concerns**
- **Independent scaling capabilities**  
- **Easier testing and maintenance**
- **Foundation for cloud deployment**

Both versions can coexist during the transition period.
