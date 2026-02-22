# S&P 500 Portfolio Optimizer

This project is a portfolio optimization tool for S&P 500 stocks. It provides a Streamlit-based user interface to analyze stocks and build optimal portfolios based on various financial metrics.

## Architecture

The project currently has two modes of operation:

1.  **Monolith:** A single Streamlit application (`app.py`) that contains all the logic.
2.  **Microservices:** A set of FastAPI services for different concerns (ticker, data, calculation) and a separate Streamlit presentation service for the UI.

A detailed analysis of the current architecture and its technical debt can be found in the `TECHNICAL_DEBT.md` file. It is highly recommended to read this file to understand the current state of the project and the recommended improvements.

## How to Run

### Monolith

To run the monolithic version of the application, use the following command:

```bash
streamlit run app.py
```

### Microservices

To run the microservices version, you first need to start the services:

```bash
./start-microservices.sh
```

Then, in a separate terminal, start the presentation layer:

```bash
streamlit run services/presentation_service.py
```

### Docker

The project can also be run using Docker. The Docker setup is configured for the microservices architecture.

To build the Docker image:

```bash
docker build -t fin-portfolio .
```

To run the application in a Docker container:

```bash
docker run -p 8501:8501 -p 8000:8000 -p 8001:8001 -p 8002:8002 fin-portfolio
```
Then open your browser to `http://localhost:8501`.
