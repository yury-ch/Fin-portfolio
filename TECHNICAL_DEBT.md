# Technical Debt Report

This document outlines the key areas of technical debt in the S&P 500 Portfolio Optimizer project. The items are prioritized from P1 (most critical) to P3 (least critical).

## P1: Critical Issues

### 1. Code Duplication between Monolith and Microservice

**Observation:**
The core financial analysis logic is duplicated almost verbatim in two key places:
- `app.py`: The monolithic Streamlit application.
- `services/data_service.py`: The data microservice.

The `compute_sp500_analysis` and `standardize_analysis_columns` functions are present in both files with nearly identical implementations.

**Why it's a problem:**
- **Increased Maintenance Overhead:** Any bug fix or feature enhancement in the analysis logic must be implemented in two separate places, doubling the development effort.
- **Risk of Inconsistency:** It is highly likely that the two implementations will diverge over time, leading to a situation where the monolith and the microservices produce different results from the same inputs.
- **Violates DRY Principle:** Duplicating code is a fundamental violation of the "Don't Repeat Yourself" principle, which leads to lower code quality and higher complexity.

**Recommendation:**
Refactor the analysis logic into a shared module. Create a new file, for example `shared/analysis_engine.py`, and move the `compute_sp500_analysis` and related functions into it. Both `app.py` and `services/data_service.py` should then import and use this shared module. This will create a single source of truth for the analysis logic.

### 2. Monolithic `app.py` Structure

**Observation:**
The `app.py` file is a single, large file (over 700 lines) that contains the entire Streamlit application. It mixes several concerns:
- UI rendering (Streamlit widgets)
- Data fetching and processing (`yfinance`)
- Complex business logic (portfolio optimization)
- Data persistence (Parquet file I/O)
- Asynchronous task management

**Why it's a problem:**
- **Hard to Maintain:** Large files are difficult to read, understand, and modify. Finding the relevant code for a specific feature or bug is a time-consuming task.
- **Difficult to Test:** The tight coupling between UI, data, and logic makes it nearly impossible to write unit tests for individual components.
- **Low Reusability:** None of the logic within `app.py` can be easily reused in other parts of the system (as evidenced by the code duplication issue).

**Recommendation:**
Break down `app.py` into smaller, more focused modules. For example:
- **`ui/`:** A directory for Streamlit UI components.
- **`data/`:** Modules for data fetching and persistence.
- **`core/`:** The core business logic for portfolio optimization.
The main `app.py` file would then be responsible for orchestrating these modules, but would contain very little logic itself.

## P2: Major Issues

### 1. Inconsistent and Competing Architectures

**Observation:**
The project currently supports two distinct architectural patterns: a monolith (`app.py`) and a set of microservices (`services/*`). These two architectures are not well-integrated. The monolith does not use the microservices, which has led to the critical code duplication issue.

**Why it's a problem:**
- **Developer Confusion:** It is not clear which architecture is the "preferred" one. This can lead to confusion for new developers joining the project.
- **Wasted Effort:** Maintaining two separate architectures is a waste of resources. The team is effectively building the same application twice.

**Recommendation:**
The team should make a strategic decision to commit to one architecture.
- **If microservices are the future:** The `app.py` monolith should be deprecated and eventually removed. The `services/presentation_service.py` should be enhanced to become the primary user interface.
- **If the monolith is simpler to maintain:** The microservices should be deprecated, and the focus should be on refactoring `app.py` into a more modular structure.

### 2. Brittle Service Dependencies and Startup Logic

**Observation:**
The microservices have several issues related to dependencies and startup:
- **Implicit Dependencies:** The `presentation_service` depends on the other services, but there is no mechanism to ensure they are running and healthy.
- **Manual Cache Seeding:** The `data_service` requires the `run-analysis-sync.sh` script to be run manually to populate the cache. The service will fail if the cache is stale or missing.
- **Naive Docker Entrypoint:** The `docker-entrypoint.sh` script starts all services but does not perform health checks or manage dependencies, which could lead to a partially running and broken application state.

**Why it's a problem:**
- **Low Reliability:** The system is not resilient to failures. A single service failing can bring down the entire application.
- **Poor Automation:** The reliance on manual scripts makes the system difficult to deploy and manage in an automated way.

**Recommendation:**
- Implement a service discovery mechanism or a simple health check polling system in the `presentation_service` to ensure its dependencies are available before it starts serving requests.
- Automate the analysis cache generation. This could be a periodic background task within the `data_service` itself, or a separate, containerized cron job.
- Use a tool like `docker-compose` to manage the microservices stack. `docker-compose` can define service dependencies, health checks, and restart policies, which would make the system much more robust.

## P3: Minor Issues

### 1. Inefficient Dockerfile

**Observation:**
The `Dockerfile` copies the entire project context (`COPY . .`) and installs build tools (`build-essential`) in the final image.

**Why it's a problem:**
- **Large Image Size:** Including unnecessary files and build tools increases the size of the Docker image, which leads to longer build times and higher storage costs.
- **Broken Layer Caching:** Copying the entire directory in one go means that any change to any file will invalidate the Docker layer cache, forcing a full rebuild.

**Recommendation:**
- Use a `.dockerignore` file to exclude unnecessary files and directories (e.g., `.git`, `__pycache__`, `.venv`).
- Use more granular `COPY` instructions to copy only the necessary parts of the application.
- Use a multi-stage build to compile any dependencies that require a build toolchain, and then copy only the compiled artifacts to the final, smaller runtime image.

### 2. Lack of Centralized Configuration

**Observation:**
The application uses hardcoded constants for configuration values like file paths, API ports, and analysis parameters.

**Why it's a problem:**
- **Inflexible:** Changing any of these values requires modifying the code and redeploying the application.
- **Error-Prone:** It's easy to make a mistake when changing a hardcoded value, and there is no single place to see all the configurable parameters.

**Recommendation:**
Externalize the configuration. Use a library like `pydantic`'s `BaseSettings` to load configuration from environment variables or a `.env` file. This would make the application much more flexible and easier to configure for different environments (development, testing, production).

### 3. Disorganized Data and Caching Logic

**Observation:**
The project contains a mix of "legacy" and "period-specific" data files. The logic for reading and writing to these files is spread across different modules.

**Why it's a problem:**
- **Complexity:** The caching logic is difficult to understand and debug.
- **Data Inconsistency:** It's not clear which data source is authoritative.

**Recommendation:**
Consolidate all data persistence and caching logic into a single, well-defined module. This module should provide a simple and consistent API for storing and retrieving data, hiding the underlying complexity of the file structure. A data migration script could be created to move all legacy data into the new, more organized structure.

### 4. Unpinned Dependencies

**Observation:**
The `requirements-microservices.txt` file does not pin the versions of most of its dependencies (e.g., `streamlit`, `fastapi`, `pandas`).

**Why it's a problem:**
- **Non-Reproducible Builds:** If you build the project today and then again in six months, you will likely get different versions of the dependencies. This could introduce breaking changes and make it difficult to reproduce bugs.

**Recommendation:**
Pin all dependencies to specific versions. Use a tool like `pip-tools` (with `pip-compile`) to generate a fully pinned `requirements.txt` file from a `requirements.in` file that specifies the top-level dependencies. This ensures that every build uses the exact same versions of all libraries.
