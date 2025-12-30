**Monolith app.py**

1) Create a virtual env (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

2) Install dependencies
pip install -r requirements.txt

3) Start the app
streamlit run app.py

**Microservices**

1) Create a virtual env (recommended)  
`python3 -m venv .venv`  
`source .venv/bin/activate`  (Windows: `.venv\Scripts\activate`)

2) Install dependencies  
`pip install -r requirements-microservices.txt`

3) Start microservices  
`python start-microservices.py` (ticker service on 8000, data on 8001, calculation on 8002)

4) Start UI  
`streamlit run services/presentation_service.py`

**Docker (all services)**

1) Build the image  
`docker build -t sp500-microservices .`

2) Run the stack  
`docker run -p 8501:8501 -p 8001:8001 -p 8002:8002 sp500-microservices`

Then open `http://localhost:8501` for the Streamlit UI (FastAPI docs at ports 8000/8001/8002).
