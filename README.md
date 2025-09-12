**Monolith app.py**

# 1) Create a virtual env (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Start the app
streamlit run app.py

**Microservices**

# 1) Create a virtual env (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r microservices-requirements.txt

# 3) Start microservices
python start-microservices.py

# 3) Start UI
streamlit run services/presentation_service.py"
