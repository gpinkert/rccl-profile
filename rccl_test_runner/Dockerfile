FROM python:3.12-slim

# set a working directory
WORKDIR /app

# install OS-level deps (if you need any; e.g. git, build-essentials)
# RUN apt-get update && apt-get install -y build-essential

# copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your app code
COPY . .

# expose Streamlit’s default port
EXPOSE 8501

# launch Streamlit in headless mode
CMD ["streamlit", "run", "dashboard_json.py", \
     "--server.headless", "true", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0"]