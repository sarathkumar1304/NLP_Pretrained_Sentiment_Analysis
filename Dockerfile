
FROM python:3.12-slim

WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install ZenML and its integrations
RUN pip install zenml
RUN zenml integration install mlflow -y

COPY . /app/

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501


CMD ["streamlit", "run", "frontend/main.py"]
