# -------- Base image --------
FROM python:3.11-slim

# -------- System deps --------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------- Working dir --------
WORKDIR /app

# -------- Copy dependencies --------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# -------- Copy project code --------
COPY . .

# -------- Streamlit settings --------
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

# -------- Expose port --------
EXPOSE 8501

# -------- Run app --------
CMD ["streamlit", "run", "main.py"]
