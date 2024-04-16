# Dockerfile

FROM python:3.8.18

COPY requirements.txt app/requirements.txt

WORKDIR /model

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "chla_app.py"]