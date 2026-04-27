FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONPATH=/opt/program

WORKDIR /opt/program

RUN pip install --no-cache-dir \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4

COPY src /opt/program/src