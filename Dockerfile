FROM python:3.11-slim
WORKDIR /app
RUN pip install pcse fastapi uvicorn pandas pyyaml
COPY oasis_api.py .
EXPOSE 8000
CMD ["uvicorn", "oasis_api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
