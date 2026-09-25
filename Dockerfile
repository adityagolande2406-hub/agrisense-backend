FROM python:3.10-slim

WORKDIR /code

# Copy requirements & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app & model
COPY app /code/app
COPY model /code/model

# Set production environment variables
ENV APP_ENV=production
ENV USE_MOCK_INFERENCE=false
ENV USE_EMBEDDED_INFERENCE=true
ENV MODEL_PATH=model/grape_disease_model.pth
ENV PORT=7860

EXPOSE 7860

# Run FastAPI server on port 7860 (Hugging Face default)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
