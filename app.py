import os
import uvicorn
from app.main import app

# Ensure embedded inference is enabled for Hugging Face Space
os.environ["APP_ENV"] = "production"
os.environ["USE_MOCK_INFERENCE"] = "false"
os.environ["USE_EMBEDDED_INFERENCE"] = "true"
os.environ["MODEL_PATH"] = "model/grape_disease_model.pth"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
