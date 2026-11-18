import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Vaulto AI Service"
    ENVIRONMENT: str = "development"
    PORT: int = 8000

    # Supabase Credentials (Shared with Node backend)
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    # AI & LLM Service Keys
    GROQ_API_KEY: str = ""
    
    # Internal Node.js Backend URL (For sending validated tool calls back)
    NODE_BACKEND_URL: str = "http://localhost:5000/api"

    class Config:
        env_file = ".env"
        extra = "ignore"

# Instantiate singleton settings object
settings = Settings()