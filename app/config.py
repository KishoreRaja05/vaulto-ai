from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Vaulto AI Service"
    ENVIRONMENT: str = "development"
    PORT: int = 8000

    # AI & LLM Service Keys
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # Internal Node.js Backend URL (tool calls go here; this service never touches the DB)
    NODE_BACKEND_URL: str = "http://localhost:5000/api"

    # Comma-separated origins allowed to call this service from a browser
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # Internal-only endpoints (e.g. /parse) require this key from the Node backend
    INTERNAL_API_KEY: str = ""
    DAILY_LLM_CALL_LIMIT: int = 200

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()