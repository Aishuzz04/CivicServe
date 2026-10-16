import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = int(os.environ.get("DB_PORT", 5432))
    DB_NAME = os.environ.get("DB_NAME", "smart_public_infra")
    DB_USER = os.environ.get("DB_USER", "postgres")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # Computer Vision & Image Validation Settings
    IMAGE_VALIDATION_ENABLED = os.environ.get("IMAGE_VALIDATION_ENABLED", "True").lower() in ("true", "1", "yes")
    # AI Image Policy: 'flag' (allow submission but flag for admin review) or 'reject' (block submission)
    AI_IMAGE_POLICY = os.environ.get("AI_IMAGE_POLICY", "flag").lower()
    AI_DETECTION_THRESHOLD = float(os.environ.get("AI_DETECTION_THRESHOLD", 0.70))
    RELEVANCE_ACCEPT_THRESHOLD = float(os.environ.get("RELEVANCE_ACCEPT_THRESHOLD", 48.0))
    RELEVANCE_UNCERTAIN_THRESHOLD = float(os.environ.get("RELEVANCE_UNCERTAIN_THRESHOLD", 28.0))
    MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB