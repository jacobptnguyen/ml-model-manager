from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from routes import train, models, predict, health
from sqlalchemy import inspect, text
import os

# Create database tables
Base.metadata.create_all(bind=engine)

# Migration: Add missing columns if they don't exist
def migrate_database():
    """Add missing columns to existing database tables"""
    inspector = inspect(engine)
    
    # Check if models table exists
    if 'models' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('models')]
        
        with engine.begin() as conn:  # begin() handles commit automatically
            if engine.dialect.name == 'sqlite':
                # SQLite doesn't support IF NOT EXISTS in ALTER TABLE
                # But we already checked above, so safe to add
                if 'feature_columns' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN feature_columns TEXT"))
                    print("✓ Added feature_columns column to models table")
                if 'sample_data' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN sample_data TEXT"))
                    print("✓ Added sample_data column to models table")
                if 'label_encoders' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN label_encoders TEXT"))
                    print("✓ Added label_encoders column to models table")
                if 'feature_importance' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN feature_importance TEXT"))
                    print("✓ Added feature_importance column to models table")
                if 'target_encoder' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN target_encoder TEXT"))
                    print("✓ Added target_encoder column to models table")
                if 'best_params' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN best_params TEXT"))
                    print("✓ Added best_params column to models table")
                if 'best_cv_score' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN best_cv_score REAL"))
                    print("✓ Added best_cv_score column to models table")
            else:
                # PostgreSQL and other databases
                if 'feature_columns' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN IF NOT EXISTS feature_columns TEXT"))
                    print("✓ Added feature_columns column to models table")
                if 'sample_data' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN IF NOT EXISTS sample_data TEXT"))
                    print("✓ Added sample_data column to models table")
                if 'label_encoders' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN IF NOT EXISTS label_encoders TEXT"))
                    print("✓ Added label_encoders column to models table")
                if 'feature_importance' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN IF NOT EXISTS feature_importance TEXT"))
                    print("✓ Added feature_importance column to models table")
                if 'target_encoder' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN IF NOT EXISTS target_encoder TEXT"))
                    print("✓ Added target_encoder column to models table")
                if 'best_params' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN IF NOT EXISTS best_params TEXT"))
                    print("✓ Added best_params column to models table")
                if 'best_cv_score' not in columns:
                    conn.execute(text("ALTER TABLE models ADD COLUMN IF NOT EXISTS best_cv_score REAL"))
                    print("✓ Added best_cv_score column to models table")

# Run migrations on startup
migrate_database()

app = FastAPI(title="MLOps Platform API", version="1.0.0")

# CORS middleware
# Allow origins from environment variable or default to localhost for development
# Supports comma-separated list of origins and strips whitespace
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,  # Set to False since we don't use cookies/auth tokens
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Mount routes
app.include_router(health.router, tags=["health"])
app.include_router(train.router, prefix="/api", tags=["train"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(predict.router, prefix="/api", tags=["predict"])


@app.get("/")
async def root():
    return {"message": "MLOps Platform API"}

