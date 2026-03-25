import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

async def run_migration():
    # Retrieve the URL
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("❌ Error: DATABASE_URL not found. Run 'export DATABASE_URL=...' or check your .env file.")
        return

    # Aiven/Heroku often provide 'postgres://', but SQLAlchemy async needs 'postgresql+asyncpg://'
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif not database_url.startswith("postgresql+asyncpg://"):
        # Ensure the driver is explicitly set if it was missing entirely
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    try:
        engine = create_async_engine(database_url)
        
        async with engine.begin() as conn:
            print("Connecting to Aiven PostgreSQL...")
            
            # Migration Queries
            await conn.execute(text("ALTER TABLE signals ADD COLUMN IF NOT EXISTS open_interest FLOAT DEFAULT 0.0;"))
            await conn.execute(text("ALTER TABLE signals ADD COLUMN IF NOT EXISTS funding FLOAT DEFAULT 0.0;"))
            
            print("✅ Migration successful: Added open_interest and funding columns.")

        await engine.dispose()
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_migration())
