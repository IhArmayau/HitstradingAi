import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

async def check():
    url = os.getenv("DATABASE_URL").replace("postgres://", "postgresql+asyncpg://")
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        res = await conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'signals';"))
        columns = res.fetchall()
        print("\n--- Current 'signals' Table Columns ---")
        for col in columns:
            print(f"Column: {col[0]:<20} | Type: {col[1]}")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(check())
