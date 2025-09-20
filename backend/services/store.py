import json
import os
from typing import Any
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text


@dataclass
class Store:
    db_url: str

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.db_url.split("///")[-1]), exist_ok=True)
        self.engine = create_async_engine(self.db_url, echo=False, future=True)
        self.Session = sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def create_job(
        self, job_id: str, media_paths: list[str], user_inputs: str, debug: str
    ):
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    media TEXT NOT NULL,
                    user_inputs TEXT,
                    debug TEXT,
                    status TEXT DEFAULT 'queued',
                    result TEXT
                )
            """
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO jobs(id, media, user_inputs, debug, status) VALUES (:id, :media, :user_inputs, :debug, 'queued')"
                ),
                {
                    "id": job_id,
                    "media": json.dumps(media_paths),
                    "user_inputs": user_inputs,
                    "debug": debug,
                },
            )

    async def set_result(self, job_id: str, result: dict[str, Any]):
        async with self.engine.begin() as conn:
            await conn.execute(
                text("UPDATE jobs SET status='done', result=:result WHERE id=:id"),
                {"id": job_id, "result": json.dumps(result)},
            )

    async def get_job(self, job_id: str):
        async with self.engine.begin() as conn:
            res = await conn.execute(
                text(
                    "SELECT id, media, user_inputs, debug, status FROM jobs WHERE id=:id"
                ),
                {"id": job_id},
            )
            row = res.first()
            if not row:
                return None
            return {
                "id": row[0],
                "media": json.loads(row[1]),
                "user_inputs": row[2] or "",
                "debug": row[3] or "none",
                "status": row[4],
            }

    async def get_result(self, job_id: str):
        async with self.engine.begin() as conn:
            res = await conn.execute(
                text("SELECT result FROM jobs WHERE id=:id AND status='done'"),
                {"id": job_id},
            )
            row = res.first()
            if not row or not row[0]:
                return None
            return json.loads(row[0])
