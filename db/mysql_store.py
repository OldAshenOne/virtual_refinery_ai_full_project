import json
import hashlib
import datetime as dt
from typing import Optional, List

import pandas as pd

try:
    import pymysql  # type: ignore
except Exception as e:  # pragma: no cover
    pymysql = None

from config import MYSQL, MySQLConfig


class MySQLStore:
    def __init__(self, conf: Optional[MySQLConfig] = None):
        self.conf = conf or MYSQL
        if pymysql is None:
            raise RuntimeError("未安装 pymysql，请执行: pip install pymysql")

    def _connect_server(self):
        return pymysql.connect(
            host=self.conf.host,
            port=self.conf.port,
            user=self.conf.user,
            password=self.conf.password,
            charset=self.conf.charset,
            autocommit=True,
        )

    def _connect_db(self):
        return pymysql.connect(
            host=self.conf.host,
            port=self.conf.port,
            user=self.conf.user,
            password=self.conf.password,
            database=self.conf.database,
            charset=self.conf.charset,
            autocommit=True,
        )

    def init_schema(self):
        # 创建数据库
        with self._connect_server() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.conf.database}` CHARACTER SET {self.conf.charset}")

        # 创建表
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS import_runs (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        file_name VARCHAR(255) NOT NULL,
                        file_hash VARCHAR(64) NOT NULL,
                        params_json TEXT NOT NULL,
                        status VARCHAR(32) NOT NULL,
                        created_at DATETIME NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS units_results (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        run_id BIGINT NOT NULL,
                        unit_id VARCHAR(64) NOT NULL,
                        unit_name VARCHAR(255) NOT NULL,
                        total_input_kgco2e DECIMAL(24,8) NOT NULL,
                        total_output_qty DECIMAL(24,8) NOT NULL,
                        INDEX idx_run (run_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS stream_results (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        run_id BIGINT NOT NULL,
                        stream_id VARCHAR(255) NOT NULL,
                        factor_stream DECIMAL(24,8) NOT NULL,
                        INDEX idx_run (run_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ai_insights (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        run_id BIGINT NOT NULL,
                        model VARCHAR(128) NOT NULL,
                        content LONGTEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        INDEX idx_run (run_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )

    @staticmethod
    def file_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def create_run(self, file_name: str, file_hash: str, params: dict, status: str = "completed") -> int:
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO import_runs (file_name, file_hash, params_json, status, created_at) VALUES (%s,%s,%s,%s,%s)",
                    (file_name, file_hash, json.dumps(params, ensure_ascii=False), status, dt.datetime.now()),
                )
                return cur.lastrowid

    def save_units_results(self, run_id: int, df_units: pd.DataFrame):
        if df_units is None or df_units.empty:
            return
        rows = [
            (
                int(run_id),
                str(r["unit_id"]),
                str(r["unit_name"]),
                float(r["total_input_kgco2e"] or 0.0),
                float(r["total_output_qty"] or 0.0),
            )
            for _, r in df_units.iterrows()
        ]
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO units_results (run_id, unit_id, unit_name, total_input_kgco2e, total_output_qty) VALUES (%s,%s,%s,%s,%s)",
                    rows,
                )

    def save_stream_results(self, run_id: int, df_stream: Optional[pd.DataFrame]):
        if df_stream is None or df_stream.empty:
            return
        rows = [
            (
                int(run_id),
                str(r["stream_id"]),
                float(r["factor_stream"] or 0.0),
            )
            for _, r in df_stream.iterrows()
        ]
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO stream_results (run_id, stream_id, factor_stream) VALUES (%s,%s,%s)",
                    rows,
                )

    def save_ai(self, run_id: int, model: str, content: str):
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ai_insights (run_id, model, content, created_at) VALUES (%s,%s,%s,%s)",
                    (int(run_id), model, content, dt.datetime.now()),
                )

    def list_runs(self, limit: int = 20) -> List[dict]:
        with self._connect_db() as conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    "SELECT id, file_name, file_hash, status, created_at FROM import_runs ORDER BY id DESC LIMIT %s",
                    (int(limit),),
                )
                return list(cur.fetchall())

    def load_units(self, run_id: int) -> pd.DataFrame:
        with self._connect_db() as conn:
            return pd.read_sql(
                "SELECT unit_id, unit_name, total_input_kgco2e, total_output_qty FROM units_results WHERE run_id=%s",
                conn,
                params=[int(run_id)],
            )

    def load_ai(self, run_id: int) -> str:
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM ai_insights WHERE run_id=%s ORDER BY id DESC LIMIT 1",
                    (int(run_id),),
                )
                row = cur.fetchone()
                return row[0] if row else ""

