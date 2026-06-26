#   Métodos para criar e fechar sessão com o banco de dados
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

#   Base declarativa que definimos em models
from src.db.models import Base

DB_URL: str = r"sqlite:///data/arxiv_continuous.db?check_same_thread=False";
"""URL de conexão ao banco de dados (SQLite),
    considerando WAL mode para concorrência."""


def get_engine(db_url: str = DB_URL):
    """Cria engine, ativa WAL mode e garante que as tabelas existam."""
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False, "timeout": 30}
    )
    _enable_wal_mode(engine)
    Base.metadata.create_all(engine)
    return engine


def _enable_wal_mode(engine):
    """Ativa Write-Ahead Logging para melhor concorrência no SQLite."""
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.commit()


def get_session_factory(db_url: str = "sqlite:///data/arxiv_continuous.db?check_same_thread=False"):
    """Retorna um SessionLocal vinculado ao engine configurado."""
    engine = get_engine(db_url)
    return sessionmaker(bind=engine)