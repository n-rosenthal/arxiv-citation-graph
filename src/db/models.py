from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Paper(Base):
    """Tabela de artigos"""
    __tablename__ = 'papers'
    
    id = Column(String(20), primary_key=True)  # arXiv ID
    title = Column(String(500))
    authors = Column(Text)
    abstract = Column(Text)
    categories = Column(String(200))
    published_date = Column(DateTime)
    updated_date = Column(DateTime)
    pdf_path = Column(String(200))
    pdf_downloaded = Column(Boolean, default=False)
    pdf_size = Column(Integer)  # bytes
    text_extracted = Column(Boolean, default=False)
    references_extracted = Column(Boolean, default=False)
    num_references = Column(Integer, default=0)
    num_citations_in_graph = Column(Integer, default=0)  # Citações que estão no grafo
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Citation(Base):
    """Tabela de citações (arestas)"""
    __tablename__ = 'citations'
    
    id = Column(Integer, primary_key=True)
    source_id = Column(String(20), ForeignKey('papers.id'))
    target_id = Column(String(20), ForeignKey('papers.id'))
    found_in_text = Column(Boolean, default=True)
    confidence = Column(Float, default=1.0)  # Confiança da extração
    created_at = Column(DateTime, default=datetime.utcnow)

class ProcessingQueue(Base):
    """Fila de processamento"""
    __tablename__ = 'processing_queue'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(String(20), ForeignKey('papers.id'))
    task_type = Column(String(50))  # 'download', 'extract_text', 'extract_refs', 'train'
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    priority = Column(Integer, default=5)  # 1-10, menor = maior prioridade
    retry_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelCheckpoint(Base):
    """Checkpoints de modelos treinados"""
    __tablename__ = 'model_checkpoints'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50))  # 'gcn', 'graphsage', 'gat'
    version = Column(Integer)
    accuracy = Column(Float)
    loss = Column(Float)
    num_nodes = Column(Integer)
    num_edges = Column(Integer)
    num_classes = Column(Integer)
    training_date = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String(200))
    metrics = Column(JSON)  # Store additional metrics

class TrainingJob(Base):
    """Jobs de treinamento"""
    __tablename__ = 'training_jobs'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50))
    status = Column(String(20), default='pending')  # pending, running, completed, failed
    params = Column(JSON)  # Hyperparameters
    metrics = Column(JSON, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    checkpoint_id = Column(Integer, ForeignKey('model_checkpoints.id'), nullable=True)

class DashboardMetrics(Base):
    """Métricas agregadas para dashboard"""
    __tablename__ = 'dashboard_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(100))
    metric_value = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    extra_data = Column(JSON, nullable=True)