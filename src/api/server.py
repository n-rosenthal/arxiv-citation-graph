# ============================================================================
# API FASTAPI — exposição do grafo de citações e papers
# ============================================================================
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.config import DB_URL
from src.db import get_session_factory, Paper, Citation

app = FastAPI(title="arXiv Citation Graph API")

_SessionLocal = get_session_factory(DB_URL)


class PaperResponse(BaseModel):
    id: str
    title: str
    categories: str
    pdf_downloaded: bool
    references_extracted: bool
    num_references: int
    num_citations_in_graph: int


class GraphStatsResponse(BaseModel):
    num_nodes: int
    num_edges: int
    num_classes: int
    graph_density: float


def get_db():
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/api/stats", response_model=GraphStatsResponse)
async def get_graph_stats(db: Session = Depends(get_db)):
    papers_count = db.query(Paper).count()
    citations_count = db.query(Citation).count()
    if papers_count == 0:
        return GraphStatsResponse(num_nodes=0, num_edges=0, num_classes=0, graph_density=0.0)

    categories = db.query(Paper.categories).all()
    unique_cats = set()
    for (cat_str,) in categories:
        if cat_str:
            unique_cats.add(cat_str.split(',')[0])
    num_classes = len(unique_cats)
    density = citations_count / (papers_count ** 2) if papers_count > 0 else 0
    return GraphStatsResponse(num_nodes=papers_count, num_edges=citations_count,
                               num_classes=num_classes, graph_density=density)


@app.get("/api/papers/{paper_id}", response_model=PaperResponse)
async def get_paper(paper_id: str, db: Session = Depends(get_db)):
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperResponse(**{c.name: getattr(paper, c.name) for c in Paper.__table__.columns})


@app.get("/api/citations/{paper_id}")
async def get_citations(paper_id: str, direction: str = "outgoing", db: Session = Depends(get_db)):
    if direction == "outgoing":
        citations = db.query(Citation).filter(Citation.source_id == paper_id).all()
        targets = [c.target_id for c in citations]
        return {"paper_id": paper_id, "direction": "outgoing", "citations": targets, "count": len(targets)}
    elif direction == "incoming":
        citations = db.query(Citation).filter(Citation.target_id == paper_id).all()
        sources = [c.source_id for c in citations]
        return {"paper_id": paper_id, "direction": "incoming", "citations": sources, "count": len(sources)}
    else:
        raise HTTPException(status_code=400, detail="direction must be 'outgoing' or 'incoming'")


@app.post("/api/search")
async def search_papers(query: str, max_results: int = 100):
    # Placeholder - pode ser integrado com src.collection.collector.search_new_papers
    return {"status": "search not implemented in API yet", "query": query, "max_results": max_results}


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)