import os
import json
import datetime
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress
from rich import print as rprint

# --- Configuration ---
COLLECTION_NAME = "surveillance_logs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_PATH = "./qdrant_db"  # Local storage mode

console = Console()

class SurveillanceRAG:
    def __init__(self):
        """Initialize Qdrant client and Embedding model."""
        try:
            # Initialize Qdrant in local storage mode
            self.client = QdrantClient(path=QDRANT_PATH)
            
            # Initialize SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME)
            
            # Create collection if it doesn't exist
            self._ensure_collection()
            
        except Exception as e:
            console.print(f"[bold red]Initialization Error:[/bold red] {e}")
            raise

    def _ensure_collection(self):
        """Creates the Qdrant collection if it's missing."""
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            rprint(f"[dim]Created collection: {COLLECTION_NAME}[/dim]")

    def ingest_log(self, vlm_text: str, timestamp: str, camera_id: str, frame_path: str = ""):
        """
        Embeds the vlm_text and upserts it into Qdrant.
        """
        try:
            embedding = self.model.encode(vlm_text).tolist()
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "vlm_output": vlm_text,
                    "timestamp": timestamp,
                    "camera_id": camera_id,
                    "frame_path": frame_path
                }
            )
            
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point]
            )
        except Exception as e:
            console.print(f"[bold red]Ingestion Error:[/bold red] {e}")

    def search_lost_item(self, query_string: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a similarity search on the query string using query_points.
        """
        try:
            query_vector = self.model.encode(query_string).tolist()
            
            # query_points is the recommended method in newer qdrant-client versions
            search_result = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=limit
            ).points
            
            results = []
            for hit in search_result:
                results.append({
                    "score": round(hit.score, 4),
                    "timestamp": hit.payload.get("timestamp"),
                    "camera_id": hit.payload.get("camera_id"),
                    "vlm_output": hit.payload.get("vlm_output"),
                    "frame_path": hit.payload.get("frame_path")
                })
            return results
        except Exception as e:
            console.print(f"[bold red]Search Error:[/bold red] {e}")
            return []

def generate_mock_data(rag: SurveillanceRAG):
    """Populates the database with 15 realistic surveillance logs."""
    mock_logs = [
        ("NORMAL | Person walking through the main lobby.", "2026-05-07T10:00:01Z", "CAM_01", "data/cache/frames/stream_0/frame_000001.jpg"),
        ("TAKING | Left hand grabbing a black leather wallet from the counter.", "2026-05-07T10:05:22Z", "CAM_02", "data/cache/frames/stream_0/frame_000025.jpg"),
        ("EYEING | Staring intently at the luxury watch display case.", "2026-05-07T10:08:45Z", "CAM_01", "data/cache/frames/stream_0/frame_000050.jpg"),
        ("NORMAL | Customer browsing magazines in the aisle.", "2026-05-07T10:12:10Z", "CAM_03", "data/cache/frames/stream_0/frame_000075.jpg"),
        ("TAKING | Right hand picking up a red backpack from the floor.", "2026-05-07T10:15:30Z", "CAM_02", "data/cache/frames/stream_0/frame_000100.jpg"),
        ("NORMAL | Staff member cleaning the glass entrance door.", "2026-05-07T10:20:00Z", "CAM_01", "data/cache/frames/stream_0/frame_000125.jpg"),
        ("EYEING | Looking closely at the security camera in the corner.", "2026-05-07T10:25:15Z", "CAM_03", "data/cache/frames/stream_0/frame_000150.jpg"),
        ("TAKING | Hand sliding a silver iPhone into a jacket pocket.", "2026-05-07T10:30:45Z", "CAM_02", "data/cache/frames/stream_0/frame_000175.jpg"),
        ("NORMAL | Two people talking near the elevator bank.", "2026-05-07T10:35:12Z", "CAM_01", "data/cache/frames/stream_0/frame_000200.jpg"),
        ("TAKING | Grabbing a blue umbrella from the rack.", "2026-05-07T10:40:05Z", "CAM_03", "data/cache/frames/stream_0/frame_000225.jpg"),
        ("EYEING | Hovering over the jewelry section for several minutes.", "2026-05-07T10:45:50Z", "CAM_02", "data/cache/frames/stream_0/frame_000250.jpg"),
        ("NORMAL | Person sitting on the bench checking their phone.", "2026-05-07T10:50:33Z", "CAM_01", "data/cache/frames/stream_0/frame_000275.jpg"),
        ("TAKING | Reaching for a gold necklace on the stand.", "2026-05-07T10:55:18Z", "CAM_02", "data/cache/frames/stream_0/frame_000300.jpg"),
        ("NORMAL | Delivery driver carrying a large cardboard box.", "2026-05-07T11:00:42Z", "CAM_03", "data/cache/frames/stream_0/frame_000325.jpg"),
        ("TAKING | Picking up a dropped set of car keys.", "2026-05-07T11:05:09Z", "CAM_01", "data/cache/frames/stream_0/frame_000350.jpg"),
    ]
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Ingesting mock logs...", total=len(mock_logs))
        for log, ts, cam, path in mock_logs:
            rag.ingest_log(log, ts, cam, path)
            progress.update(task, advance=1)

def display_results(query: str, results: List[Dict[str, Any]]):
    """Renders the search results in a rich table."""
    if not results:
        console.print(f"\n[yellow]No relevant events found for:[/yellow] '{query}'\n")
        return

    table = Table(title=f"Search Results for: '{query}'", header_style="bold magenta", border_style="blue")
    table.add_column("Score", justify="center", style="cyan")
    table.add_column("Timestamp", style="green")
    table.add_column("Camera", style="yellow")
    table.add_column("VLM Log", style="white")
    table.add_column("Frame Path (View)", style="dim")

    for res in results:
        table.add_row(
            str(res["score"]),
            res["timestamp"],
            res["camera_id"],
            res["vlm_output"],
            res["frame_path"]
        )
    
    console.print(table)
    rprint("\n[bold blue]Tip:[/bold blue] Since you are on SSH, you can copy the 'Frame Path' to your local machine via scp or view it if using VS Code.")

def run_tui():
    """Main Terminal UI loop."""
    # Clear screen
    console.clear()
    
    # ASCII Header
    header_text = """
[bold cyan]
  ██████  ██    ██ ██████  ██    ██ ███████ ██ ██      ██       █████  ███    ██  ██████ ███████ 
 ██       ██    ██ ██   ██ ██    ██ ██      ██ ██      ██      ██   ██ ████   ██ ██      ██      
 ██   ███ ██    ██ ██████  ██    ██ █████   ██ ██      ██      ███████ ██ ██  ██ ██      █████   
 ██    ██ ██    ██ ██   ██  ██  ██  ██      ██ ██      ██      ██   ██ ██  ██ ██ ██      ██      
  ██████   ██████  ██   ██   ████   ███████ ██ ███████ ███████ ██   ██ ██   ████  ██████ ███████ 
                                                                                                 
                 ██████   █████   ██████      ██████  ███████ ████████ ██████  ██ ███████ ██    ██  █████  ██      
                 ██   ██ ██   ██ ██           ██   ██ ██         ██    ██   ██ ██ ██      ██    ██ ██   ██ ██      
                 ██████  ███████ ██   ███     ██████  █████      ██    ██████  ██ █████   ██    ██ ███████ ██      
                 ██   ██ ██   ██ ██    ██     ██   ██ ██         ██    ██   ██ ██ ██       ██  ██  ██   ██ ██      
                 ██   ██ ██   ██  ██████      ██   ██ ███████    ██    ██   ██ ██ ███████   ████   ██   ██ ███████ 
[/bold cyan]
    """
    console.print(Panel(header_text, subtitle="Vector-Based Surveillance Intelligence", border_style="cyan"))

    # Initialize RAG
    rag = SurveillanceRAG()
    
    # Optional: Check if we need to ingest mock data
    if rag.client.get_collection(COLLECTION_NAME).points_count == 0:
        generate_mock_data(rag)

    while True:
        try:
            query = Prompt.ask("\n[bold green]>[/bold green] Describe the lost item or event to search (or type 'exit')")
            
            if query.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Shutting down retrieval system...[/yellow]")
                break
            
            if not query.strip():
                continue

            with console.status("[bold blue]Searching vector database..."):
                results = rag.search_lost_item(query)
            
            display_results(query, results)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {e}")

if __name__ == "__main__":
    run_tui()
