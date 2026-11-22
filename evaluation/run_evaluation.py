import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from core.vector_retriever import VectorSchemaRetriever
from core.workflow import sql_workflow
from evaluation.evaluator import run_evaluation

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("CHATSQL_DB_PATH", str(BASE_DIR / "data/benchmark_ecommerce.db"))
DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


async def main():
    model = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
    retriever = VectorSchemaRetriever(DB_PATH)
    graph = sql_workflow(model, retriever)
    await run_evaluation(DATASET_PATH, DB_PATH, graph, retriever)


if __name__ == "__main__":
    asyncio.run(main())
