import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer

class VectorSchemaRetriever:
    def __init__(self,db_path:str):
        self.db_path=db_path

        self.encoder=SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client=chromadb.Client()
        self.collection=self.chroma_client.get_or_create_collection(name="database_schema")
        self._index_schema()
        
    # PILLAR 1: Schema Context Optimization (Simple Version)
    def _index_schema(self):
        """Introspects SQLite and indexes table/column metadata into Chroma."""
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        documents = []
        metadatas = []
        ids = []

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Create descriptive text chunks for semantic matching
            doc_text = f"Table '{table}' contains columns: {', '.join(columns)}."
            
            documents.append(doc_text)
            metadatas.append({"table_name": table, "columns": str(columns)})
            ids.append(table)

        conn.close()

        # Embed and store in vector database
        embeddings = self.encoder.encode(documents).tolist()
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve_relevant_schema(self,user_query:str,n_results:int=2)->str:
        """Finds the most sematically relevant tables/columns for a given user query."""
        query_embedding=self.encoder.encode([user_query]).tolist()

        results=self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        retrieved_schema={}
        for metadata in results["metadatas"][0]:
            table=metadata["table_name"]
            retrieved_schema[table]=eval(metadata["columns"])

        return str(retrieved_schema)