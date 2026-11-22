import sqlite3

import chromadb
from sentence_transformers import SentenceTransformer


class VectorSchemaRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection("database_schema")

        self.table_schemas = {}
        self.last_retrieved_tables = []
        self._index_schema()



    def _index_schema(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
        """)
        tables = cursor.fetchall()



        for (table,) in tables:
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]

            schema = f"{table} ({', '.join(column_names)})"
            self.table_schemas[table] = schema

            document = f"Table {table} contains columns: {', '.join(column_names)}"
            embedding = self.encoder.encode(document).tolist()


            self.collection.upsert(
                ids=[table],
                documents=[document],
                embeddings=[embedding],
                metadatas=[{"table": table}]
            )

        conn.close()



    def retrieve_relevant_schema(self, question: str, n_results: int = 4):
        if not self.table_schemas:
            self.last_retrieved_tables = []
            return ""



        embedding = self.encoder.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, len(self.table_schemas))
        )

        tables = []
        for metadata in results["metadatas"][0]:
            table = metadata["table"]
            if table not in tables:
                tables.append(table)


        # Include tables directly connected by foreign keys.
        for table in list(tables):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA foreign_key_list('{table}')")
            foreign_keys = cursor.fetchall()


            for fk in foreign_keys:
                related_table = fk[2]
                if related_table in self.table_schemas and related_table not in tables:
                    tables.append(related_table)
            conn.close()



        self.last_retrieved_tables = tables.copy()

        return "\n".join(self.table_schemas[table] for table in tables)



    def get_allowed_tables(self):
        return set(self.table_schemas.keys())
