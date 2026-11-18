"""Integration and unit tests for RAG components: Hugging Face Embeddings and ChromaDB vector store."""

import asyncio
import io
import math
import os
import shutil
import tempfile
import unittest

from fastapi import HTTPException, UploadFile
import pymupdf as fitz

from backend.embeddings import (
    get_embedding_model,
    generate_embeddings,
    generate_query_embedding,
    MODEL_NAME,
    DEVICE,
)
from backend.main import app, index_pdf, search_endpoint, search_get_endpoint, SearchRequest
from backend.pdf_parser import extract_and_chunk_pdf
from backend.vector_store import (
    get_chroma_client,
    get_collection,
    index_document,
    search_documents,
    DEFAULT_COLLECTION_NAME,
)


class TestEmbeddings(unittest.TestCase):

    def test_model_loaded_on_cpu(self):
        model = get_embedding_model()
        self.assertIsNotNone(model)
        self.assertEqual(str(model.device), "cpu")

    def test_generate_embeddings_shape_and_norm(self):
        texts = ["Artificial intelligence and deep learning.", "Offline study assistant."]
        embeddings = generate_embeddings(texts)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 384)
        self.assertEqual(len(embeddings[1]), 384)

        # Verify vectors are normalized (L2 norm ≈ 1.0)
        norm = math.sqrt(sum(x * x for x in embeddings[0]))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_generate_query_embedding(self):
        query = "What is supervised learning?"
        q_emb = generate_query_embedding(query)
        self.assertEqual(len(q_emb), 384)
        norm = math.sqrt(sum(x * x for x in q_emb))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_empty_embeddings(self):
        self.assertEqual(generate_embeddings([]), [])


class TestVectorStoreAndIndexing(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.collection_name = "test_study_collection"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_sample_pdf_bytes(self) -> bytes:
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            "Alan Turing proposed the Turing Test in 1950 to evaluate machine intelligence. " * 3,
            fontsize=10
        )
        p2 = doc.new_page()
        p2.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            "Supervised learning algorithms map inputs to desired outputs using training labels. " * 3,
            fontsize=10
        )
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_index_document_preserves_metadata(self):
        pdf_bytes = self._create_sample_pdf_bytes()
        total_pages, chunks = extract_and_chunk_pdf(pdf_bytes, "turing_notes.pdf")

        res = index_document(
            filename="turing_notes.pdf",
            total_pages=total_pages,
            chunks=chunks,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )

        self.assertEqual(res["filename"], "turing_notes.pdf")
        self.assertEqual(res["total_pages"], 2)
        self.assertEqual(res["indexed_chunks"], len(chunks))
        self.assertTrue(res["document_id"].startswith("doc_"))

        # Verify collection contents in ChromaDB
        collection = get_collection(collection_name=self.collection_name, persist_directory=self.test_dir)
        self.assertEqual(collection.count(), len(chunks))

        items = collection.get(include=["metadatas", "documents"])
        for meta in items["metadatas"]:
            self.assertEqual(meta["filename"], "turing_notes.pdf")
            self.assertIn(meta["page_number"], [1, 2])
            self.assertGreaterEqual(meta["chunk_number"], 1)
            self.assertEqual(meta["document_id"], res["document_id"])

    def test_avoid_duplicate_document_entries(self):
        pdf_bytes = self._create_sample_pdf_bytes()
        total_pages, chunks = extract_and_chunk_pdf(pdf_bytes, "turing_notes.pdf")

        # First indexing
        res1 = index_document(
            filename="turing_notes.pdf",
            total_pages=total_pages,
            chunks=chunks,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )

        collection = get_collection(collection_name=self.collection_name, persist_directory=self.test_dir)
        initial_count = collection.count()

        # Re-index the same document
        res2 = index_document(
            filename="turing_notes.pdf",
            total_pages=total_pages,
            chunks=chunks,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )

        # Count must remain unchanged (no duplicate chunks)
        reindexed_count = collection.count()
        self.assertEqual(reindexed_count, initial_count)
        self.assertEqual(res1["document_id"], res2["document_id"])

    def test_search_documents(self):
        pdf_bytes = self._create_sample_pdf_bytes()
        total_pages, chunks = extract_and_chunk_pdf(pdf_bytes, "turing_notes.pdf")

        index_document(
            filename="turing_notes.pdf",
            total_pages=total_pages,
            chunks=chunks,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )

        # Search for Turing
        results = search_documents(
            query="Who invented the Turing Test?",
            top_k=2,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )

        self.assertGreaterEqual(len(results), 1)
        top = results[0]
        self.assertEqual(top["filename"], "turing_notes.pdf")
        self.assertEqual(top["page_number"], 1)
        self.assertIn("Alan Turing", top["text"])
        self.assertIn("distance", top)
        self.assertIn("similarity_score", top)
        self.assertGreater(top["similarity_score"], 0.3)

    def test_chroma_persistence_across_restart(self):
        pdf_bytes = self._create_sample_pdf_bytes()
        total_pages, chunks = extract_and_chunk_pdf(pdf_bytes, "persist_check.pdf")

        index_document(
            filename="persist_check.pdf",
            total_pages=total_pages,
            chunks=chunks,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )

        # Reopen a completely new PersistentClient on the same directory
        reopened_client = get_chroma_client(persist_directory=self.test_dir)
        reopened_collection = reopened_client.get_collection(name=self.collection_name)

        self.assertEqual(reopened_collection.count(), len(chunks))
        matched = reopened_collection.get()
        self.assertEqual(len(matched["ids"]), len(chunks))


class TestRagApiEndpoints(unittest.TestCase):

    def _create_test_pdf_file(self) -> UploadFile:
        doc = fitz.open()
        p = doc.new_page()
        p.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            "Neural networks use backpropagation and gradient descent for weight optimization.",
            fontsize=11
        )
        pdf_bytes = doc.tobytes()
        doc.close()
        return UploadFile(filename="nn_opt.pdf", file=io.BytesIO(pdf_bytes))

    def test_index_and_search_endpoints(self):
        async def run():
            file = self._create_test_pdf_file()
            index_res = await index_pdf(file)

            self.assertEqual(index_res["filename"], "nn_opt.pdf")
            self.assertEqual(index_res["total_pages"], 1)
            self.assertGreaterEqual(index_res["indexed_chunks"], 1)

            # Test POST /search
            search_req = SearchRequest(query="How are neural networks optimized?", top_k=3)
            search_res = await search_endpoint(search_req)

            self.assertEqual(search_res["query"], "How are neural networks optimized?")
            self.assertGreaterEqual(search_res["total_results"], 1)
            top_match = search_res["results"][0]
            self.assertEqual(top_match["filename"], "nn_opt.pdf")
            self.assertEqual(top_match["page_number"], 1)
            self.assertIn("backpropagation", top_match["text"])

            # Test GET /search endpoint
            get_res = await search_get_endpoint(query="gradient descent", top_k=2)
            self.assertGreaterEqual(get_res["total_results"], 1)
            self.assertEqual(get_res["results"][0]["filename"], "nn_opt.pdf")

        asyncio.run(run())

    def test_search_empty_query(self):
        async def run():
            with self.assertRaises(HTTPException) as cm:
                await search_endpoint(SearchRequest(query="   "))
            self.assertEqual(cm.exception.status_code, 400)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
