"""Unit and integration tests for AI Study Summarizer feature in Offline Study AI.

Verifies:
1. Document isolation (summary contains only content from the selected PDF).
2. Empty/non-extractable document handling.
3. Long PDF chunk batching and multi-pass combination without context overflow.
4. Chapter-wise summary detection and extraction.
5. Ollama error handling (connection errors).
6. Source page references tracking.
7. FastAPI /summarize endpoint.
"""

import asyncio
import shutil
import tempfile
import unittest

from fastapi import HTTPException
import pymupdf as fitz

from backend.main import summarize_endpoint, SummarizeRequest, app
from backend.ollama_client import OllamaConnectionError
from backend.pdf_parser import extract_and_chunk_pdf
from backend.summarizer import (
    summarize_document,
    extract_chapters_from_chunks,
    parse_summary_sections,
)
from backend.vector_store import index_document


def create_test_pdf_with_text(pages_content: list[str]) -> bytes:
    """Helper to generate an in-memory PDF with specified text per page."""
    doc = fitz.open()
    for text in pages_content:
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(50, 50, 500, 700),
            text,
            fontsize=11,
        )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestSummarizerSectionsParsing(unittest.TestCase):

    def test_parse_summary_sections_with_standard_headers(self):
        markdown_text = (
            "### Document Overview\n"
            "This document covers introductory microeconomics principles.\n\n"
            "### Main Topics\n"
            "- Supply and Demand Dynamics\n"
            "- Market Equilibrium and Price Elasticity\n\n"
            "### Important Points\n"
            "- Price elasticity measures responsiveness to price shifts.\n"
            "- Equilibrium occurs where supply intersects demand.\n\n"
            "### Key Takeaways\n"
            "- Market forces allocate scarce resources efficiently.\n"
        )
        parsed = parse_summary_sections(markdown_text)
        self.assertIn("microeconomics", parsed["overview"].lower())
        self.assertEqual(len(parsed["main_topics"]), 2)
        self.assertEqual(len(parsed["important_points"]), 2)
        self.assertEqual(len(parsed["key_takeaways"]), 1)

    def test_extract_chapters_from_chunks(self):
        chunks = [
            {"page_number": 1, "text": "Chapter 1: Foundations of Microeconomics\nBasic definitions of scarcity."},
            {"page_number": 2, "text": "Continuation of Chapter 1 concepts and models."},
            {"page_number": 3, "text": "Chapter 2: Consumer Behavior\nUtility functions and indifference curves."},
            {"page_number": 4, "text": "Budget constraints and optimal choice."},
        ]
        chapters = extract_chapters_from_chunks(chunks)
        self.assertEqual(len(chapters), 2)
        self.assertIn("Chapter 1", chapters[0]["chapter_title"])
        self.assertEqual(chapters[0]["page_range"], "Pages 1–2")
        self.assertIn("Chapter 2", chapters[1]["chapter_title"])
        self.assertEqual(chapters[1]["page_range"], "Pages 3–4")


class TestDocumentSummarizationWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_summarizer_col"

        # Document 1: Operating Systems (2 pages)
        pdf1 = create_test_pdf_with_text([
            "Chapter 1: Operating System Fundamentals\n"
            "A process is an instance of a running computer program with dedicated virtual memory. "
            "Threads share address space within a process to allow lightweight concurrency.",

            "Chapter 2: Synchronization Primitives\n"
            "Semaphores and mutexes prevent race conditions when multiple threads access shared resources."
        ])
        p1, chunks1 = extract_and_chunk_pdf(pdf1, "os_lecture.pdf")
        res1 = index_document(
            filename="os_lecture.pdf",
            total_pages=p1,
            chunks=chunks1,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.doc1_id = res1["document_id"]

        # Document 2: History (2 pages)
        pdf2 = create_test_pdf_with_text([
            "The French Revolution began in 1789 with the Storming of the Bastille in Paris.",
            "The Rise of Napoleon Bonaparte transformed Europe through the Napoleonic Code and military campaigns."
        ])
        p2, chunks2 = extract_and_chunk_pdf(pdf2, "french_history.pdf")
        res2 = index_document(
            filename="french_history.pdf",
            total_pages=p2,
            chunks=chunks2,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.doc2_id = res2["document_id"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_document_isolation_summary(self):
        # Summarizing doc1 must only contain OS concepts and cite doc1
        summary = summarize_document(
            document_id=self.doc1_id,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertEqual(summary["document_id"], self.doc1_id)
        self.assertEqual(summary["filename"], "os_lecture.pdf")
        self.assertEqual(summary["total_pages"], 2)
        self.assertEqual(summary["source_pages"], [1, 2])

        text = summary["full_summary"].lower()
        self.assertTrue("process" in text or "thread" in text or "semaphore" in text)
        # Must NOT leak doc2 topics
        self.assertNotIn("bastille", text)
        self.assertNotIn("napoleon", text)

        # Overview, Main Topics, Important Points, Key Takeaways must be present
        self.assertTrue(len(summary["overview"]) > 0)
        self.assertGreaterEqual(len(summary["main_topics"]), 1)
        self.assertGreaterEqual(len(summary["important_points"]), 1)

    def test_chapter_wise_summary_generation(self):
        # doc1 has Chapter 1 and Chapter 2 headers
        summary = summarize_document(
            document_id=self.doc1_id,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            include_chapters=True,
        )
        self.assertGreaterEqual(len(summary["chapters"]), 2)
        chap_titles = [c["chapter_title"] for c in summary["chapters"]]
        self.assertTrue(any("Chapter 1" in t for t in chap_titles))
        self.assertTrue(any("Chapter 2" in t for t in chap_titles))

    def test_non_existent_document_raises_404(self):
        with self.assertRaises(HTTPException) as cm:
            summarize_document(
                document_id="non_existent_doc_id_9999",
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
            )
        self.assertEqual(cm.exception.status_code, 404)

    def test_empty_document_id_raises_400(self):
        with self.assertRaises(HTTPException) as cm:
            summarize_document(
                document_id="   ",
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
            )
        self.assertEqual(cm.exception.status_code, 400)

    def test_empty_extractable_text_document_raises_400(self):
        # Index an artificial document with empty chunk text
        dummy_dir = tempfile.mkdtemp()
        dummy_col = "test_empty_summarize_col"
        try:
            empty_chunks = [{"chunk_number": 1, "page_number": 1, "text": "   "}]
            res = index_document(
                filename="scanned_blank_notes.pdf",
                total_pages=1,
                chunks=empty_chunks,
                collection_name=dummy_col,
                persist_directory=dummy_dir,
            )
            with self.assertRaises(HTTPException) as cm:
                summarize_document(
                    document_id=res["document_id"],
                    collection_name=dummy_col,
                    persist_directory=dummy_dir,
                )
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn("no extractable text", cm.exception.detail.lower())
        finally:
            shutil.rmtree(dummy_dir, ignore_errors=True)

    def test_long_pdf_batch_processing(self):
        # Create a document with 6 pages / chunks (> 4 chunks, triggering batching)
        long_pdf = create_test_pdf_with_text([
            "Page 1: Introduction to Data Structures and Arrays.",
            "Page 2: Linked Lists and pointer manipulations.",
            "Page 3: Stacks and Queues with push/pop operations.",
            "Page 4: Binary Search Trees and tree traversals.",
            "Page 5: Hash Tables, collision resolution, and load factors.",
            "Page 6: Graph algorithms, BFS, DFS, and topological sort."
        ])
        p_long, chunks_long = extract_and_chunk_pdf(long_pdf, "data_structures.pdf")
        res_long = index_document(
            filename="data_structures.pdf",
            total_pages=p_long,
            chunks=chunks_long,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )

        summary = summarize_document(
            document_id=res_long["document_id"],
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertEqual(summary["total_pages"], 6)
        self.assertEqual(summary["source_pages"], [1, 2, 3, 4, 5, 6])
        self.assertTrue(len(summary["overview"]) > 0)
        self.assertGreaterEqual(len(summary["main_topics"]), 1)
        self.assertGreaterEqual(len(summary["important_points"]), 1)

    def test_ollama_error_handling(self):
        # Point to unreachable Ollama URL
        with self.assertRaises(OllamaConnectionError):
            summarize_document(
                document_id=self.doc1_id,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
                ollama_base_url="http://127.0.0.1:59999",
            )


class TestFastApiSummarizeEndpoint(unittest.TestCase):

    def test_summarize_endpoint_empty_id_raises_400(self):
        async def run():
            with self.assertRaises(HTTPException) as cm:
                await summarize_endpoint(SummarizeRequest(document_id="   "))
            self.assertEqual(cm.exception.status_code, 400)

        asyncio.run(run())

    def test_summarize_endpoint_invalid_id_raises_404(self):
        async def run():
            with self.assertRaises(HTTPException) as cm:
                await summarize_endpoint(SummarizeRequest(document_id="invalid_id_9999"))
            self.assertEqual(cm.exception.status_code, 404)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
