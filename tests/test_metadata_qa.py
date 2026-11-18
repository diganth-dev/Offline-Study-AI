"""Unit and integration tests for document metadata question answering in Offline Study AI.

Verifies that questions regarding document counts, filenames, page counts,
total pages, and chunk counts are resolved directly from ChromaDB metadata
without invoking the Ollama LLM, with proper disambiguation for multiple documents.
"""

import asyncio
import io
import shutil
import tempfile
import unittest

from fastapi import HTTPException
import pymupdf as fitz

from backend.main import ask_endpoint, AskRequest
from backend.metadata_service import (
    find_matching_document,
    is_documents_list_or_count_query,
    is_total_pages_query,
    is_page_count_query,
    is_chunk_count_query,
    handle_metadata_question,
)
from backend.ollama_client import OllamaConnectionError
from backend.pdf_parser import extract_and_chunk_pdf
from backend.qa_service import answer_study_question
from backend.vector_store import index_document, get_indexed_documents


def create_dummy_pdf(num_pages: int, text_prefix: str) -> bytes:
    """Helper to generate an in-memory PDF with specified number of pages."""
    doc = fitz.open()
    for i in range(1, num_pages + 1):
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(50, 50, 500, 700),
            f"This is page {i} of {text_prefix}. It contains lecture notes and concepts.",
            fontsize=12,
        )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestMetadataClassification(unittest.TestCase):
    """Tests the classification helpers for metadata question detection."""

    def test_document_count_and_list_queries(self):
        self.assertTrue(is_documents_list_or_count_query("how many documents do i have"))
        self.assertTrue(is_documents_list_or_count_query("how many files are uploaded"))
        self.assertTrue(is_documents_list_or_count_query("total documents"))
        self.assertTrue(is_documents_list_or_count_query("what are the filenames"))
        self.assertTrue(is_documents_list_or_count_query("what documents are uploaded"))
        self.assertTrue(is_documents_list_or_count_query("list my documents"))
        self.assertTrue(is_documents_list_or_count_query("show uploaded files"))
        self.assertFalse(is_documents_list_or_count_query("what is the summary of the document"))

    def test_total_pages_queries(self):
        self.assertTrue(is_total_pages_query("how many total pages are there"))
        self.assertTrue(is_total_pages_query("total pages"))
        self.assertTrue(is_total_pages_query("total page count"))
        self.assertTrue(is_total_pages_query("pages across all documents"))
        self.assertTrue(is_total_pages_query("how many pages in all files"))
        self.assertFalse(is_total_pages_query("what is on page 2"))

    def test_page_count_queries(self):
        self.assertTrue(is_page_count_query("how many pages are there"))
        self.assertTrue(is_page_count_query("how many pages"))
        self.assertTrue(is_page_count_query("page count"))
        self.assertTrue(is_page_count_query("number of pages in the pdf"))
        self.assertTrue(is_page_count_query("how many pages does lecture1.pdf have"))
        # Excludes chapter or section inquiries
        self.assertFalse(is_page_count_query("how many pages are in chapter 2"))
        self.assertFalse(is_page_count_query("what is the main idea on page 1"))

    def test_chunk_count_queries(self):
        self.assertTrue(is_chunk_count_query("how many chunks are there"))
        self.assertTrue(is_chunk_count_query("how many total chunks"))
        self.assertTrue(is_chunk_count_query("chunk count"))
        self.assertTrue(is_chunk_count_query("number of indexed chunks"))
        self.assertFalse(is_chunk_count_query("explain chunking in rag"))

    def test_find_matching_document(self):
        docs = [
            {"filename": "operating_systems.pdf", "total_pages": 10, "total_chunks": 15},
            {"filename": "linear_algebra.pdf", "total_pages": 5, "total_chunks": 8},
        ]
        # Exact filename match
        self.assertEqual(
            find_matching_document("How many pages in operating_systems.pdf?", docs)["filename"],
            "operating_systems.pdf"
        )
        # Stem match
        self.assertEqual(
            find_matching_document("How many pages does linear_algebra have?", docs)["filename"],
            "linear_algebra.pdf"
        )
        # Normalized stem match with spaces
        self.assertEqual(
            find_matching_document("What is the page count of linear algebra?", docs)["filename"],
            "linear_algebra.pdf"
        )
        # No match
        self.assertIsNone(find_matching_document("How many pages are there?", docs))


class TestMetadataEmptyStore(unittest.TestCase):
    """Tests metadata query handling when no documents are indexed."""

    def test_empty_documents_inquiries(self):
        empty_docs = []

        res = handle_metadata_question("How many documents are there?", empty_docs)
        self.assertIsNotNone(res)
        self.assertIn("No documents have been uploaded yet", res["answer"])
        self.assertEqual(res["sources"], [])

        res_pages = handle_metadata_question("How many pages are there?", empty_docs)
        self.assertIsNotNone(res_pages)
        self.assertIn("No documents have been uploaded yet", res_pages["answer"])

        res_total = handle_metadata_question("How many total pages are there?", empty_docs)
        self.assertIsNotNone(res_total)
        self.assertIn("0 pages", res_total["answer"])

        res_chunks = handle_metadata_question("How many chunks are there?", empty_docs)
        self.assertIsNotNone(res_chunks)
        self.assertIn("0 indexed chunks", res_chunks["answer"])


class TestMetadataSingleDocument(unittest.TestCase):
    """Tests metadata queries when exactly one document is indexed."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_single_doc_meta"

        pdf_bytes = create_dummy_pdf(num_pages=3, text_prefix="Biology Notes")
        total_pages, chunks = extract_and_chunk_pdf(pdf_bytes, "biology_101.pdf")
        index_document(
            filename="biology_101.pdf",
            total_pages=total_pages,
            chunks=chunks,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_single_doc_page_count_unambiguous(self):
        # When 1 doc exists, "How many pages are there?" should answer directly without ambiguity
        res = answer_study_question(
            question="How many pages are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("biology_101.pdf", res["answer"])
        self.assertIn("3 page(s)", res["answer"])
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["filename"], "biology_101.pdf")
        self.assertEqual(res["context_used"], [])

    def test_single_doc_named_page_count(self):
        res = answer_study_question(
            question="How many pages in biology_101.pdf?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("biology_101.pdf", res["answer"])
        self.assertIn("3 page(s)", res["answer"])

    def test_single_doc_total_pages(self):
        res = answer_study_question(
            question="What is the total number of pages?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("3 page(s)", res["answer"])

    def test_single_doc_document_count_and_filenames(self):
        res = answer_study_question(
            question="What are the filenames of my documents?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("biology_101.pdf", res["answer"])
        self.assertIn("1 document uploaded", res["answer"])

    def test_single_doc_chunk_count(self):
        res = answer_study_question(
            question="How many chunks are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("biology_101.pdf", res["answer"])
        self.assertIn("indexed chunk(s)", res["answer"])


class TestMetadataMultipleDocuments(unittest.TestCase):
    """Tests metadata queries when multiple documents are indexed."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_multi_doc_meta"

        # Doc 1: physics.pdf (2 pages)
        pdf1 = create_dummy_pdf(num_pages=2, text_prefix="Physics")
        pages1, chunks1 = extract_and_chunk_pdf(pdf1, "physics.pdf")
        index_document(
            filename="physics.pdf",
            total_pages=pages1,
            chunks=chunks1,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )

        # Doc 2: chemistry.pdf (4 pages)
        pdf2 = create_dummy_pdf(num_pages=4, text_prefix="Chemistry")
        pages2, chunks2 = extract_and_chunk_pdf(pdf2, "chemistry.pdf")
        index_document(
            filename="chemistry.pdf",
            total_pages=pages2,
            chunks=chunks2,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_multi_doc_ambiguous_page_count_asks_which_document(self):
        # Multiple documents exist, user asked without specifying which one
        res = answer_study_question(
            question="How many pages are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        # Should ask which document the user means and list both candidates
        self.assertIn("Which document do you mean?", res["answer"])
        self.assertIn("physics.pdf", res["answer"])
        self.assertIn("chemistry.pdf", res["answer"])
        self.assertEqual(len(res["sources"]), 2)

    def test_multi_doc_ambiguous_chunk_count_asks_which_document(self):
        res = answer_study_question(
            question="How many chunks are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("Which document do you mean?", res["answer"])
        self.assertIn("physics.pdf", res["answer"])
        self.assertIn("chemistry.pdf", res["answer"])

    def test_multi_doc_specific_document_page_count(self):
        res1 = answer_study_question(
            question="How many pages are in physics.pdf?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("physics.pdf", res1["answer"])
        self.assertIn("2 page(s)", res1["answer"])
        self.assertEqual(len(res1["sources"]), 1)
        self.assertEqual(res1["sources"][0]["filename"], "physics.pdf")

        res2 = answer_study_question(
            question="How many pages does chemistry have?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("chemistry.pdf", res2["answer"])
        self.assertIn("4 page(s)", res2["answer"])
        self.assertEqual(len(res2["sources"]), 1)
        self.assertEqual(res2["sources"][0]["filename"], "chemistry.pdf")

    def test_multi_doc_explicit_total_pages(self):
        # 2 + 4 = 6 total pages
        res = answer_study_question(
            question="How many total pages are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("6 total pages", res["answer"])
        self.assertIn("physics.pdf", res["answer"])
        self.assertIn("chemistry.pdf", res["answer"])
        self.assertEqual(len(res["sources"]), 2)

    def test_multi_doc_total_chunks(self):
        res = answer_study_question(
            question="How many total chunks are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("total indexed chunks", res["answer"])
        self.assertIn("physics.pdf", res["answer"])
        self.assertIn("chemistry.pdf", res["answer"])

    def test_multi_doc_list_documents(self):
        res = answer_study_question(
            question="What documents are uploaded?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )
        self.assertIn("You have 2 documents uploaded", res["answer"])
        self.assertIn("physics.pdf", res["answer"])
        self.assertIn("chemistry.pdf", res["answer"])
        self.assertEqual(len(res["sources"]), 2)


class TestMetadataOllamaBypass(unittest.TestCase):
    """Ensures that metadata inquiries bypass Ollama completely without network calls."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_ollama_bypass"

        pdf = create_dummy_pdf(num_pages=1, text_prefix="Bypass")
        pages, chunks = extract_and_chunk_pdf(pdf, "bypass_doc.pdf")
        index_document(
            filename="bypass_doc.pdf",
            total_pages=pages,
            chunks=chunks,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_metadata_question_bypasses_offline_ollama(self):
        # Pointing to an invalid Ollama port (e.g. 59999).
        # If metadata question incorrectly invoked Ollama, this would raise OllamaConnectionError.
        # Since it's handled directly from ChromaDB metadata, it must succeed cleanly!
        res = answer_study_question(
            question="How many pages are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            ollama_base_url="http://127.0.0.1:59999",
        )
        self.assertIn("bypass_doc.pdf", res["answer"])
        self.assertIn("1 page(s)", res["answer"])

    def test_normal_study_question_still_calls_ollama(self):
        # Normal content question with an unreachable Ollama URL must raise OllamaConnectionError,
        # confirming that normal questions still route through Ollama RAG as expected.
        with self.assertRaises(OllamaConnectionError):
            answer_study_question(
                question="What is the key theorem explained in the text?",
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
                ollama_base_url="http://127.0.0.1:59999",
            )


if __name__ == "__main__":
    unittest.main()
