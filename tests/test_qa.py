"""Unit and integration tests for RAG Question Answering feature with local Ollama."""

import asyncio
import io
import shutil
import tempfile
import unittest

from fastapi import HTTPException
import pymupdf as fitz

from backend.main import app, ask_endpoint, AskRequest
from backend.ollama_client import (
    check_ollama_availability,
    generate_rag_answer,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaError,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)
from backend.pdf_parser import extract_and_chunk_pdf
from backend.qa_service import answer_study_question
from backend.vector_store import index_document


class TestOllamaClient(unittest.TestCase):

    def test_ollama_availability_and_qwen_model(self):
        # Local Ollama must be reachable and qwen2.5:1.5b must be present
        try:
            check_ollama_availability(base_url=OLLAMA_BASE_URL, model=DEFAULT_OLLAMA_MODEL)
        except (OllamaConnectionError, OllamaModelNotFoundError) as e:
            self.fail(f"Ollama availability check failed: {e}")

    def test_ollama_connection_error_on_invalid_port(self):
        # Pointing to an invalid port should raise OllamaConnectionError
        with self.assertRaises(OllamaConnectionError) as cm:
            check_ollama_availability(base_url="http://127.0.0.1:59999", model=DEFAULT_OLLAMA_MODEL, timeout=1.0)
        self.assertIn("unreachable", str(cm.exception).lower())

    def test_ollama_missing_model_error(self):
        # Requesting a non-existent model should raise OllamaModelNotFoundError
        with self.assertRaises(OllamaModelNotFoundError) as cm:
            check_ollama_availability(base_url=OLLAMA_BASE_URL, model="definitely_fake_model:99b", timeout=3.0)
        self.assertIn("not installed", str(cm.exception).lower())


class TestQuestionAnsweringWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_qa_collection"

        # Create a real 2-page test PDF
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            "Alan Turing proposed the Turing Test in 1950 as a test of machine intelligence. "
            "John McCarthy coined the term Artificial Intelligence at the Dartmouth Workshop in 1956.",
            fontsize=11
        )
        p2 = doc.new_page()
        p2.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            "Supervised learning algorithms map inputs to outputs using labeled training datasets. "
            "Common supervised learning tasks include classification and numerical regression.",
            fontsize=11
        )
        pdf_bytes = doc.tobytes()
        doc.close()

        total_pages, chunks = extract_and_chunk_pdf(pdf_bytes, "ai_history.pdf")
        index_document(
            filename="ai_history.pdf",
            total_pages=total_pages,
            chunks=chunks,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_qa_factual_answer_with_sources(self):
        # Question answerable from Page 1
        res = answer_study_question(
            question="Who proposed the Turing Test and in what year?",
            top_k=2,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )

        self.assertIn("Turing", res["answer"])
        self.assertIn("1950", res["answer"])
        self.assertGreaterEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["filename"], "ai_history.pdf")
        self.assertEqual(res["sources"][0]["page_number"], 1)
        self.assertGreaterEqual(len(res["context_used"]), 1)

    def test_qa_second_page_factual_answer(self):
        # Question answerable from Page 2
        res = answer_study_question(
            question="What are two common tasks in supervised learning?",
            top_k=2,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )

        lower_ans = res["answer"].lower()
        self.assertTrue("classification" in lower_ans or "regression" in lower_ans)
        self.assertGreaterEqual(len(res["sources"]), 1)
        # Check source points to page 2
        page_nums = [s["page_number"] for s in res["sources"]]
        self.assertIn(2, page_nums)

    def test_qa_missing_context_offtopic(self):
        # Completely off-topic question not present in the documents
        res = answer_study_question(
            question="How do you make strawberry ice cream at home?",
            top_k=2,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
        )

        lower_ans = res["answer"].lower()
        self.assertTrue("not contain" in lower_ans or "no information" in lower_ans)
        self.assertEqual(res["sources"], [])

    def test_qa_empty_question_validation(self):
        with self.assertRaises(ValueError):
            answer_study_question(
                question="   ",
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
            )


class TestAskApiEndpoint(unittest.TestCase):

    def test_ask_endpoint_empty_question_raises_400(self):
        async def run():
            with self.assertRaises(HTTPException) as cm:
                await ask_endpoint(AskRequest(question="   "))
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn("empty", cm.exception.detail.lower())

        asyncio.run(run())

    def test_ask_endpoint_success(self):
        async def run():
            req = AskRequest(question="What is the Turing Test?", top_k=2)
            resp = await ask_endpoint(req)

            self.assertIsNotNone(resp["answer"])
            self.assertIn("question", resp)
            self.assertIn("sources", resp)
            self.assertIn("context_used", resp)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
