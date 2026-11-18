"""Unit and integration tests for document-specific AI conversations in Offline Study AI.

Verifies:
1. Document filtering in ChromaDB vector store search.
2. Metadata question handling in document-specific mode (answers strictly for selected doc without asking for clarification).
3. Preservation of general multi-document chat when document_id is omitted.
4. Source citations strictly isolated to the selected document in scoped mode.
5. FastAPI /ask and /search endpoints with optional document_id parameter.
"""

import asyncio
import io
import shutil
import tempfile
import unittest

from fastapi import HTTPException
import pymupdf as fitz

from backend.main import ask_endpoint, AskRequest, search_endpoint, SearchRequest
from backend.ollama_client import OllamaConnectionError
from backend.pdf_parser import extract_and_chunk_pdf
from backend.qa_service import answer_study_question
from backend.vector_store import index_document, search_documents, get_indexed_documents


def make_test_pdf(num_pages: int, subject: str, text_content: str) -> bytes:
    """Helper to generate an in-memory PDF with known pages and subject content."""
    doc = fitz.open()
    for i in range(1, num_pages + 1):
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(50, 50, 500, 700),
            f"Subject: {subject}. Page {i}. {text_content}",
            fontsize=11,
        )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestDocumentSpecificConversations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_doc_specific_col"

        # Document 1: Operating Systems (2 pages)
        pdf1 = make_test_pdf(
            num_pages=2,
            subject="Operating Systems",
            text_content="A process is an instance of a computer program being executed. "
                         "Threads share memory space within the same process. "
                         "Semaphores prevent race conditions and synchronize concurrent threads."
        )
        p1, chunks1 = extract_and_chunk_pdf(pdf1, "operating_systems.pdf")
        res1 = index_document(
            filename="operating_systems.pdf",
            total_pages=p1,
            chunks=chunks1,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.doc1_id = res1["document_id"]

        # Document 2: World History (3 pages)
        pdf2 = make_test_pdf(
            num_pages=3,
            subject="French Revolution",
            text_content="The French Revolution began in 1789 with the storming of the Bastille. "
                         "The Declaration of the Rights of Man established fundamental liberties. "
                         "Napoleon Bonaparte seized power in 1799."
        )
        p2, chunks2 = extract_and_chunk_pdf(pdf2, "french_revolution.pdf")
        res2 = index_document(
            filename="french_revolution.pdf",
            total_pages=p2,
            chunks=chunks2,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.doc2_id = res2["document_id"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. Vector Search Scoping Tests
    # -------------------------------------------------------------

    def test_vector_search_filtered_to_selected_document(self):
        # Search for OS term "process" restricted to doc1
        results1 = search_documents(
            query="What is a process and thread?",
            top_k=5,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc1_id,
        )
        self.assertGreaterEqual(len(results1), 1)
        for r in results1:
            self.assertEqual(r["document_id"], self.doc1_id)
            self.assertEqual(r["filename"], "operating_systems.pdf")

        # Search for OS term "process" restricted to doc2 (History) -> should find nothing or low similarity
        results2 = search_documents(
            query="What is a process and thread?",
            top_k=5,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc2_id,
        )
        # All returned chunks if any must only be from doc2
        for r in results2:
            self.assertEqual(r["document_id"], self.doc2_id)
            self.assertEqual(r["filename"], "french_revolution.pdf")

    def test_vector_search_unfiltered_general_mode(self):
        # Without document_id, search returns relevant chunks across all documents
        results = search_documents(
            query="Bastille Napoleon",
            top_k=5,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=None,
        )
        self.assertGreaterEqual(len(results), 1)
        filenames = [r["filename"] for r in results]
        self.assertIn("french_revolution.pdf", filenames)

    # -------------------------------------------------------------
    # 2. Document-Specific Metadata Queries Tests
    # -------------------------------------------------------------

    def test_metadata_page_count_scoped_to_doc1(self):
        # User asks "How many pages are there?" while doc1 (2 pages) is selected
        res = answer_study_question(
            question="How many pages are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc1_id,
        )
        self.assertIn("operating_systems.pdf", res["answer"])
        self.assertIn("2 page(s)", res["answer"])
        # Must NOT ask which document the user means
        self.assertNotIn("Which document do you mean?", res["answer"])
        # Sources must strictly be doc1
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["filename"], "operating_systems.pdf")
        self.assertEqual(res.get("document_id"), self.doc1_id)

    def test_metadata_page_count_scoped_to_doc2(self):
        # User asks "How many pages are there?" while doc2 (3 pages) is selected
        res = answer_study_question(
            question="How many pages are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc2_id,
        )
        self.assertIn("french_revolution.pdf", res["answer"])
        self.assertIn("3 page(s)", res["answer"])
        self.assertNotIn("Which document do you mean?", res["answer"])
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["filename"], "french_revolution.pdf")
        self.assertEqual(res.get("document_id"), self.doc2_id)

    def test_metadata_chunks_count_scoped(self):
        res = answer_study_question(
            question="How many chunks are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc1_id,
        )
        self.assertIn("operating_systems.pdf", res["answer"])
        self.assertIn("indexed chunk(s)", res["answer"])
        self.assertNotIn("Which document do you mean?", res["answer"])
        self.assertEqual(res.get("document_id"), self.doc1_id)

    def test_metadata_total_pages_scoped(self):
        res = answer_study_question(
            question="What is the total page count?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc2_id,
        )
        self.assertIn("french_revolution.pdf", res["answer"])
        self.assertIn("3 page(s)", res["answer"])
        self.assertEqual(res.get("document_id"), self.doc2_id)

    def test_metadata_current_document_inquiry(self):
        res = answer_study_question(
            question="What document is this?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc1_id,
        )
        self.assertIn("operating_systems.pdf", res["answer"])
        self.assertIn("switch to general chat", res["answer"].lower())
        self.assertEqual(res.get("document_id"), self.doc1_id)

    # -------------------------------------------------------------
    # 3. General Chat Preservation Tests
    # -------------------------------------------------------------

    def test_general_chat_ambiguous_question_asks_user(self):
        # In general chat (document_id=None), ambiguous question triggers clarification prompt
        res = answer_study_question(
            question="How many pages are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=None,
        )
        self.assertIn("Which document do you mean?", res["answer"])
        self.assertIn("operating_systems.pdf", res["answer"])
        self.assertIn("french_revolution.pdf", res["answer"])
        self.assertIsNone(res.get("document_id"))

    def test_general_chat_total_pages(self):
        # 2 + 3 = 5 total pages across both documents
        res = answer_study_question(
            question="How many total pages are there?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=None,
        )
        self.assertIn("5 total pages", res["answer"])
        self.assertIn("operating_systems.pdf", res["answer"])
        self.assertIn("french_revolution.pdf", res["answer"])

    # -------------------------------------------------------------
    # 4. Source Citation & Scope Isolation Tests
    # -------------------------------------------------------------

    def test_study_question_sources_isolated_to_selected_document(self):
        # Question answerable by doc1
        res = answer_study_question(
            question="What is a process and what do semaphores do?",
            top_k=3,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc1_id,
        )
        self.assertGreaterEqual(len(res["sources"]), 1)
        for s in res["sources"]:
            self.assertEqual(s["filename"], "operating_systems.pdf")
        for ctx in res["context_used"]:
            self.assertEqual(ctx["document_id"], self.doc1_id)
            self.assertEqual(ctx["filename"], "operating_systems.pdf")

    def test_study_question_cross_document_leakage_prevented(self):
        # While doc1 (OS) is selected, ask a question about Napoleon (doc2 topic)
        # It must NOT retrieve French Revolution chunks from doc2
        res = answer_study_question(
            question="When did Napoleon seize power and what was the Bastille?",
            top_k=3,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc1_id,
        )
        # Even if chunks were retrieved from doc1, none of them should be from doc2
        for ctx in res.get("context_used", []):
            self.assertEqual(ctx["document_id"], self.doc1_id)
            self.assertNotEqual(ctx["filename"], "french_revolution.pdf")
        for s in res.get("sources", []):
            self.assertNotEqual(s["filename"], "french_revolution.pdf")

    # -------------------------------------------------------------
    # 5. Contextual Document Instructions Tests
    # -------------------------------------------------------------

    def test_contextual_instruction_explain_briefly(self):
        # User says "Explain briefly." while doc1 (OS) is selected
        res = answer_study_question(
            question="Explain briefly.",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc1_id,
        )
        self.assertNotIn("does not contain information", res["answer"].lower())
        self.assertTrue(
            "process" in res["answer"].lower()
            or "thread" in res["answer"].lower()
            or "semaphore" in res["answer"].lower()
            or "operating" in res["answer"].lower()
        )
        self.assertEqual(res.get("document_id"), self.doc1_id)
        self.assertGreaterEqual(len(res["sources"]), 1)
        for s in res["sources"]:
            self.assertEqual(s["filename"], "operating_systems.pdf")

    def test_contextual_instruction_summarize_document(self):
        # User says "Summarize this document" while doc2 (History) is selected
        res = answer_study_question(
            question="Summarize this document",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc2_id,
        )
        self.assertNotIn("does not contain information", res["answer"].lower())
        self.assertTrue(
            "french" in res["answer"].lower()
            or "revolution" in res["answer"].lower()
            or "bastille" in res["answer"].lower()
            or "napoleon" in res["answer"].lower()
        )
        self.assertEqual(res.get("document_id"), self.doc2_id)
        self.assertGreaterEqual(len(res["sources"]), 1)
        for s in res["sources"]:
            self.assertEqual(s["filename"], "french_revolution.pdf")

    def test_contextual_instruction_main_topics(self):
        # User asks "What are the main topics?" while doc1 is selected
        res = answer_study_question(
            question="What are the main topics?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc1_id,
        )
        self.assertNotIn("does not contain information", res["answer"].lower())
        self.assertGreaterEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["filename"], "operating_systems.pdf")

    def test_contextual_instruction_give_important_points(self):
        # User says "Give important points" while doc2 is selected
        res = answer_study_question(
            question="Give important points",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc2_id,
        )
        self.assertNotIn("does not contain information", res["answer"].lower())
        self.assertGreaterEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["filename"], "french_revolution.pdf")

    def test_contextual_instruction_explain_complete_document(self):
        # User says "Explain the complete document" while doc1 is selected
        res = answer_study_question(
            question="Explain the complete document",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.doc1_id,
        )
        self.assertNotIn("does not contain information", res["answer"].lower())
        self.assertGreaterEqual(len(res["sources"]), 1)

    def test_non_extractable_text_document_handling(self):
        # Create an artificial document record with empty chunk text
        dummy_dir = tempfile.mkdtemp()
        dummy_col = "test_empty_text_col"
        try:
            # Index a document that has empty/whitespace text
            empty_chunks = [{"chunk_number": 1, "page_number": 1, "text": "   "}]
            meta_res = index_document(
                filename="scanned_blank.pdf",
                total_pages=1,
                chunks=empty_chunks,
                collection_name=dummy_col,
                persist_directory=dummy_dir,
            )
            blank_doc_id = meta_res["document_id"]

            res = answer_study_question(
                question="Explain briefly.",
                collection_name=dummy_col,
                persist_directory=dummy_dir,
                document_id=blank_doc_id,
            )
            self.assertIn("no extractable text", res["answer"].lower())
            self.assertEqual(res["sources"], [])
        finally:
            shutil.rmtree(dummy_dir, ignore_errors=True)

    def test_pipeline_verification_for_both_tested_pdfs(self):
        # Verify complete extraction and chunking pipeline directly for both PDFs
        pdf1 = make_test_pdf(num_pages=2, subject="OS", text_content="Process execution and thread safety.")
        p1, chunks1 = extract_and_chunk_pdf(pdf1, "test_os_pipeline.pdf")
        self.assertEqual(p1, 2)
        self.assertGreaterEqual(len(chunks1), 2)
        self.assertEqual(chunks1[0]["page_number"], 1)
        self.assertEqual(chunks1[0]["chunk_number"], 1)
        self.assertIn("Process", chunks1[0]["text"])

        pdf2 = make_test_pdf(num_pages=3, subject="History", text_content="Bastille and Napoleon.")
        p2, chunks2 = extract_and_chunk_pdf(pdf2, "test_hist_pipeline.pdf")
        self.assertEqual(p2, 3)
        self.assertGreaterEqual(len(chunks2), 3)
        self.assertEqual(chunks2[0]["page_number"], 1)
        self.assertEqual(chunks2[-1]["page_number"], 3)
        self.assertIn("Bastille", chunks2[0]["text"])

    def test_invalid_document_id_gracefully_handled(self):
        res = answer_study_question(
            question="What is a process?",
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id="non_existent_doc_12345",
        )
        self.assertIn("not found", res["answer"].lower())
        self.assertEqual(res["sources"], [])


class TestFastApiDocumentScopedEndpoints(unittest.TestCase):

    def test_ask_endpoint_with_document_id(self):
        async def run():
            req = AskRequest(
                question="How many pages are there?",
                document_id="some_test_doc_id"
            )
            self.assertEqual(req.document_id, "some_test_doc_id")

            # Calling endpoint with invalid document_id should return graceful not-found message
            resp = await ask_endpoint(req)
            self.assertIn("not found", resp["answer"].lower())
            self.assertEqual(resp["document_id"], "some_test_doc_id")

        asyncio.run(run())

    def test_search_endpoint_with_document_id(self):
        async def run():
            req = SearchRequest(
                query="processes and memory",
                top_k=3,
                document_id="filter_doc_id"
            )
            self.assertEqual(req.document_id, "filter_doc_id")
            resp = await search_endpoint(req)
            self.assertIn("total_results", resp)
            self.assertIn("results", resp)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
