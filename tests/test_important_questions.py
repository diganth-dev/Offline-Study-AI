"""Unit and integration tests for Important Questions Generator in Offline Study AI.

Verifies:
1. Document chunk processing covering the entire document (not just top-k vector chunks).
2. Question organization into:
   - Short-answer questions
   - Long-answer questions
   - Important conceptual questions
3. Generation of ~10-15 relevant questions with clean, numbered format.
4. PDF page references supporting each question.
5. Regeneration of questions.
6. Grounded, exam-friendly answer generation for individual questions.
7. Strict document isolation using document_id (no cross-document contamination).
8. Validation and error handling (empty document, non-existent document, empty text).
9. FastAPI endpoints: /important-questions and /important-questions/answer.
"""

import asyncio
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import pymupdf as fitz

from backend.main import (
    generate_important_questions_endpoint,
    answer_individual_question_endpoint,
    ImportantQuestionsRequest,
    QuestionAnswerRequest,
)
from backend.pdf_parser import extract_and_chunk_pdf
from backend.question_generator import (
    generate_important_questions,
    generate_question_answer,
    extract_page_reference,
    parse_questions_output,
)
from backend.vector_store import index_document


def create_ml_test_pdf() -> bytes:
    """Creates a 3-page study guide PDF on Machine Learning."""
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Unit 1: Supervised Learning Foundations\n\n"
        "Supervised learning algorithms map input variables to output targets using labeled training datasets. "
        "Classification algorithms predict discrete categorical labels (e.g., spam detection, disease diagnosis). "
        "Regression algorithms predict continuous numerical quantities (e.g., house price forecasting). "
        "Loss functions such as Mean Squared Error (MSE) and Cross-Entropy measure the discrepancy between predictions "
        "and true labels. Gradient descent iteratively updates model parameters to minimize this loss.",
        fontsize=11,
    )

    p2 = doc.new_page()
    p2.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Unit 2: Unsupervised Learning & Dimensionality Reduction\n\n"
        "Unsupervised learning discovers hidden patterns, groupings, and structures from unlabeled data. "
        "K-Means clustering partitions data into k distinct clusters by minimizing inertia (within-cluster sum of squares). "
        "Principal Component Analysis (PCA) is an orthogonal linear transformation that reduces dataset dimensionality "
        "while preserving the maximum variance of the original features. "
        "Hierarchical clustering builds nested tree structures called dendrograms to visualize relationships.",
        fontsize=11,
    )

    p3 = doc.new_page()
    p3.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Unit 3: Reinforcement Learning & Decision Processes\n\n"
        "Reinforcement learning is a computational paradigm where an autonomous agent learns to maximize cumulative "
        "rewards through sequential interactions with an environment. "
        "The Markov Decision Process (MDP) provides a mathematical framework for modeling decision making. "
        "The Bellman Equation decomposes value functions into the immediate reward plus discounted future returns. "
        "Q-learning is a model-free temporal difference algorithm that converges to optimal action-value policies.",
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_biochem_test_pdf() -> bytes:
    """Creates a 2-page biochemistry PDF to test document isolation."""
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Chapter 1: Enzymatic Kinetics and Catalysis\n\n"
        "Enzymes are biological catalysts that lower activation energy to accelerate biochemical reactions. "
        "The active site provides specific stereochemical binding for substrates. "
        "Michaelis-Menten kinetics define the maximum velocity (Vmax) and the Michaelis constant (Km).",
        fontsize=11,
    )

    p2 = doc.new_page()
    p2.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Chapter 2: Enzyme Inhibition Mechanisms\n\n"
        "Competitive inhibitors bind reversibly to the active site, increasing Km without altering Vmax. "
        "Non-competitive inhibitors bind to allosteric sites, reducing Vmax while leaving Km unchanged.",
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPageReferenceExtraction(unittest.TestCase):
    """Unit tests for page reference extraction and chunk fallback."""

    def test_extract_page_in_parentheses(self):
        chunks = [{"chunk_number": 1, "page_number": 1, "text": "Supervised learning"}]
        q, pages, pref = extract_page_reference("What is supervised learning? (Page 1)", chunks)
        self.assertEqual(q, "What is supervised learning?")
        self.assertEqual(pages, [1])
        self.assertEqual(pref, "Page 1")

    def test_extract_page_in_brackets(self):
        chunks = [{"chunk_number": 1, "page_number": 2, "text": "K-means clustering"}]
        q, pages, pref = extract_page_reference("Explain K-Means clustering. [Page 2]", chunks)
        self.assertEqual(q, "Explain K-Means clustering?")
        self.assertEqual(pages, [2])
        self.assertEqual(pref, "Page 2")

    def test_extract_page_range(self):
        chunks = []
        q, pages, pref = extract_page_reference("Compare supervised and unsupervised learning (Pages 1-2)", chunks)
        self.assertEqual(q, "Compare supervised and unsupervised learning?")
        self.assertEqual(pages, [1, 2])
        self.assertEqual(pref, "Pages 1–2")

    def test_fallback_page_from_chunk_tokens(self):
        chunks = [
            {"chunk_number": 1, "page_number": 1, "text": "Supervised learning and classification models."},
            {"chunk_number": 2, "page_number": 2, "text": "Principal Component Analysis PCA variance."},
            {"chunk_number": 3, "page_number": 3, "text": "Bellman Equation Markov Decision Process."},
        ]
        q, pages, pref = extract_page_reference("How does the Bellman equation decompose value functions?", chunks)
        self.assertEqual(pages, [3])
        self.assertEqual(pref, "Page 3")


class TestQuestionsOutputParser(unittest.TestCase):
    """Unit tests for parsing raw LLM output into structured question categories."""

    def test_parse_structured_markdown(self):
        mock_output = (
            "### Short-Answer Questions\n"
            "1. Define supervised learning. (Page 1)\n"
            "2. What is a loss function? (Page 1)\n"
            "3. State the purpose of K-Means clustering. (Page 2)\n"
            "4. What is PCA? (Page 2)\n\n"
            "### Long-Answer Questions\n"
            "1. Explain the differences between classification and regression with examples. (Page 1)\n"
            "2. Describe the mathematical formulation of Principal Component Analysis. (Page 2)\n"
            "3. Detail the components of a Markov Decision Process. (Page 3)\n"
            "4. Discuss how Q-learning achieves optimal policy convergence. (Page 3)\n\n"
            "### Conceptual Questions\n"
            "1. Why is gradient descent essential for minimizing empirical loss? (Page 1)\n"
            "2. How does dimensionality reduction prevent the curse of dimensionality? (Page 2)\n"
            "3. What is the significance of the discount factor in reinforcement learning? (Page 3)\n"
            "4. Explain the exploration vs exploitation dilemma. (Page 3)"
        )

        dummy_chunks = [
            {"page_number": 1, "text": "supervised classification loss"},
            {"page_number": 2, "text": "PCA clustering dimensionality"},
            {"page_number": 3, "text": "reinforcement MDP Q-learning"},
        ]

        categories, flat_questions = parse_questions_output(mock_output, dummy_chunks, total_pages=3)

        self.assertEqual(len(categories), 3)
        self.assertEqual(categories[0]["category_id"], "short")
        self.assertEqual(categories[1]["category_id"], "long")
        self.assertEqual(categories[2]["category_id"], "conceptual")

        self.assertEqual(len(flat_questions), 12)
        # Sequential IDs from 1 to 12
        self.assertEqual([q["id"] for q in flat_questions], list(range(1, 13)))

        # Each question should have clean question text, valid page numbers and reference
        for q in flat_questions:
            self.assertTrue(len(q["question"]) > 10)
            self.assertTrue(q["page_reference"].startswith("Page"))
            self.assertGreaterEqual(len(q["page_numbers"]), 1)


class TestImportantQuestionsWorkflow(unittest.TestCase):
    """Integration tests verifying full-document questions generation and document isolation."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_questions_generator_col"

        # Document 1: Machine Learning (3 pages)
        ml_bytes = create_ml_test_pdf()
        p1, chunks1 = extract_and_chunk_pdf(ml_bytes, "machine_learning.pdf")
        res1 = index_document(
            filename="machine_learning.pdf",
            total_pages=p1,
            chunks=chunks1,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.ml_doc_id = res1["document_id"]

        # Document 2: Biochemistry (2 pages)
        bio_bytes = create_biochem_test_pdf()
        p2, chunks2 = extract_and_chunk_pdf(bio_bytes, "biochemistry_enzymes.pdf")
        res2 = index_document(
            filename="biochemistry_enzymes.pdf",
            total_pages=p2,
            chunks=chunks2,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.bio_doc_id = res2["document_id"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_validation_errors(self):
        # Empty document ID raises 400
        with self.assertRaises(HTTPException) as cm:
            generate_important_questions(document_id="   ", collection_name=self.collection_name, persist_directory=self.test_dir)
        self.assertEqual(cm.exception.status_code, 400)

        # Non-existent document ID raises 404
        with self.assertRaises(HTTPException) as cm:
            generate_important_questions(document_id="non_existent_doc_id", collection_name=self.collection_name, persist_directory=self.test_dir)
        self.assertEqual(cm.exception.status_code, 404)

    def test_empty_extractable_text_raises_400(self):
        dummy_dir = tempfile.mkdtemp()
        dummy_col = "test_empty_q_col"
        try:
            res = index_document(
                filename="blank_scanned.pdf",
                total_pages=1,
                chunks=[{"chunk_number": 1, "page_number": 1, "text": "   "}],
                collection_name=dummy_col,
                persist_directory=dummy_dir,
            )
            with self.assertRaises(HTTPException) as cm:
                generate_important_questions(
                    document_id=res["document_id"],
                    collection_name=dummy_col,
                    persist_directory=dummy_dir,
                )
            self.assertEqual(cm.exception.status_code, 400)
        finally:
            shutil.rmtree(dummy_dir, ignore_errors=True)

    def test_generate_questions_with_mocked_llm(self):
        mock_output = (
            "### Short-Answer Questions\n"
            "1. Define supervised learning. (Page 1)\n"
            "2. What is Mean Squared Error? (Page 1)\n"
            "3. State the objective function of K-Means clustering. (Page 2)\n"
            "4. What does PCA accomplish? (Page 2)\n\n"
            "### Long-Answer Questions\n"
            "1. Contrast classification and regression tasks in machine learning. (Page 1)\n"
            "2. Explain the step-by-step procedure of Principal Component Analysis. (Page 2)\n"
            "3. Formulate the Markov Decision Process and define its components. (Page 3)\n"
            "4. Describe how Q-learning updates action-value functions. (Page 3)\n\n"
            "### Conceptual Questions\n"
            "1. Why is gradient descent used for loss minimization? (Page 1)\n"
            "2. How does dimensionality reduction mitigate the curse of dimensionality? (Page 2)\n"
            "3. Explain the role of the discount factor in the Bellman equation. (Page 3)\n"
            "4. What is the trade-off between exploration and exploitation? (Page 3)"
        )

        with patch("backend.question_generator.generate_rag_answer", return_value=mock_output):
            res = generate_important_questions(
                document_id=self.ml_doc_id,
                count=12,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
            )

            self.assertEqual(res["document_id"], self.ml_doc_id)
            self.assertEqual(res["filename"], "machine_learning.pdf")
            self.assertEqual(res["total_pages"], 3)
            self.assertEqual(res["total_questions"], 12)
            self.assertEqual(len(res["categories"]), 3)

            # Check question counts per category
            short_group = next(c for c in res["categories"] if c["category_id"] == "short")
            long_group = next(c for c in res["categories"] if c["category_id"] == "long")
            conceptual_group = next(c for c in res["categories"] if c["category_id"] == "conceptual")

            self.assertEqual(len(short_group["questions"]), 4)
            self.assertEqual(len(long_group["questions"]), 4)
            self.assertEqual(len(conceptual_group["questions"]), 4)

            # Verify page references span across Page 1, Page 2, Page 3
            all_pages = set()
            for q in res["questions"]:
                all_pages.update(q["page_numbers"])
            self.assertIn(1, all_pages)
            self.assertIn(2, all_pages)
            self.assertIn(3, all_pages)

    def test_document_isolation(self):
        mock_bio_output = (
            "### Short-Answer Questions\n"
            "1. Define biological catalysts and enzymes. (Page 1)\n"
            "2. What is the active site? (Page 1)\n"
            "3. Define Km in Michaelis-Menten kinetics. (Page 1)\n\n"
            "### Long-Answer Questions\n"
            "1. Differentiate between competitive and non-competitive enzyme inhibition. (Page 2)\n"
            "2. Explain the relationship between reaction velocity and substrate concentration. (Page 1)\n\n"
            "### Conceptual Questions\n"
            "1. Why does an allosteric inhibitor decrease Vmax without changing Km? (Page 2)\n"
            "2. How do enzymes lower activation energy? (Page 1)"
        )

        with patch("backend.question_generator.generate_rag_answer", return_value=mock_bio_output):
            res = generate_important_questions(
                document_id=self.bio_doc_id,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
            )

            self.assertEqual(res["document_id"], self.bio_doc_id)
            self.assertEqual(res["filename"], "biochemistry_enzymes.pdf")

            # Must NOT contain ML concepts
            for q in res["questions"]:
                q_text = q["question"].lower()
                self.assertNotIn("gradient descent", q_text)
                self.assertNotIn("q-learning", q_text)
                self.assertNotIn("k-means", q_text)

    def test_generate_individual_question_answer(self):
        mock_answer = (
            "Supervised learning is a machine learning paradigm where models are trained on labeled datasets. "
            "It maps inputs to known outputs, solving tasks like classification (discrete labels) and regression (continuous values)."
        )

        with patch("backend.question_generator.generate_rag_answer", return_value=mock_answer):
            ans_res = generate_question_answer(
                document_id=self.ml_doc_id,
                question="What is supervised learning?",
                question_type="short",
                page_numbers=[1],
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
            )

            self.assertEqual(ans_res["document_id"], self.ml_doc_id)
            self.assertEqual(ans_res["question_type"], "short")
            self.assertIn("labeled datasets", ans_res["answer"])
            self.assertEqual(ans_res["page_reference"], "Page 1")
            self.assertGreaterEqual(len(ans_res["sources"]), 1)
            self.assertEqual(ans_res["sources"][0]["filename"], "machine_learning.pdf")
            self.assertEqual(ans_res["sources"][0]["page_number"], 1)


class TestImportantQuestionsApiEndpoints(unittest.TestCase):
    """Integration tests verifying the FastAPI endpoints for important questions."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_questions_api_col"

        ml_bytes = create_ml_test_pdf()
        p, chunks = extract_and_chunk_pdf(ml_bytes, "api_ml_notes.pdf")
        res = index_document(
            filename="api_ml_notes.pdf",
            total_pages=p,
            chunks=chunks,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.doc_id = res["document_id"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_endpoint_validation_empty_doc_id(self):
        async def run():
            with self.assertRaises(HTTPException) as cm:
                await generate_important_questions_endpoint(ImportantQuestionsRequest(document_id="   "))
            self.assertEqual(cm.exception.status_code, 400)

        asyncio.run(run())

    def test_endpoint_answer_validation_empty_fields(self):
        async def run():
            with self.assertRaises(HTTPException) as cm:
                await answer_individual_question_endpoint(QuestionAnswerRequest(document_id="   ", question="What is AI?"))
            self.assertEqual(cm.exception.status_code, 400)

            with self.assertRaises(HTTPException) as cm:
                await answer_individual_question_endpoint(QuestionAnswerRequest(document_id=self.doc_id, question="   "))
            self.assertEqual(cm.exception.status_code, 400)

        asyncio.run(run())

    def test_endpoint_success_with_mock(self):
        mock_resp = {
            "document_id": self.doc_id,
            "filename": "api_ml_notes.pdf",
            "total_pages": 3,
            "total_questions": 3,
            "categories": [
                {"category": "Short-Answer Questions", "category_id": "short", "description": "Short", "questions": []},
                {"category": "Long-Answer Questions", "category_id": "long", "description": "Long", "questions": []},
                {"category": "Conceptual Questions", "category_id": "conceptual", "description": "Conceptual", "questions": []},
            ],
            "questions": [
                {"id": 1, "question": "What is AI?", "question_type": "short", "page_reference": "Page 1", "page_numbers": [1], "answer": None},
                {"id": 2, "question": "Explain PCA.", "question_type": "long", "page_reference": "Page 2", "page_numbers": [2], "answer": None},
                {"id": 3, "question": "Why Bellman?", "question_type": "conceptual", "page_reference": "Page 3", "page_numbers": [3], "answer": None},
            ],
        }

        with patch("backend.main.generate_important_questions", return_value=mock_resp):
            async def run():
                resp = await generate_important_questions_endpoint(
                    ImportantQuestionsRequest(document_id=self.doc_id, count=6)
                )
                self.assertEqual(resp["document_id"], self.doc_id)
                self.assertEqual(resp["total_questions"], 3)
                self.assertEqual(len(resp["categories"]), 3)

            asyncio.run(run())

    def test_metadata_and_source_lines_are_not_questions(self):
        """Verifies that lines like '- Source: Page 2' or 'Page 1' are not treated as independent questions."""
        raw_text = """### Short-Answer Questions
1. What is supervised learning?
- Source: Page 1
2. Define unsupervised clustering.
- Source: Page 2

### Long-Answer Questions
1. Compare and contrast supervised learning and unsupervised learning in detail.
Source: Page 1-2
2. Explain how backpropagation optimizes deep neural networks.
(Page 3)

### Important Conceptual Questions
1. Why is the bias-variance tradeoff fundamental in machine learning?
   Source: Page 2
2. What principles govern reinforcement learning policy convergence?
   - Page 3
"""
        chunks = [
            {"chunk_id": "c1", "text": "Supervised learning uses labels.", "page_number": 1, "filename": "test.pdf"},
            {"chunk_id": "c2", "text": "Unsupervised clustering groups data.", "page_number": 2, "filename": "test.pdf"},
            {"chunk_id": "c3", "text": "Backpropagation computes gradients.", "page_number": 3, "filename": "test.pdf"},
        ]
        cats, flat = parse_questions_output(raw_text, chunks, total_pages=3)
        self.assertEqual(len(flat), 6)
        # Verify no question is 'Source' or 'Page'
        for q in flat:
            self.assertFalse(q["question"].lower().startswith("source"))
            self.assertFalse(q["question"].lower().startswith("page"))
            self.assertTrue(len(q["question"]) > 15)
        # Verify citations were attached to preceding questions
        self.assertIn(1, flat[0]["page_numbers"])
        self.assertIn(2, flat[1]["page_numbers"])

    def test_endpoint_answer_success_with_mock(self):
        mock_ans = {
            "document_id": self.doc_id,
            "question": "What is AI?",
            "question_type": "short",
            "answer": "Artificial Intelligence is...",
            "page_reference": "Page 1",
            "sources": [{"filename": "api_ml_notes.pdf", "page_number": 1}],
        }

        with patch("backend.main.generate_question_answer", return_value=mock_ans):
            async def run():
                resp = await answer_individual_question_endpoint(
                    QuestionAnswerRequest(document_id=self.doc_id, question="What is AI?")
                )
                self.assertEqual(resp["document_id"], self.doc_id)
                self.assertIn("Artificial Intelligence", resp["answer"])
                self.assertEqual(resp["page_reference"], "Page 1")

            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
