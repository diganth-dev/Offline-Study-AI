"""Unit and integration tests for Structured Comparison Formatting in Offline Study AI.

Verifies:
1. Intelligent detection of comparison questions:
   - 'Differentiate between supervised and unsupervised learning'
   - 'What is the difference between classification and regression?'
   - 'Compare CNN and RNN'
   - 'Distinguish between BFS and DFS'
   - 'Supervised vs Unsupervised learning'
   - 'How does overfitting differ from underfitting?'
2. Negative classification: ordinary questions (definitions, summaries, factual lookups)
   must NOT be treated as comparison questions.
3. Extraction of comparison topics (Topic A, Topic B).
4. Response formatting via format_comparison_answer:
   - Short introduction.
   - Clean, readable Markdown comparison table with valid header and separator syntax.
   - Important points explained in student-friendly bullet points.
   - Relevant examples cited from the document.
   - Short exam-ready conclusion.
   - Inclusion of related concepts ONLY when supported, omitted otherwise.
   - Removal of conversational preambles and chatty intros.
5. Integration with answer_study_question (AI Study Chat):
   - Generates comparison format for comparison questions.
   - Preserves standard formatting for ordinary questions.
   - Preserves source citations (filename and page numbers).
6. Integration with generate_question_answer (Important Questions Generator):
   - Generates comparison format for comparison questions.
   - Preserves standard formatting for non-comparison questions.
   - Preserves document-grounded page references.
"""

import shutil
import tempfile
import unittest
from unittest.mock import patch

import pymupdf as fitz

from backend.metadata_service import (
    is_comparison_question,
    extract_comparison_topics,
    format_comparison_answer,
)
from backend.pdf_parser import extract_and_chunk_pdf
from backend.qa_service import answer_study_question
from backend.question_generator import generate_question_answer
from backend.vector_store import index_document


def create_sample_ml_pdf() -> bytes:
    """Creates a 2-page sample PDF covering Supervised and Unsupervised learning."""
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Unit 1: Supervised Learning Fundamentals\n\n"
        "Supervised learning is a machine learning paradigm where models are trained using labeled datasets. "
        "Each training instance consists of an input feature vector and a corresponding ground-truth target label. "
        "The model learns an optimal mapping function by minimizing error through backpropagation and loss functions.\n\n"
        "Primary Tasks & Examples:\n"
        "- Classification: Predicting discrete categorical class labels, such as email spam filtering or medical diagnosis.\n"
        "- Regression: Predicting continuous numeric values, such as house price prediction or temperature forecasting.\n\n"
        "Common Algorithms:\n"
        "Algorithms include Linear Regression, Logistic Regression, Support Vector Machines (SVM), and Decision Trees.",
    )

    p2 = doc.new_page()
    p2.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Unit 2: Unsupervised Learning & Clustering\n\n"
        "Unsupervised learning deals with unlabeled input data where no supervisor, teacher, or target output is available. "
        "The system autonomously uncovers intrinsic geometric patterns, cluster groupings, or low-dimensional representations.\n\n"
        "Primary Tasks & Examples:\n"
        "- Clustering: Grouping similar instances without prior labels, such as customer market segmentation.\n"
        "- Dimensionality Reduction: Compressing high-dimensional feature spaces, such as Principal Component Analysis (PCA).\n\n"
        "Common Algorithms:\n"
        "Algorithms include K-Means Clustering, Hierarchical Clustering, DBSCAN, and PCA.\n\n"
        "Related Concepts:\n"
        "Semi-Supervised Learning bridges the gap by leveraging a small amount of labeled data with abundant unlabeled data.",
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestComparisonDetection(unittest.TestCase):
    """Tests intelligent detection of comparison vs ordinary questions."""

    def test_comparison_questions_detection(self):
        comparison_queries = [
            "Differentiate between supervised and unsupervised learning",
            "Differentiate supervised and unsupervised learning",
            "What is the difference between classification and regression?",
            "What are the differences between supervised and unsupervised learning",
            "Tell me the difference between CNN and RNN",
            "Compare supervised and unsupervised learning",
            "Compare and contrast BFS and DFS",
            "Comparison of supervised and unsupervised learning",
            "Comparison between precision and recall",
            "Distinguish between precision and recall",
            "Distinguish overfitting from underfitting",
            "Supervised Learning vs Unsupervised Learning",
            "Supervised learning vs. unsupervised learning",
            "Supervised learning versus unsupervised learning",
            "How does supervised learning differ from unsupervised learning?",
            "How is classification different from regression?",
            "Contrast supervised learning with unsupervised learning",
        ]

        for q in comparison_queries:
            with self.subTest(query=q):
                self.assertTrue(
                    is_comparison_question(q),
                    f"Expected '{q}' to be classified as a comparison question.",
                )

    def test_ordinary_questions_not_detected_as_comparison(self):
        ordinary_queries = [
            "What is supervised learning?",
            "Define classification.",
            "Explain linear regression in simple words.",
            "Give me a detailed explanation of this document.",
            "Summarize this document briefly.",
            "What are the main topics?",
            "Give important points.",
            "How many pages are in this PDF?",
            "How many total chunks?",
            "Who is the mentor for this project?",
            "What is the conclusion of the research paper?",
            "List the advantages of Python.",
        ]

        for q in ordinary_queries:
            with self.subTest(query=q):
                self.assertFalse(
                    is_comparison_question(q),
                    f"Expected '{q}' NOT to be classified as a comparison question.",
                )

    def test_extract_comparison_topics(self):
        topics = extract_comparison_topics("Differentiate between Supervised Learning and Unsupervised Learning")
        self.assertEqual(len(topics), 2)
        self.assertEqual(topics[0].lower(), "supervised learning")
        self.assertEqual(topics[1].lower(), "unsupervised learning")

        topics2 = extract_comparison_topics("How does CNN differ from RNN?")
        self.assertEqual(len(topics2), 2)
        self.assertEqual(topics2[0].lower(), "cnn")
        self.assertEqual(topics2[1].lower(), "rnn")

        topics3 = extract_comparison_topics("Classification vs Regression")
        self.assertEqual(len(topics3), 2)
        self.assertEqual(topics3[0].lower(), "classification")
        self.assertEqual(topics3[1].lower(), "regression")


class TestComparisonFormatting(unittest.TestCase):
    """Tests the format_comparison_answer standardizer and formatter."""

    def test_format_comparison_answer_with_all_sections(self):
        raw = """
        Here is the comparison:
        ### Introduction
        Supervised learning uses labeled data whereas unsupervised learning analyzes unlabeled data.

        ### Comparison Table
        | Basis | Supervised | Unsupervised |
        | Training Data | Labeled input-output | Unlabeled data |
        | Core Goal | Predict targets | Discover clusters |

        ### Important Points
        - Supervised models optimize via loss functions.
        - Unsupervised models evaluate cluster cohesion.

        ### Examples from Document
        - Supervised: Spam detection (Page 1)
        - Unsupervised: Customer segmentation (Page 2)

        ### Related Concepts
        - Semi-Supervised Learning (Page 2)

        ### Conclusion
        Supervised learning is suited for prediction while unsupervised learning is for pattern discovery.
        """

        formatted = format_comparison_answer(
            raw,
            question="Differentiate between Supervised and Unsupervised Learning",
        )

        self.assertNotIn("Here is the comparison:", formatted)
        self.assertIn("### Introduction", formatted)
        self.assertIn("### Comparison Table", formatted)
        self.assertIn("| :--- | :--- | :--- |", formatted)
        self.assertIn("### Important Points", formatted)
        self.assertIn("### Examples from Document", formatted)
        self.assertIn("### Related Concepts", formatted)
        self.assertIn("### Conclusion", formatted)
        self.assertIn("Spam detection", formatted)
        self.assertIn("Customer segmentation", formatted)

    def test_omits_related_concepts_when_none_mentioned(self):
        raw = """
        ### Introduction
        Supervised and unsupervised learning differ in their training data.

        ### Comparison Table
        | Basis | Supervised | Unsupervised |
        | Data | Labeled | Unlabeled |

        ### Important Points
        - Supervised maps inputs to outputs.

        ### Examples from Document
        - Supervised: Email spam filter

        ### Related Concepts
        None mentioned in document.

        ### Conclusion
        Both paradigms serve distinct roles.
        """

        formatted = format_comparison_answer(
            raw,
            question="Compare Supervised and Unsupervised Learning",
        )

        self.assertNotIn("### Related Concepts", formatted)
        self.assertNotIn("None mentioned in document", formatted)
        self.assertIn("### Conclusion", formatted)

    def test_synthesizes_table_if_missing(self):
        raw = """
        Supervised learning uses labeled inputs while unsupervised learning finds patterns without labels.

        Differences:
        - Input Data: Supervised uses labeled data vs Unsupervised uses unlabeled data
        - Goal: Supervised predicts target labels vs Unsupervised groups data
        - Supervision: Supervised has external supervision vs Unsupervised is self-organizing

        Important Points:
        - Supervised models are evaluated against ground truth.
        - Unsupervised models are evaluated using clustering metrics.

        Examples from Document:
        - Supervised: Classification (Page 1)
        - Unsupervised: Clustering (Page 2)

        Conclusion:
        Supervised is for prediction, unsupervised is for discovery.
        """

        formatted = format_comparison_answer(
            raw,
            question="Differentiate between Supervised Learning and Unsupervised Learning",
        )

        self.assertIn("### Comparison Table", formatted)
        self.assertIn("| Basis of Comparison |", formatted)
        self.assertIn("| :--- |", formatted)
        self.assertIn("### Introduction", formatted)
        self.assertIn("### Conclusion", formatted)


class TestComparisonIntegration(unittest.TestCase):
    """Integration tests verifying RAG answer formatting in Chat and Important Questions."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_comparison_col"

        pdf_bytes = create_sample_ml_pdf()
        total_pages, chunks = extract_and_chunk_pdf(pdf_bytes, "ml_guide.pdf")
        cls.doc_summary = index_document(
            filename="ml_guide.pdf",
            total_pages=total_pages,
            chunks=chunks,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.doc_id = cls.doc_summary["document_id"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_chat_comparison_question_generates_structured_answer(self):
        mock_raw_answer = """
        ### Introduction
        Supervised learning uses labeled data whereas unsupervised learning works with unlabeled data.

        ### Comparison Table
        | Basis of Comparison | Supervised Learning | Unsupervised Learning |
        | Training Data | Labeled input-output pairs | Unlabeled raw data |
        | Goal | Predict discrete or continuous targets | Discover hidden groupings |
        | Feedback | Direct feedback via loss function | No external feedback |

        ### Important Points
        - Supervised learning learns a functional mapping between inputs and outputs.
        - Unsupervised learning relies on geometric distance metrics like Euclidean distance.

        ### Examples from Document
        - Supervised: Email spam filtering and house price prediction (Page 1).
        - Unsupervised: Customer market segmentation and PCA (Page 2).

        ### Related Concepts
        - Semi-Supervised Learning leverages labeled and unlabeled data together (Page 2).

        ### Conclusion
        Supervised learning is chosen when ground-truth labels are available, whereas unsupervised learning is ideal for initial exploratory analysis.
        """

        with patch("backend.qa_service.generate_rag_answer", return_value=mock_raw_answer):
            res = answer_study_question(
                question="Differentiate between supervised and unsupervised learning",
                top_k=4,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
                document_id=self.doc_id,
            )

            answer = res["answer"]
            self.assertIn("### Introduction", answer)
            self.assertIn("### Comparison Table", answer)
            self.assertIn("| :--- | :--- | :--- |", answer)
            self.assertIn("### Important Points", answer)
            self.assertIn("### Examples from Document", answer)
            self.assertIn("### Conclusion", answer)
            self.assertGreaterEqual(len(res["sources"]), 1)
            # Ensure citations are preserved
            page_numbers = [s["page_number"] for s in res["sources"]]
            self.assertTrue(1 in page_numbers or 2 in page_numbers)

    def test_chat_ordinary_question_does_not_force_comparison_table(self):
        mock_raw_answer = (
            "Supervised learning is a machine learning paradigm where models are trained using labeled datasets. "
            "Each training instance consists of input features and ground-truth target labels."
        )

        with patch("backend.qa_service.generate_rag_answer", return_value=mock_raw_answer):
            res = answer_study_question(
                question="What is supervised learning?",
                top_k=4,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
                document_id=self.doc_id,
            )

            answer = res["answer"]
            self.assertNotIn("### Comparison Table", answer)
            self.assertIn("labeled datasets", answer)

    def test_important_questions_comparison_generates_structured_answer(self):
        mock_raw_answer = """
        ### Introduction
        Classification predicts categorical labels while regression predicts continuous numbers.

        ### Comparison Table
        | Basis of Comparison | Classification | Regression |
        | Target Output | Discrete categorical labels | Continuous numerical values |
        | Example Task | Email spam filtering | House price prediction |

        ### Important Points
        - Classification models assign instances to discrete classes.
        - Regression models estimate mathematical relationships on a continuous scale.

        ### Examples from Document
        - Classification: Email spam filtering (Page 1)
        - Regression: House price prediction (Page 1)

        ### Conclusion
        Use classification for category identification and regression for numeric forecasting in exams.
        """

        with patch("backend.question_generator.generate_rag_answer", return_value=mock_raw_answer):
            res = generate_question_answer(
                document_id=self.doc_id,
                question="What is the difference between classification and regression?",
                question_type="long",
                page_numbers=[1],
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
            )

            answer = res["answer"]
            self.assertIn("### Introduction", answer)
            self.assertIn("### Comparison Table", answer)
            self.assertIn("| :--- | :--- | :--- |", answer)
            self.assertIn("### Important Points", answer)
            self.assertIn("### Examples from Document", answer)
            self.assertIn("### Conclusion", answer)
            self.assertEqual(res["page_reference"], "Page 1")


if __name__ == "__main__":
    unittest.main()
