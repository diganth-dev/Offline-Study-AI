"""Unit and integration tests for Short Summary and Brevity Response Formatting in Offline Study AI.

Verifies:
1. Detection and classification of brevity and summary instructions:
   - 'Give me a short description of this PDF.'
   - 'Summarize this document briefly.'
   - 'Explain this PDF in simple words.'
   - 'Give me a quick overview.'
   - 'What is this PDF about?'
2. Detection and classification of detailed explanation instructions:
   - 'Give me a detailed explanation of this PDF.'
   - 'Explain this document in detail.'
3. Distinction from metadata queries (e.g., page counts) and factual queries (e.g., mentor, department).
4. Response formatting via format_short_description_answer:
   - Mandatory '**Short Description**' header.
   - 4 to 6 concise, student-friendly bullet points (- ).
   - Each point 1-2 short sentences.
   - Stripping of conversational chatter, preambles, and postambles.
   - Converting plain unstructured paragraphs into bullet points.
5. Preservation of source page citations and document isolation.
6. Support for detailed explanations without forcing brevity or bullet points.
7. End-to-end integration with answer_study_question.
"""

import shutil
import tempfile
import unittest
from unittest.mock import patch

import pymupdf as fitz

from backend.metadata_service import (
    is_short_summary_request,
    is_detailed_explanation_request,
    classify_contextual_instruction,
    get_instruction_guidance,
)
from backend.pdf_parser import extract_and_chunk_pdf
from backend.qa_service import (
    format_short_description_answer,
    answer_study_question,
)
from backend.vector_store import index_document


def create_sample_ai_pdf() -> bytes:
    """Creates a sample multi-page study guide PDF on AI fundamentals."""
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Foundations of Artificial Intelligence: A Study Guide\n\n"
        "1. Document Overview and Purpose\n"
        "The purpose of this guide is to introduce undergraduate computer science students to the fundamental "
        "concepts, historical milestones, and modern paradigms of artificial intelligence. It serves as a concise "
        "revision resource for semester exams.\n\n"
        "2. Key Objectives\n"
        "- Establish a clear understanding of symbolic AI vs statistical machine learning.\n"
        "- Learn the mathematical foundations of gradient descent and neural network training.\n"
        "- Evaluate ethics and practical deployment challenges in production AI systems.",
        fontsize=11,
    )

    p2 = doc.new_page()
    p2.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "3. Core Machine Learning Paradigms\n\n"
        "Machine learning is categorized into three primary branches:\n"
        "- Supervised Learning: Models learn input-output mappings using labeled training examples (e.g., regression, classification).\n"
        "- Unsupervised Learning: Uncovers latent patterns and cluster structures without explicit labels (e.g., k-means, PCA).\n"
        "- Reinforcement Learning: Agents interact with environments to maximize cumulative rewards through trial and error.\n\n"
        "4. Conclusion & Learning Outcomes\n"
        "Mastering these core principles allows students to transition smoothly into advanced deep learning frameworks. "
        "Understanding algorithmic trade-offs ensures effective model selection for real-world tasks.",
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_sample_biology_pdf() -> bytes:
    """Creates a separate domain PDF (Cell Biology) to test document isolation."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Cell Biology and Cellular Respiration\n\n"
        "Overview: This laboratory guide details mitochondria and cellular respiration pathways. "
        "Glycolysis occurs in the cytoplasm yielding pyruvate and ATP.",
        fontsize=11,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestShortSummaryClassification(unittest.TestCase):
    """Verifies that query classification accurately identifies summary and brevity instructions."""

    def test_required_short_summary_questions(self):
        required_questions = [
            "Give me a short description of this PDF.",
            "Summarize this document briefly.",
            "Explain this PDF in simple words.",
            "Give me a quick overview.",
            "What is this PDF about?",
        ]
        for q in required_questions:
            self.assertTrue(
                is_short_summary_request(q),
                f"Expected is_short_summary_request to be True for: '{q}'",
            )
            self.assertEqual(
                classify_contextual_instruction(q),
                "short_summary",
                f"Expected classify_contextual_instruction to return 'short_summary' for: '{q}'",
            )

    def test_additional_brevity_variations(self):
        variations = [
            "short description",
            "give me a short description",
            "brief overview of this document",
            "explain this document simply",
            "can you briefly explain this file?",
            "summarize this briefly",
            "what is this about?",
            "give a quick overview of the file",
        ]
        for q in variations:
            self.assertTrue(
                is_short_summary_request(q),
                f"Failed detection for variation: '{q}'",
            )

    def test_detailed_explanation_detection(self):
        detailed_questions = [
            "Give me a detailed explanation of this PDF.",
            "Explain this document in detail.",
            "Can you provide a comprehensive in-depth explanation of this file?",
            "Explain this PDF thoroughly and in detail.",
        ]
        for q in detailed_questions:
            self.assertTrue(
                is_detailed_explanation_request(q),
                f"Expected is_detailed_explanation_request True for: '{q}'",
            )
            self.assertFalse(
                is_short_summary_request(q),
                f"Detailed explanation should NOT be treated as short summary: '{q}'",
            )
            self.assertEqual(
                classify_contextual_instruction(q),
                "detailed_explanation",
                f"Expected classify_contextual_instruction 'detailed_explanation' for: '{q}'",
            )

    def test_metadata_queries_not_classified_as_summary(self):
        metadata_queries = [
            "How many pages are in this PDF?",
            "Total number of pages",
            "How many chunks were extracted?",
            "List all uploaded documents",
            "What files do I have?",
        ]
        for q in metadata_queries:
            self.assertIsNone(
                classify_contextual_instruction(q),
                f"Metadata question should not be classified as contextual instruction: '{q}'",
            )

    def test_factual_entity_queries_not_classified_as_summary(self):
        factual_queries = [
            "What is the department name?",
            "Who is the mentor?",
            "Student name",
            "College name",
            "Contact details",
        ]
        for q in factual_queries:
            self.assertIsNone(
                classify_contextual_instruction(q),
                f"Factual query should not be classified as contextual instruction: '{q}'",
            )


class TestShortSummaryFormatter(unittest.TestCase):
    """Verifies that format_short_description_answer strictly enforces the required formatting."""

    def test_already_formatted_markdown(self):
        raw = (
            "**Short Description**\n"
            "- Main purpose is to introduce artificial intelligence concepts to students.\n"
            "- Covers symbolic AI and modern machine learning techniques.\n"
            "- Focuses on supervised, unsupervised, and reinforcement learning.\n"
            "- Prepares students for practical deep learning implementations."
        )
        formatted = format_short_description_answer(raw)
        self.assertTrue(formatted.startswith("**Short Description**"))
        lines = [line for line in formatted.split("\n") if line.strip()]
        self.assertEqual(lines[0], "**Short Description**")
        self.assertEqual(len(lines), 5)  # Header + 4 bullets
        for bullet in lines[1:]:
            self.assertTrue(bullet.startswith("- "))

    def test_converts_raw_paragraph_into_points(self):
        paragraph = (
            "This study guide covers the fundamentals of artificial intelligence for undergraduate students. "
            "It discusses the historical evolution from symbolic reasoning to modern statistical machine learning. "
            "Key topics include supervised, unsupervised, and reinforcement learning paradigms. "
            "Finally, the guide emphasizes practical ethical considerations and deployment challenges."
        )
        formatted = format_short_description_answer(paragraph)
        self.assertTrue(formatted.startswith("**Short Description**"))
        lines = [line for line in formatted.split("\n") if line.strip()]
        self.assertGreaterEqual(len(lines), 4)
        for bullet in lines[1:]:
            self.assertTrue(bullet.startswith("- "))

    def test_strips_conversational_chatter_and_preambles(self):
        noisy_output = (
            "Sure! Here is a short description of the document:\n\n"
            "### Short Description:\n"
            "* Main purpose of the guide is foundational AI education.\n"
            "* Important topics include supervised and unsupervised learning.\n"
            "* Key objectives focus on mathematical foundations and algorithmic trade-offs.\n"
            "* Conclusion highlights readiness for advanced machine learning.\n\n"
            "Hope this helps with your studies!"
        )
        formatted = format_short_description_answer(noisy_output)
        self.assertTrue(formatted.startswith("**Short Description**"))
        self.assertNotIn("Sure!", formatted)
        self.assertNotIn("Here is a short description", formatted)
        self.assertNotIn("Hope this helps", formatted)
        lines = [line for line in formatted.split("\n") if line.strip()]
        for bullet in lines[1:]:
            self.assertTrue(bullet.startswith("- "))

    def test_bullet_points_conciseness(self):
        raw = (
            "**Short Description**\n"
            "- Main purpose: Introduces core AI concepts clearly.\n"
            "- Important topics: Machine learning paradigms and neural network fundamentals.\n"
            "- Key objectives: Mathematical foundations and model trade-offs.\n"
            "- Outcome: Prepares students for semester exams and advanced study."
        )
        formatted = format_short_description_answer(raw)
        bullets = [line[2:].strip() for line in formatted.split("\n") if line.startswith("- ")]
        self.assertTrue(4 <= len(bullets) <= 6)
        for bullet in bullets:
            # Bullet point should be 1-2 short sentences (typically < 200 characters)
            self.assertLess(len(bullet), 250)


class TestGuidancePromptGeneration(unittest.TestCase):
    """Verifies the system prompts and guidance messages passed to the LLM."""

    def test_short_summary_guidance_rules(self):
        guidance = get_instruction_guidance("short_summary", "test_guide.pdf")
        self.assertIn("**Short Description**", guidance)
        self.assertIn("Main Purpose", guidance)
        self.assertIn("Key Topics", guidance)
        self.assertIn("Key Objectives", guidance)
        self.assertIn("Conclusion", guidance)
        self.assertIn("simple english", guidance.lower())
        self.assertIn("4 to 6", guidance)

    def test_detailed_explanation_guidance_rules(self):
        guidance = get_instruction_guidance("detailed_explanation", "test_guide.pdf")
        self.assertIn("comprehensive, detailed, and in-depth", guidance)
        self.assertNotIn("**Short Description**", guidance)


class TestShortSummaryWorkflowIntegration(unittest.TestCase):
    """Integration tests verifying RAG answer generation for short summaries and detailed explanations."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_summary_formatting_col"

        # 1. Index AI study guide PDF (2 pages)
        ai_bytes = create_sample_ai_pdf()
        p1, chunks1 = extract_and_chunk_pdf(ai_bytes, "ai_study_guide.pdf")
        res1 = index_document(
            filename="ai_study_guide.pdf",
            total_pages=p1,
            chunks=chunks1,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.ai_doc_id = res1["document_id"]

        # 2. Index Cell Biology PDF (1 page) for document isolation testing
        bio_bytes = create_sample_biology_pdf()
        p2, chunks2 = extract_and_chunk_pdf(bio_bytes, "cell_biology.pdf")
        res2 = index_document(
            filename="cell_biology.pdf",
            total_pages=p2,
            chunks=chunks2,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.bio_doc_id = res2["document_id"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_short_description_with_mocked_llm(self):
        mock_raw_answer = (
            "Here is the summary of the document:\n\n"
            "**Short Description**\n"
            "- **Main Purpose:** Introduces foundational artificial intelligence concepts for students.\n"
            "- **Key Topics:** Covers symbolic AI, supervised learning, and reinforcement learning.\n"
            "- **Key Objectives:** Focuses on mathematical training concepts and model trade-offs.\n"
            "- **Conclusion:** Prepares learners for semester exams and modern AI applications."
        )

        with patch("backend.qa_service.generate_rag_answer", return_value=mock_raw_answer):
            for question in [
                "Give me a short description of this PDF.",
                "Summarize this document briefly.",
                "Explain this PDF in simple words.",
                "Give me a quick overview.",
                "What is this PDF about?",
            ]:
                res = answer_study_question(
                    question=question,
                    document_id=self.ai_doc_id,
                    collection_name=self.collection_name,
                    persist_directory=self.test_dir,
                )

                self.assertTrue(
                    res["answer"].startswith("**Short Description**"),
                    f"Answer did not start with '**Short Description**' for '{question}'",
                )
                self.assertNotIn("Here is the summary", res["answer"])
                bullets = [line for line in res["answer"].split("\n") if line.startswith("- ")]
                self.assertEqual(len(bullets), 4)

                # Source page citations must be preserved
                self.assertGreaterEqual(len(res["sources"]), 1)
                for src in res["sources"]:
                    self.assertEqual(src["filename"], "ai_study_guide.pdf")
                    self.assertIn(src["page_number"], [1, 2])

                # Document ID must be preserved
                self.assertEqual(res.get("document_id"), self.ai_doc_id)

    def test_detailed_explanation_provides_more_detail(self):
        mock_detailed_answer = (
            "### Detailed Explanation of Artificial Intelligence Fundamentals\n\n"
            "This document provides an exhaustive, multi-part curriculum for undergraduate students. "
            "It begins by examining symbolic AI, where knowledge is represented using formal logic and inference engines. "
            "Next, it explores statistical machine learning models across supervised, unsupervised, and reinforcement paradigms.\n\n"
            "In supervised learning, models map inputs to targets using loss optimization techniques like gradient descent. "
            "Reinforcement learning operates via policy iteration and reward maximization. "
            "Finally, the guide analyzes real-world deployment challenges including bias, fairness, and computational complexity."
        )

        with patch("backend.qa_service.generate_rag_answer", return_value=mock_detailed_answer):
            res = answer_study_question(
                question="Give me a detailed explanation of this PDF.",
                document_id=self.ai_doc_id,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
            )

            # Detailed answers should NOT be forced into the Short Description bullet-point template
            self.assertFalse(res["answer"].startswith("**Short Description**"))
            self.assertIn("Detailed Explanation", res["answer"])
            self.assertIn("gradient descent", res["answer"])
            self.assertGreaterEqual(len(res["sources"]), 1)

    def test_document_isolation_for_short_summary(self):
        mock_ai_summary = (
            "**Short Description**\n"
            "- Explains artificial intelligence fundamentals and machine learning algorithms.\n"
            "- Discusses supervised, unsupervised, and reinforcement learning.\n"
            "- Covers historical milestones and gradient descent foundations.\n"
            "- Concludes with preparation for advanced deep learning topics."
        )

        with patch("backend.qa_service.generate_rag_answer", return_value=mock_ai_summary):
            res_ai = answer_study_question(
                question="Give me a short description of this PDF.",
                document_id=self.ai_doc_id,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
            )
            # Must not cite cell biology
            for src in res_ai["sources"]:
                self.assertEqual(src["filename"], "ai_study_guide.pdf")

            # Context chunks used must come strictly from ai_study_guide.pdf
            for chunk in res_ai["context_used"]:
                self.assertEqual(chunk["filename"], "ai_study_guide.pdf")


if __name__ == "__main__":
    unittest.main()
