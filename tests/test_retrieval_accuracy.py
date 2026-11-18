"""Unit and integration tests for RAG retrieval accuracy and hybrid search in Offline Study AI.

Verifies:
1. Hybrid retrieval combining semantic vector search and BM25 lexical matching.
2. Department name retrieval with various phrasing (e.g., 'Department', 'Department name', 'What is the department name?').
3. Mentor name retrieval with various phrasing (e.g., 'Who is the mentor?', 'Mentor name', 'Faculty guide').
4. Student name retrieval with various phrasing (e.g., 'Student name', 'Who are the students?', 'List team members').
5. College/Institute name retrieval with various phrasing (e.g., 'College name', 'Institute name', 'What is the college?').
6. Contact details retrieval (e.g., 'Contact details', 'Email and phone').
7. Anti-hallucination protection: genuinely unavailable information returns clear not-found notification.
8. Strict document isolation: scoped search never retrieves chunks or answers from another PDF.
9. Source page citation preservation.
10. Dynamic performance with different uploaded PDFs without hardcoding.
"""

import shutil
import tempfile
import unittest
import pymupdf as fitz

from backend.hybrid_retrieval import (
    tokenize,
    compute_lexical_score,
    combine_hybrid_scores,
    is_short_or_factual_query,
    expand_query_terms,
)
from backend.pdf_parser import extract_and_chunk_pdf
from backend.qa_service import answer_study_question
from backend.vector_store import (
    index_document,
    search_documents,
    get_document_chunks,
    get_indexed_documents,
)


def create_docket_test_pdf() -> bytes:
    """Generates an in-memory registration docket PDF with realistic administrative fields."""
    doc = fitz.open()

    # Page 1: Partner Institute & Project Overview
    p1 = doc.new_page()
    p1.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "PALS Innovation Challenge 2026\n"
        "Official Project Registration Docket\n\n"
        "1. PARTNER INSTITUTE INFORMATION\n"
        "Institute Name\n"
        "APEX INSTITUTE OF AERONAUTICAL ENGINEERING & TECHNOLOGY\n"
        "Short Code: AIAET\n"
        "Regional Cluster: South Zone\n\n"
        "2. PROJECT SPECIFICATIONS\n"
        "Project Name: Autonomous Drone Pathfinding Engine\n"
        "Domain: Robotics and Machine Learning\n"
        "Executive Summary: Developing an obstacle-avoidance AI pipeline for autonomous quadcopters.",
        fontsize=11,
    )

    # Page 2: Faculty Mentor & Student Team Members
    p2 = doc.new_page()
    p2.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "3. FACULTY MENTOR INFORMATION\n"
        "Mentor Name\n"
        "Dr. Rajesh Kumar\n"
        "Designation\n"
        "Associate Professor & HOD\n"
        "Department\n"
        "DEPARTMENT OF AEROSPACE AND AVIONICS ENGINEERING\n"
        "Contact Details\n"
        "rajesh.kumar@apexavionics.edu | +91-9845012345\n\n"
        "4. STUDENT TEAM MEMBERS\n"
        "Seq  Student Name  Roll / ID      Department                         Year\n"
        "#1   Arjun Rao     AERO-2024-001  AEROSPACE AND AVIONICS ENGINEERING  4th\n"
        "#2   Sneha Nair    AERO-2024-015  AEROSPACE AND AVIONICS ENGINEERING  4th\n"
        "#3   Kiran Patel   AERO-2024-032  AEROSPACE AND AVIONICS ENGINEERING  4th\n\n"
        "PROJECT TEAM UNDERTAKING\n"
        "We verify that the information regarding our mentor Dr. Rajesh Kumar and students is true.\n"
        "Arjun Rao (Lead) Signature | Dr. Rajesh Kumar (Faculty Mentor) Signature",
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_second_test_pdf() -> bytes:
    """Generates a separate domain PDF (Biochemistry) to test strict document isolation."""
    doc = fitz.open()
    p = doc.new_page()
    p.insert_textbox(
        fitz.Rect(50, 50, 550, 750),
        "Chapter 1: Principles of Enzymatic Catalysis\n"
        "Enzymes are biological catalysts that increase the rate of chemical reactions. "
        "The active site binds the substrate forming an enzyme-substrate complex. "
        "Michaelis-Menten kinetics describe the relationship between reaction rate and substrate concentration.",
        fontsize=11,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestHybridRetrievalComponents(unittest.TestCase):
    """Unit tests for the underlying hybrid retrieval algorithms and heuristics."""

    def test_short_query_detection(self):
        self.assertTrue(is_short_or_factual_query("Department"))
        self.assertTrue(is_short_or_factual_query("Department name"))
        self.assertTrue(is_short_or_factual_query("What is the department name?"))
        self.assertTrue(is_short_or_factual_query("Mentor name"))
        self.assertTrue(is_short_or_factual_query("Who is the mentor?"))
        self.assertTrue(is_short_or_factual_query("Student name"))
        self.assertTrue(is_short_or_factual_query("College name"))
        self.assertTrue(is_short_or_factual_query("Contact details"))
        self.assertTrue(is_short_or_factual_query("Which institute is this?"))

        # Complex conceptual essay query should not be flagged as a short terse query
        long_query = "Can you thoroughly discuss how gradient descent updates the weights across successive training epochs?"
        self.assertFalse(is_short_or_factual_query(long_query))

    def test_query_synonym_expansion(self):
        expanded = expand_query_terms(["college", "name"])
        self.assertIn("college", expanded)
        self.assertIn("institute", expanded)
        self.assertIn("university", expanded)

        expanded_mentor = expand_query_terms(["mentor"])
        self.assertIn("guide", expanded_mentor)
        self.assertIn("advisor", expanded_mentor)
        self.assertIn("faculty", expanded_mentor)

    def test_lexical_scoring_ranks_exact_matches_at_top(self):
        chunks = [
            {"chunk_number": 1, "text": "General introduction to aerial robotics and drones."},
            {"chunk_number": 2, "text": "Department\nDEPARTMENT OF AEROSPACE AND AVIONICS ENGINEERING\nMentor Name: Dr. Rajesh Kumar"},
        ]
        scores = compute_lexical_score("Department name", chunks)
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], 4.0)
        self.assertEqual(scores[0], 0.0)

    def test_combine_hybrid_scores_offtopic_preservation(self):
        # When lexical score is 0 and dense similarity is low, score must remain strictly below 0.20
        combined = combine_hybrid_scores(dense_similarity=0.08, lexical_score=0.0)
        self.assertLess(combined, 0.20)
        self.assertEqual(combined, 0.08)

    def test_combine_hybrid_scores_factual_boost(self):
        # When dense similarity is low (0.11) but lexical match is strong (6.0), score must surpass 0.20
        combined = combine_hybrid_scores(dense_similarity=0.11, lexical_score=6.0)
        self.assertGreater(combined, 0.35)


class TestRetrievalAccuracyWorkflow(unittest.TestCase):
    """Integration tests verifying retrieval accuracy on realistic documents."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.collection_name = "test_retrieval_acc_col"

        # 1. Index Docket PDF (2 pages)
        docket_bytes = create_docket_test_pdf()
        p1, chunks1 = extract_and_chunk_pdf(docket_bytes, "apex_docket.pdf")
        res1 = index_document(
            filename="apex_docket.pdf",
            total_pages=p1,
            chunks=chunks1,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.docket_id = res1["document_id"]

        # 2. Index Biochemistry PDF (1 page)
        bio_bytes = create_second_test_pdf()
        p2, chunks2 = extract_and_chunk_pdf(bio_bytes, "biochem_enzymes.pdf")
        res2 = index_document(
            filename="biochem_enzymes.pdf",
            total_pages=p2,
            chunks=chunks2,
            collection_name=cls.collection_name,
            persist_directory=cls.test_dir,
        )
        cls.bio_id = res2["document_id"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. Department Name Retrieval Tests
    # -------------------------------------------------------------

    def test_department_name_retrieval(self):
        for q in ["Department", "Department name", "What is the department name?", "Which department?"]:
            res = search_documents(
                query=q,
                top_k=2,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
                document_id=self.docket_id,
            )
            self.assertGreaterEqual(len(res), 1, f"Failed for query: {q}")
            top = res[0]
            self.assertIn("AEROSPACE AND AVIONICS", top["text"], f"Department chunk not top for query: {q}")
            self.assertGreaterEqual(top["similarity_score"], 0.20, f"Score below threshold for query: {q}")

    # -------------------------------------------------------------
    # 2. Mentor Name Retrieval Tests
    # -------------------------------------------------------------

    def test_mentor_name_retrieval(self):
        for q in ["Mentor name", "Who is the mentor?", "Faculty guide", "Mentor"]:
            res = search_documents(
                query=q,
                top_k=2,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
                document_id=self.docket_id,
            )
            self.assertGreaterEqual(len(res), 1, f"Failed for query: {q}")
            top = res[0]
            self.assertIn("Rajesh Kumar", top["text"], f"Mentor chunk not top for query: {q}")
            self.assertGreaterEqual(top["similarity_score"], 0.20, f"Score below threshold for query: {q}")

    # -------------------------------------------------------------
    # 3. Student Name Retrieval Tests
    # -------------------------------------------------------------

    def test_student_name_retrieval(self):
        for q in ["Student name", "Who are the students?", "List student team members", "Student members"]:
            res = search_documents(
                query=q,
                top_k=2,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
                document_id=self.docket_id,
            )
            self.assertGreaterEqual(len(res), 1, f"Failed for query: {q}")
            top = res[0]
            self.assertIn("Arjun Rao", top["text"], f"Student team chunk not top for query: {q}")
            self.assertGreaterEqual(top["similarity_score"], 0.20, f"Score below threshold for query: {q}")

    # -------------------------------------------------------------
    # 4. College Name Retrieval Tests (using synonym expansion)
    # -------------------------------------------------------------

    def test_college_name_retrieval(self):
        for q in ["College name", "Institute name", "What is the college name?", "Which university?"]:
            res = search_documents(
                query=q,
                top_k=2,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
                document_id=self.docket_id,
            )
            self.assertGreaterEqual(len(res), 1, f"Failed for query: {q}")
            top = res[0]
            self.assertIn("APEX INSTITUTE", top["text"], f"Institute chunk not top for query: {q}")
            self.assertEqual(top["page_number"], 1)
            self.assertGreaterEqual(top["similarity_score"], 0.20, f"Score below threshold for query: {q}")

    # -------------------------------------------------------------
    # 5. Contact Details Retrieval Tests
    # -------------------------------------------------------------

    def test_contact_details_retrieval(self):
        for q in ["Contact details", "Phone and email", "Contact info"]:
            res = search_documents(
                query=q,
                top_k=2,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
                document_id=self.docket_id,
            )
            self.assertGreaterEqual(len(res), 1, f"Failed for query: {q}")
            top = res[0]
            self.assertTrue("rajesh.kumar@apexavionics.edu" in top["text"] or "9845012345" in top["text"])
            self.assertGreaterEqual(top["similarity_score"], 0.20)

    # -------------------------------------------------------------
    # 6. Genuinely Unavailable Information Tests
    # -------------------------------------------------------------

    def test_genuinely_unavailable_information_rejected(self):
        # In docket PDF, there is no principal, fee structure, or strawberry ice cream
        for q in [
            "Who is the principal?",
            "What is the annual tuition fee?",
            "How do you make strawberry ice cream at home?",
            "What is the speed of light in vacuum?",
        ]:
            res = answer_study_question(
                question=q,
                top_k=2,
                collection_name=self.collection_name,
                persist_directory=self.test_dir,
                document_id=self.docket_id,
            )
            lower_ans = res["answer"].lower()
            self.assertTrue(
                "does not contain" in lower_ans
                or "not contain" in lower_ans
                or "no information" in lower_ans
                or "not found" in lower_ans
                or "not mentioned" in lower_ans,
                f"Expected not found message for '{q}', got: {res['answer']}"
            )
            # Sources must be empty when rejected before LLM or ungrounded
            if "does not contain" in lower_ans:
                self.assertEqual(res["sources"], [])

    # -------------------------------------------------------------
    # 7. Document Isolation Tests
    # -------------------------------------------------------------

    def test_document_isolation_scoped_mode(self):
        # Querying enzyme kinetics while scoped to docket must return not found
        res_docket = answer_study_question(
            question="What is Michaelis-Menten kinetics and substrate concentration?",
            top_k=2,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.docket_id,
        )
        self.assertIn("does not contain", res_docket["answer"].lower())
        self.assertEqual(res_docket["sources"], [])

        # Querying aerospace department while scoped to biochemistry must return not found
        res_bio = answer_study_question(
            question="What is the department name?",
            top_k=2,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.bio_id,
        )
        self.assertIn("does not contain", res_bio["answer"].lower())
        self.assertEqual(res_bio["sources"], [])

    # -------------------------------------------------------------
    # 8. End-to-End QA with Local Ollama
    # -------------------------------------------------------------

    def test_qa_end_to_end_department_name(self):
        res = answer_study_question(
            question="What is the department name?",
            top_k=2,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.docket_id,
        )
        self.assertIn("aerospace", res["answer"].lower())
        self.assertGreaterEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["filename"], "apex_docket.pdf")
        page_numbers = [s["page_number"] for s in res["sources"]]
        self.assertIn(2, page_numbers)

    def test_qa_end_to_end_mentor_name(self):
        res = answer_study_question(
            question="Who is the mentor?",
            top_k=2,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.docket_id,
        )
        self.assertIn("rajesh", res["answer"].lower())
        self.assertGreaterEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["filename"], "apex_docket.pdf")
        page_numbers = [s["page_number"] for s in res["sources"]]
        self.assertIn(2, page_numbers)

    def test_qa_end_to_end_college_name(self):
        res = answer_study_question(
            question="What is the college name?",
            top_k=2,
            collection_name=self.collection_name,
            persist_directory=self.test_dir,
            document_id=self.docket_id,
        )
        self.assertIn("apex", res["answer"].lower())
        self.assertGreaterEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["filename"], "apex_docket.pdf")
        page_numbers = [s["page_number"] for s in res["sources"]]
        self.assertIn(1, page_numbers)


if __name__ == "__main__":
    unittest.main()
