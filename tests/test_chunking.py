import asyncio
import io
import unittest
from fastapi import HTTPException, UploadFile
import pymupdf as fitz

from backend.chunking import (
    chunk_text,
    chunk_pdf_pages,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from backend.main import app, upload_pdf, home, health_check


class TestChunking(unittest.TestCase):

    def test_chunk_text_empty_and_whitespace(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t  "), [])

    def test_chunk_text_smaller_than_chunk_size(self):
        text = "Hello, Offline Study AI!"
        result = chunk_text(text)
        self.assertEqual(result, [text])

    def test_chunk_text_exact_boundary(self):
        text = "x" * DEFAULT_CHUNK_SIZE
        result = chunk_text(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), DEFAULT_CHUNK_SIZE)

    def test_chunk_text_overlap(self):
        chunk_size = 1200
        chunk_overlap = 200
        # Text with 1201 characters should split into 2 chunks
        text = "a" * 1200 + "b"
        result = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "a" * 1200)
        self.assertEqual(result[1], "a" * 200 + "b")

    def test_chunk_text_sliding_window_continuity(self):
        # 2500 character pattern
        pattern = "".join(f"{i:04d}" for i in range(625))
        chunks = chunk_text(pattern, chunk_size=1200, chunk_overlap=200)
        self.assertEqual(len(chunks), 3)
        # Check overlap between consecutive chunks
        self.assertEqual(chunks[0][-200:], chunks[1][:200])
        self.assertEqual(chunks[1][-200:], chunks[2][:200])

    def test_chunk_text_validation_errors(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=-10)
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=100, chunk_overlap=-1)
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=100, chunk_overlap=150)

    def test_chunk_pdf_pages_metadata_preservation(self):
        pages = [
            (1, "Page one text content."),
            (2, "   "),  # Blank page
            (3, "Page three text: " + "K" * 1500),
        ]
        chunks = chunk_pdf_pages(pages, chunk_size=1200, chunk_overlap=200)
        self.assertEqual(len(chunks), 3)

        # Chunk 1 belongs to Page 1
        self.assertEqual(chunks[0]["chunk_number"], 1)
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertEqual(chunks[0]["text"], "Page one text content.")

        # Page 2 was blank, so Chunk 2 and 3 belong to Page 3
        self.assertEqual(chunks[1]["chunk_number"], 2)
        self.assertEqual(chunks[1]["page_number"], 3)
        self.assertTrue(chunks[1]["text"].startswith("Page three text:"))

        self.assertEqual(chunks[2]["chunk_number"], 3)
        self.assertEqual(chunks[2]["page_number"], 3)


class TestUploadPdfEndpoint(unittest.TestCase):

    def test_home_and_health(self):
        self.assertEqual(home(), {"message": "Welcome to Offline Study AI!"})
        self.assertEqual(health_check(), {"status": "healthy"})

    def test_upload_non_pdf(self):
        async def run():
            file = UploadFile(filename="notes.txt", file=io.BytesIO(b"Hello world"))
            with self.assertRaises(HTTPException) as cm:
                await upload_pdf(file)
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn("Only PDF files are allowed", cm.exception.detail)

        asyncio.run(run())

    def test_upload_empty_file(self):
        async def run():
            file = UploadFile(filename="empty.pdf", file=io.BytesIO(b""))
            with self.assertRaises(HTTPException) as cm:
                await upload_pdf(file)
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn("empty", cm.exception.detail.lower())

        asyncio.run(run())

    def test_upload_corrupt_pdf(self):
        async def run():
            file = UploadFile(filename="corrupt.pdf", file=io.BytesIO(b"garbage not pdf"))
            with self.assertRaises(HTTPException) as cm:
                await upload_pdf(file)
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn("Unable to read this PDF", cm.exception.detail)

        asyncio.run(run())

    def test_upload_blank_pdf(self):
        async def run():
            doc = fitz.open()
            doc.new_page()
            pdf_bytes = doc.tobytes()
            doc.close()

            file = UploadFile(filename="blank.pdf", file=io.BytesIO(pdf_bytes))
            with self.assertRaises(HTTPException) as cm:
                await upload_pdf(file)
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn("no extractable text", cm.exception.detail.lower())

        asyncio.run(run())

    def test_upload_valid_multipage_pdf(self):
        async def run():
            doc = fitz.open()
            p1 = doc.new_page()
            p1.insert_textbox(fitz.Rect(50, 50, 550, 750), "Overview of Artificial Intelligence.\n" * 10)
            p2 = doc.new_page()
            p2.insert_textbox(fitz.Rect(50, 50, 550, 750), "Deep Neural Networks and Backpropagation.\n" * 35)
            pdf_bytes = doc.tobytes()
            doc.close()

            file = UploadFile(filename="sample_lecture.pdf", file=io.BytesIO(pdf_bytes))
            response = await upload_pdf(file)

            self.assertEqual(response["filename"], "sample_lecture.pdf")
            self.assertEqual(response["total_pages"], 2)
            self.assertGreaterEqual(response["total_chunks"], 2)
            self.assertEqual(len(response["chunks"]), response["total_chunks"])

            for idx, chunk in enumerate(response["chunks"], start=1):
                self.assertEqual(chunk["chunk_number"], idx)
                self.assertIn(chunk["page_number"], [1, 2])
                self.assertTrue(len(chunk["text"]) > 0)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
