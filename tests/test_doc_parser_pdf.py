"""
Unit tests for PDF parser in OllaForge document parser module.

This module contains unit tests for the PDFParser class, testing
PDF text extraction and multi-page handling.

Feature: document-to-dataset
Requirements: 1.2
"""

import os
import tempfile
from pathlib import Path

import pytest

from ollaforge.doc_parser import (
    DocumentParserFactory,
    DocumentType,
    ParsedDocument,
    PDFParser,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def pdf_parser():
    """Create a PDFParser instance."""
    return PDFParser()


@pytest.fixture
def sample_pdf_path():
    """Create a simple PDF file for testing using PyPDF2."""
    try:
        from PyPDF2 import PdfWriter  # noqa: F401
        from PyPDF2.generic import NameObject, TextStringObject  # noqa: F401
    except ImportError:
        pytest.skip("PyPDF2 is required for PDF tests")

    # Create a simple PDF with text
    writer = PdfWriter()

    # Add a blank page and add text annotation (simple approach)
    # For proper text content, we need to use reportlab or similar
    # This creates a minimal valid PDF structure
    writer.add_blank_page(width=612, height=792)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
        writer.write(f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def multi_page_pdf_path():
    """Create a multi-page PDF file for testing."""
    try:
        from PyPDF2 import PdfWriter
    except ImportError:
        pytest.skip("PyPDF2 is required for PDF tests")

    writer = PdfWriter()

    # Add multiple blank pages
    for _i in range(3):
        writer.add_blank_page(width=612, height=792)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
        writer.write(f)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def pdf_with_text_path():
    """Create a PDF with actual text content using reportlab if available."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        # Fall back to creating a minimal PDF without reportlab
        pytest.skip("reportlab is required for text PDF tests")

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
        temp_path = f.name

    # Create PDF with text using reportlab
    c = canvas.Canvas(temp_path, pagesize=letter)
    c.drawString(100, 750, "Hello World")
    c.drawString(100, 700, "This is a test PDF document.")
    c.drawString(100, 650, "It contains multiple lines of text.")
    c.showPage()
    c.save()

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def multi_page_pdf_with_text_path():
    """Create a multi-page PDF with text content."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab is required for multi-page text PDF tests")

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
        temp_path = f.name

    c = canvas.Canvas(temp_path, pagesize=letter)

    # Page 1
    c.drawString(100, 750, "Page 1 Content")
    c.drawString(100, 700, "First page of the document.")
    c.showPage()

    # Page 2
    c.drawString(100, 750, "Page 2 Content")
    c.drawString(100, 700, "Second page of the document.")
    c.showPage()

    # Page 3
    c.drawString(100, 750, "Page 3 Content")
    c.drawString(100, 700, "Third page of the document.")
    c.showPage()

    c.save()

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# Unit Tests for PDFParser
# ============================================================================


class TestPDFParserBasics:
    """Basic tests for PDFParser functionality."""

    def test_supports_pdf_extension(self, pdf_parser):
        """Test that PDFParser supports .pdf extension."""
        assert pdf_parser.supports("document.pdf") is True
        assert pdf_parser.supports("path/to/file.pdf") is True
        assert pdf_parser.supports("FILE.PDF") is True  # Case insensitive

    def test_does_not_support_other_extensions(self, pdf_parser):
        """Test that PDFParser does not support non-PDF extensions."""
        assert pdf_parser.supports("document.txt") is False
        assert pdf_parser.supports("document.md") is False
        assert pdf_parser.supports("document.html") is False
        assert pdf_parser.supports("document.docx") is False

    def test_factory_returns_pdf_parser(self):
        """Test that DocumentParserFactory returns PDFParser for .pdf files."""
        parser = DocumentParserFactory.get_parser("test.pdf")
        assert isinstance(parser, PDFParser)

    def test_pdf_in_supported_formats(self):
        """Test that .pdf is in the list of supported formats."""
        supported = DocumentParserFactory.get_supported_formats()
        assert ".pdf" in supported


class TestPDFParserParsing:
    """Tests for PDF parsing functionality."""

    def test_parse_returns_parsed_document(self, pdf_parser, sample_pdf_path):
        """Test that parsing returns a ParsedDocument instance."""
        doc = pdf_parser.parse(sample_pdf_path)

        assert isinstance(doc, ParsedDocument)
        assert doc.doc_type == DocumentType.PDF
        assert doc.source_path == str(Path(sample_pdf_path).absolute())

    def test_parse_extracts_metadata(self, pdf_parser, sample_pdf_path):
        """Test that parsing extracts PDF metadata."""
        doc = pdf_parser.parse(sample_pdf_path)

        assert isinstance(doc.metadata, dict)
        assert "page_count" in doc.metadata
        assert doc.metadata["page_count"] >= 1

    def test_parse_creates_sections(self, pdf_parser, sample_pdf_path):
        """Test that parsing creates document sections."""
        doc = pdf_parser.parse(sample_pdf_path)

        assert isinstance(doc.sections, list)
        assert len(doc.sections) >= 1


class TestPDFParserMultiPage:
    """Tests for multi-page PDF handling."""

    def test_parse_multi_page_pdf(self, pdf_parser, multi_page_pdf_path):
        """Test parsing a multi-page PDF."""
        doc = pdf_parser.parse(multi_page_pdf_path)

        assert doc.metadata["page_count"] == 3

    def test_multi_page_creates_sections_per_page(
        self, pdf_parser, multi_page_pdf_with_text_path
    ):
        """Test that multi-page PDFs create sections for each page with content."""
        doc = pdf_parser.parse(multi_page_pdf_with_text_path)

        # Should have sections for pages with content
        assert len(doc.sections) >= 1
        assert doc.metadata["page_count"] == 3


class TestPDFParserTextExtraction:
    """Tests for PDF text extraction."""

    def test_extract_text_content(self, pdf_parser, pdf_with_text_path):
        """Test that text content is extracted from PDF."""
        doc = pdf_parser.parse(pdf_with_text_path)

        assert "Hello World" in doc.content
        assert "test PDF document" in doc.content

    def test_extract_multi_page_text(self, pdf_parser, multi_page_pdf_with_text_path):
        """Test that text is extracted from all pages."""
        doc = pdf_parser.parse(multi_page_pdf_with_text_path)

        assert "Page 1 Content" in doc.content
        assert "Page 2 Content" in doc.content
        assert "Page 3 Content" in doc.content


class TestPDFParserErrorHandling:
    """Tests for PDF parser error handling."""

    def test_file_not_found_error(self, pdf_parser):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError) as exc_info:
            pdf_parser.parse("/nonexistent/path/to/file.pdf")

        assert "File not found" in str(exc_info.value)

    def test_invalid_pdf_raises_error(self, pdf_parser):
        """Test that invalid PDF files raise ValueError."""
        # Create a file that is not a valid PDF
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("This is not a valid PDF file")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                pdf_parser.parse(temp_path)

            assert "Failed to parse PDF" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_empty_pdf_handling(self, pdf_parser):
        """Test handling of empty PDF files."""
        # Create an empty file with .pdf extension
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                pdf_parser.parse(temp_path)
        finally:
            os.unlink(temp_path)


class TestPDFParserIntegration:
    """Integration tests for PDF parser with factory."""

    def test_factory_parse_pdf(self, sample_pdf_path):
        """Test parsing PDF through the factory."""
        parser = DocumentParserFactory.get_parser(sample_pdf_path)
        doc = parser.parse(sample_pdf_path)

        assert doc.doc_type == DocumentType.PDF

    def test_is_supported_returns_true_for_pdf(self):
        """Test that is_supported returns True for PDF files."""
        assert DocumentParserFactory.is_supported("document.pdf") is True
        assert DocumentParserFactory.is_supported("path/to/file.PDF") is True
