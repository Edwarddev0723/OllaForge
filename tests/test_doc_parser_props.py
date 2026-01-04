"""
Property-based tests for OllaForge document parser.

This module contains property-based tests using Hypothesis to verify
the correctness properties of the document parser module.

Feature: document-to-dataset
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ollaforge.doc_parser import (
    BaseParser,
    DocumentParserFactory,
    DocumentType,
    ParsedDocument,
    UnsupportedFormatError,
)

# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


class DummyParser(BaseParser):
    """A dummy parser for testing factory registration."""

    def __init__(self, doc_type: DocumentType = DocumentType.TEXT):
        self.doc_type = doc_type

    def parse(self, file_path: str) -> ParsedDocument:
        return ParsedDocument(
            content="dummy content", doc_type=self.doc_type, source_path=file_path
        )

    def supports(self, file_path: str) -> bool:
        return True


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset the factory before and after each test."""
    # Import register function to restore default parsers after test
    from ollaforge.doc_parser import register_default_parsers

    DocumentParserFactory.clear()
    yield
    DocumentParserFactory.clear()
    # Re-register default parsers for other tests/modules
    register_default_parsers()


# ============================================================================
# Strategies for generating test data
# ============================================================================

# Strategy for generating valid file extensions (with leading dot)
valid_extensions = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=10
).map(lambda x: "." + x.lower())

# Strategy for generating unsupported file extensions
# These are extensions that won't be registered in the factory
unsupported_extensions = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=10
).map(lambda x: "." + x.lower() + "_unsupported")


# Strategy for generating file paths with specific extensions
def file_path_with_extension(ext: str) -> st.SearchStrategy[str]:
    """Generate a file path with the given extension."""
    # Generate a filename (without extension) that doesn't start with a dot
    # to avoid issues with hidden files and extension detection
    return (
        st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"), whitelist_characters="_-"
            ),
            min_size=1,
            max_size=50,
        )
        .filter(lambda x: x.strip() and not x.startswith("."))
        .map(lambda x: x + ext)
    )


# ============================================================================
# Property Tests
# ============================================================================


@given(data=st.data())
@settings(max_examples=20)
def test_unsupported_format_error_contains_supported_formats(data):
    """
    **Feature: document-to-dataset, Property 2: Unsupported Format Error Handling**
    **Validates: Requirements 1.7**

    For any file with an unsupported extension, the Document_Parser SHALL raise
    an UnsupportedFormatError that includes the list of supported formats.

    This property verifies that:
    1. When an unsupported format is requested, UnsupportedFormatError is raised
    2. The error contains the list of all registered supported formats
    3. The error message includes the unsupported extension
    """
    # Generate 1-5 supported extensions and register them
    num_supported = data.draw(st.integers(min_value=1, max_value=5))
    supported_exts = data.draw(
        st.lists(
            valid_extensions,
            min_size=num_supported,
            max_size=num_supported,
            unique=True,
        )
    )

    # Register parsers for supported extensions
    for ext in supported_exts:
        DocumentParserFactory.register([ext], DummyParser())

    # Generate an unsupported extension that is NOT in the supported list
    unsupported_ext = data.draw(
        unsupported_extensions.filter(lambda x: x not in supported_exts)
    )

    # Generate a file path with the unsupported extension
    file_path = data.draw(file_path_with_extension(unsupported_ext))

    # Attempting to get a parser should raise UnsupportedFormatError
    with pytest.raises(UnsupportedFormatError) as exc_info:
        DocumentParserFactory.get_parser(file_path)

    error = exc_info.value

    # Verify the error contains the list of supported formats
    assert (
        error.supported_formats is not None
    ), "Error should contain supported_formats list"

    # Verify all registered formats are in the error's supported_formats
    for ext in supported_exts:
        normalized_ext = ext.lower() if ext.startswith(".") else "." + ext.lower()
        assert (
            normalized_ext in error.supported_formats
        ), f"Supported format '{normalized_ext}' should be in error.supported_formats: {error.supported_formats}"

    # Verify the unsupported extension is mentioned in the error message
    assert unsupported_ext in error.message or unsupported_ext in str(
        error
    ), f"Error message should mention the unsupported extension '{unsupported_ext}'"


@given(data=st.data())
@settings(max_examples=20)
def test_unsupported_format_error_with_empty_registry(data):
    """
    **Feature: document-to-dataset, Property 2: Unsupported Format Error Handling (Empty Registry)**
    **Validates: Requirements 1.7**

    For any file extension when no parsers are registered, the Document_Parser
    SHALL raise an UnsupportedFormatError with an empty supported formats list.
    """
    # Generate any file extension
    ext = data.draw(valid_extensions)
    file_path = data.draw(file_path_with_extension(ext))

    # Factory should be empty (cleared by fixture)
    assert (
        len(DocumentParserFactory.get_supported_formats()) == 0
    ), "Factory should be empty for this test"

    # Attempting to get a parser should raise UnsupportedFormatError
    with pytest.raises(UnsupportedFormatError) as exc_info:
        DocumentParserFactory.get_parser(file_path)

    error = exc_info.value

    # Verify the error contains an empty supported_formats list
    assert (
        error.supported_formats is not None
    ), "Error should contain supported_formats list"
    assert (
        len(error.supported_formats) == 0
    ), f"supported_formats should be empty, got: {error.supported_formats}"


@given(data=st.data())
@settings(max_examples=20)
def test_supported_format_returns_parser(data):
    """
    **Feature: document-to-dataset, Property 2: Supported Format Returns Parser**
    **Validates: Requirements 1.7**

    For any registered file extension, the Document_Parser SHALL return
    the appropriate parser without raising an error.

    This is the inverse property - verifying that supported formats work correctly.
    """
    # Generate 1-5 supported extensions and register them
    num_supported = data.draw(st.integers(min_value=1, max_value=5))
    supported_exts = data.draw(
        st.lists(
            valid_extensions,
            min_size=num_supported,
            max_size=num_supported,
            unique=True,
        )
    )

    # Register parsers for supported extensions
    parsers = {}
    for ext in supported_exts:
        parser = DummyParser()
        parsers[ext] = parser
        DocumentParserFactory.register([ext], parser)

    # Pick one of the supported extensions
    chosen_ext = data.draw(st.sampled_from(supported_exts))
    file_path = data.draw(file_path_with_extension(chosen_ext))

    # Getting a parser should succeed without raising an error
    parser = DocumentParserFactory.get_parser(file_path)

    # Verify we got a valid parser
    assert parser is not None, "Should return a parser for supported format"
    assert isinstance(parser, BaseParser), "Returned object should be a BaseParser"


# ============================================================================
# Property 1: Parser Content Extraction Tests
# ============================================================================

import os  # noqa: E402
import tempfile  # noqa: E402

# Strategy for generating Markdown content
markdown_content = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=500,
)


# Strategy for generating Markdown with headings
@st.composite
def markdown_with_headings(draw):
    """Generate Markdown content with headings."""
    num_sections = draw(st.integers(min_value=1, max_value=5))
    sections = []

    for _i in range(num_sections):
        level = draw(st.integers(min_value=1, max_value=6))
        title = draw(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N"), whitelist_characters=" "
                ),
                min_size=1,
                max_size=30,
            ).filter(lambda x: x.strip())
        )
        content = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=0,
                max_size=100,
            )
        )
        sections.append(f"{'#' * level} {title}\n\n{content}")

    return "\n\n".join(sections)


# Strategy for generating plain text content
plain_text_content = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=500,
)


# Strategy for generating JSON content
@st.composite
def json_content(draw):
    """Generate valid JSON content with string values."""
    import json

    num_fields = draw(st.integers(min_value=1, max_value=5))
    data = {}

    for _i in range(num_fields):
        key = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L",)),
                min_size=1,
                max_size=10,
            )
        )
        value = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=1,
                max_size=50,
            )
        )
        data[key] = value

    return json.dumps(data, ensure_ascii=False)


# Strategy for generating simple Python code
@st.composite
def python_code_content(draw):
    """Generate simple Python code content."""
    func_name = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L",)), min_size=1, max_size=20
        ).filter(lambda x: x.isidentifier())
    )

    body = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=50,
        )
    )

    return f'''def {func_name}():
    """A sample function."""
    # {body}
    pass
'''


@given(content=markdown_with_headings())
@settings(max_examples=20)
def test_markdown_parser_extracts_content(content):
    """
    **Feature: document-to-dataset, Property 1: Parser Content Extraction (Markdown)**
    **Validates: Requirements 1.1**

    For any Markdown file, parsing SHALL extract the text content,
    and the extracted content SHALL contain all textual information from the source.
    """
    from ollaforge.doc_parser import DocumentType, MarkdownParser

    # Create a temporary file with the content
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        temp_path = f.name

    try:
        parser = MarkdownParser()
        doc = parser.parse(temp_path)

        # Verify document type
        assert (
            doc.doc_type == DocumentType.MARKDOWN
        ), f"Expected MARKDOWN type, got {doc.doc_type}"

        # Verify content is extracted (not empty if source is not empty)
        if content.strip():
            assert (
                doc.content.strip()
            ), "Extracted content should not be empty for non-empty source"

        # Verify sections are created
        assert len(doc.sections) > 0, "At least one section should be created"

        # Verify source path is set
        assert doc.source_path, "Source path should be set"

    finally:
        os.unlink(temp_path)


@given(content=plain_text_content)
@settings(max_examples=20)
def test_text_parser_extracts_content(content):
    """
    **Feature: document-to-dataset, Property 1: Parser Content Extraction (Text)**
    **Validates: Requirements 1.4**

    For any plain text file, parsing SHALL read the content directly,
    and the extracted content SHALL match the source content.
    """
    from ollaforge.doc_parser import DocumentType, TextParser

    # Create a temporary file with the content
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        temp_path = f.name

    try:
        parser = TextParser()
        doc = parser.parse(temp_path)

        # Verify document type
        assert (
            doc.doc_type == DocumentType.TEXT
        ), f"Expected TEXT type, got {doc.doc_type}"

        # Verify content matches source (text parser should preserve content exactly)
        assert (
            doc.content == content
        ), "Extracted content should match source content exactly"

        # Verify sections are created
        assert len(doc.sections) > 0, "At least one section should be created"

    finally:
        os.unlink(temp_path)


@given(content=json_content())
@settings(max_examples=20)
def test_json_parser_extracts_strings(content):
    """
    **Feature: document-to-dataset, Property 1: Parser Content Extraction (JSON)**
    **Validates: Requirements 1.5**

    For any JSON file, parsing SHALL extract text values from the JSON structure,
    and all string values from the source SHALL be present in the extracted content.
    """
    import json

    from ollaforge.doc_parser import DocumentType, JSONParser

    # Create a temporary file with the content
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        temp_path = f.name

    try:
        parser = JSONParser()
        doc = parser.parse(temp_path)

        # Verify document type
        assert (
            doc.doc_type == DocumentType.JSON
        ), f"Expected JSON type, got {doc.doc_type}"

        # Parse the original JSON to get string values
        original_data = json.loads(content)

        # Extract all string values from original
        def get_strings(data):
            strings = []
            if isinstance(data, str) and data.strip():
                strings.append(data)
            elif isinstance(data, dict):
                for v in data.values():
                    strings.extend(get_strings(v))
            elif isinstance(data, list):
                for item in data:
                    strings.extend(get_strings(item))
            return strings

        original_strings = get_strings(original_data)

        # Verify all original strings are in extracted content
        for s in original_strings:
            assert (
                s in doc.content
            ), f"String '{s}' from source should be in extracted content"

    finally:
        os.unlink(temp_path)


@given(content=python_code_content())
@settings(max_examples=20)
def test_code_parser_extracts_content(content):
    """
    **Feature: document-to-dataset, Property 1: Parser Content Extraction (Code)**
    **Validates: Requirements 1.6**

    For any code file, parsing SHALL extract the code content with language detection,
    and the extracted content SHALL match the source code.
    """
    from ollaforge.doc_parser import CodeParser, DocumentType

    # Create a temporary file with the content
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        temp_path = f.name

    try:
        parser = CodeParser()
        doc = parser.parse(temp_path)

        # Verify document type
        assert (
            doc.doc_type == DocumentType.CODE
        ), f"Expected CODE type, got {doc.doc_type}"

        # Verify content matches source
        assert (
            doc.content == content
        ), "Extracted content should match source code exactly"

        # Verify language is detected
        assert (
            doc.metadata.get("language") == "python"
        ), f"Expected language 'python', got {doc.metadata.get('language')}"

        # Verify sections are created
        assert len(doc.sections) > 0, "At least one section should be created"

    finally:
        os.unlink(temp_path)


@given(data=st.data())
@settings(max_examples=20, deadline=None)
def test_html_parser_removes_tags(data):
    """
    **Feature: document-to-dataset, Property 1: Parser Content Extraction (HTML)**
    **Validates: Requirements 1.3**

    For any HTML file, parsing SHALL extract text content removing HTML tags,
    and the extracted content SHALL contain the text but not the HTML tags.
    """
    from ollaforge.doc_parser import DocumentType, HTMLParser

    # Generate text content
    text_content = data.draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P"), whitelist_characters=" "
            ),
            min_size=1,
            max_size=100,
        ).filter(lambda x: x.strip())
    )

    # Wrap in HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
<p>{text_content}</p>
</body>
</html>"""

    # Create a temporary file with the content
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html_content)
        temp_path = f.name

    try:
        parser = HTMLParser()
        doc = parser.parse(temp_path)

        # Verify document type
        assert (
            doc.doc_type == DocumentType.HTML
        ), f"Expected HTML type, got {doc.doc_type}"

        # Verify text content is extracted (compare stripped versions since HTML parser strips whitespace)
        stripped_text = text_content.strip()
        assert (
            stripped_text in doc.content
        ), f"Text content '{stripped_text}' should be in extracted content"

        # Verify HTML tags are removed
        assert (
            "<p>" not in doc.content
        ), "HTML tags should be removed from extracted content"
        assert (
            "<html>" not in doc.content
        ), "HTML tags should be removed from extracted content"
        assert (
            "</body>" not in doc.content
        ), "HTML tags should be removed from extracted content"

    finally:
        os.unlink(temp_path)
