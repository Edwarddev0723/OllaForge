"""
Document parser module for OllaForge.

This module provides functionality to parse various document formats and extract
text content for dataset generation. It supports Markdown, PDF, HTML, plain text,
JSON, and code files.

Key features:
- Modular parser architecture with factory pattern
- Support for multiple document formats
- Structured document representation with sections
- Extensible design for adding new format parsers

Requirements satisfied:
- 1.1: Markdown file parsing with heading structure preservation
- 1.2: PDF file text extraction
- 1.3: HTML file parsing with tag removal
- 1.4: Plain text file reading
- 1.5: JSON file text value extraction
- 1.6: Code file parsing with language detection
- 1.7: Unsupported format error handling with supported formats list
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Type
from pathlib import Path


class DocumentType(Enum):
    """
    Supported document types for parsing.
    
    Each type corresponds to a specific parser implementation that handles
    the format-specific extraction logic.
    """
    MARKDOWN = "markdown"
    PDF = "pdf"
    HTML = "html"
    TEXT = "text"
    JSON = "json"
    CODE = "code"


@dataclass
class DocumentSection:
    """
    Represents a section within a parsed document.
    
    Sections are used to preserve document structure during parsing,
    enabling intelligent chunking that respects semantic boundaries.
    
    Attributes:
        title: Optional section title (e.g., Markdown heading)
        content: The text content of the section
        level: Hierarchy level (1 for top-level, higher for nested)
        start_line: Starting line number in the source document
        end_line: Ending line number in the source document
    """
    title: Optional[str]
    content: str
    level: int
    start_line: int
    end_line: int


@dataclass
class ParsedDocument:
    """
    Represents a fully parsed document with extracted content and metadata.
    
    This is the primary output of all parser implementations, providing
    a unified structure regardless of the source document format.
    
    Attributes:
        content: The complete extracted text content
        doc_type: The type of document that was parsed
        metadata: Additional metadata (title, author, language, etc.)
        sections: List of document sections for structured access
        source_path: Path to the original source file
    """
    content: str
    doc_type: DocumentType
    metadata: Dict[str, any] = field(default_factory=dict)
    sections: List[DocumentSection] = field(default_factory=list)
    source_path: str = ""


class UnsupportedFormatError(Exception):
    """
    Exception raised when attempting to parse an unsupported file format.
    
    This exception includes the list of supported formats to help users
    understand which file types can be processed.
    
    Attributes:
        message: Human-readable error message
        supported_formats: List of file extensions that are supported
        
    Requirements satisfied:
    - 1.7: Return descriptive error message listing supported formats
    """
    
    def __init__(self, message: str, supported_formats: List[str]):
        self.message = message
        self.supported_formats = supported_formats
        super().__init__(message)
    
    def __str__(self) -> str:
        formats_str = ", ".join(self.supported_formats) if self.supported_formats else "none"
        return f"{self.message}. Supported formats: {formats_str}"


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.
    
    Each concrete parser implementation must inherit from this class
    and implement the parse() and supports() methods for their specific
    document format.
    """
    
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a document file and extract its content.
        
        Args:
            file_path: Path to the document file to parse
            
        Returns:
            ParsedDocument: Structured representation of the parsed content
            
        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be read
            ParseError: If parsing fails for format-specific reasons
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """
        Check if this parser supports the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if this parser can handle the file format
        """
        pass


class DocumentParserFactory:
    """
    Factory class for creating and managing document parsers.
    
    This factory maintains a registry of parsers mapped to file extensions,
    allowing automatic parser selection based on file type. New parsers
    can be registered dynamically using the register() method.
    
    Usage:
        # Register a parser for specific extensions
        DocumentParserFactory.register(['.md', '.markdown'], MarkdownParser())
        
        # Get the appropriate parser for a file
        parser = DocumentParserFactory.get_parser('document.md')
        result = parser.parse('document.md')
    """
    
    _parsers: Dict[str, BaseParser] = {}
    
    @classmethod
    def register(cls, extensions: List[str], parser: BaseParser) -> None:
        """
        Register a parser for one or more file extensions.
        
        Args:
            extensions: List of file extensions (e.g., ['.md', '.markdown'])
            parser: Parser instance to handle these extensions
        """
        for ext in extensions:
            # Normalize extension to lowercase with leading dot
            normalized_ext = ext.lower()
            if not normalized_ext.startswith('.'):
                normalized_ext = '.' + normalized_ext
            cls._parsers[normalized_ext] = parser
    
    @classmethod
    def get_parser(cls, file_path: str) -> BaseParser:
        """
        Get the appropriate parser for a file based on its extension.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            BaseParser: Parser instance that can handle the file format
            
        Raises:
            UnsupportedFormatError: If no parser is registered for the file extension
            
        Requirements satisfied:
        - 1.7: Return descriptive error message listing supported formats
        """
        ext = Path(file_path).suffix.lower()
        
        if ext not in cls._parsers:
            raise UnsupportedFormatError(
                f"Unsupported format: {ext}",
                supported_formats=cls.get_supported_formats()
            )
        
        return cls._parsers[ext]
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """
        Get a list of all supported file extensions.
        
        Returns:
            List[str]: Sorted list of supported file extensions
        """
        return sorted(cls._parsers.keys())
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered parsers.
        
        This is primarily useful for testing to reset the factory state.
        """
        cls._parsers.clear()
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """
        Check if a file format is supported.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if the file format is supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in cls._parsers


# ============================================================================
# Concrete Parser Implementations
# ============================================================================


class MarkdownParser(BaseParser):
    """
    Parser for Markdown files (.md).
    
    Extracts text content while preserving heading structure. Sections are
    created based on heading hierarchy (# for level 1, ## for level 2, etc.).
    
    Requirements satisfied:
    - 1.1: Extract text content preserving heading structure
    """
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a Markdown file and extract its content with structure.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            ParsedDocument with content and sections based on headings
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            content = path.read_text(encoding='utf-8')
        except PermissionError:
            raise PermissionError(f"Permission denied: {file_path}")
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    content = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Unable to decode file: {file_path}")
        
        sections = self._extract_sections(content)
        metadata = self._extract_metadata(content)
        
        return ParsedDocument(
            content=content,
            doc_type=DocumentType.MARKDOWN,
            metadata=metadata,
            sections=sections,
            source_path=str(path.absolute())
        )
    
    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file."""
        ext = Path(file_path).suffix.lower()
        return ext in ['.md', '.markdown']
    
    def _extract_sections(self, content: str) -> List[DocumentSection]:
        """Extract sections based on Markdown headings."""
        sections = []
        lines = content.split('\n')
        
        current_section_start = 0
        current_section_title = None
        current_section_level = 0
        current_section_content_lines = []
        
        for i, line in enumerate(lines):
            # Check for ATX-style headings (# Heading)
            stripped = line.lstrip()
            if stripped.startswith('#'):
                # Count the heading level
                level = 0
                for char in stripped:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                # Valid heading: 1-6 # followed by space or end of line
                if level <= 6 and (len(stripped) == level or stripped[level] == ' '):
                    # Save previous section if exists
                    if current_section_content_lines or current_section_title:
                        sections.append(DocumentSection(
                            title=current_section_title,
                            content='\n'.join(current_section_content_lines).strip(),
                            level=current_section_level,
                            start_line=current_section_start,
                            end_line=i - 1
                        ))
                    
                    # Start new section
                    current_section_start = i
                    current_section_title = stripped[level:].strip()
                    current_section_level = level
                    current_section_content_lines = []
                    continue
            
            current_section_content_lines.append(line)
        
        # Add the last section
        if current_section_content_lines or current_section_title:
            sections.append(DocumentSection(
                title=current_section_title,
                content='\n'.join(current_section_content_lines).strip(),
                level=current_section_level,
                start_line=current_section_start,
                end_line=len(lines) - 1
            ))
        
        # If no sections found, create one section with all content
        if not sections:
            sections.append(DocumentSection(
                title=None,
                content=content.strip(),
                level=0,
                start_line=0,
                end_line=len(lines) - 1
            ))
        
        return sections
    
    def _extract_metadata(self, content: str) -> Dict[str, any]:
        """Extract metadata from Markdown content (e.g., YAML front matter)."""
        metadata = {}
        
        # Check for YAML front matter
        if content.startswith('---'):
            lines = content.split('\n')
            end_index = -1
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    end_index = i
                    break
            
            if end_index > 0:
                # Simple YAML parsing for common fields
                for line in lines[1:end_index]:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
        
        # Extract first heading as title if not in metadata
        if 'title' not in metadata:
            for line in content.split('\n'):
                stripped = line.strip()
                if stripped.startswith('# '):
                    metadata['title'] = stripped[2:].strip()
                    break
        
        return metadata


class HTMLParser(BaseParser):
    """
    Parser for HTML files (.html, .htm).
    
    Extracts text content by removing HTML tags while preserving
    basic structure through whitespace.
    
    Requirements satisfied:
    - 1.3: Extract text content removing HTML tags
    """
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse an HTML file and extract text content.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            ParsedDocument with extracted text content
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            html_content = path.read_text(encoding='utf-8')
        except PermissionError:
            raise PermissionError(f"Permission denied: {file_path}")
        except UnicodeDecodeError:
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    html_content = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Unable to decode file: {file_path}")
        
        # Try to use BeautifulSoup if available, otherwise use regex
        try:
            from bs4 import BeautifulSoup
            text_content = self._extract_with_beautifulsoup(html_content)
        except ImportError:
            text_content = self._extract_with_regex(html_content)
        
        metadata = self._extract_metadata(html_content)
        sections = self._extract_sections(text_content)
        
        return ParsedDocument(
            content=text_content,
            doc_type=DocumentType.HTML,
            metadata=metadata,
            sections=sections,
            source_path=str(path.absolute())
        )
    
    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file."""
        ext = Path(file_path).suffix.lower()
        return ext in ['.html', '.htm']
    
    def _extract_with_beautifulsoup(self, html_content: str) -> str:
        """Extract text using BeautifulSoup."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'head', 'meta', 'link']):
            element.decompose()
        
        # Get text with proper spacing
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up multiple newlines
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _extract_with_regex(self, html_content: str) -> str:
        """Extract text using regex (fallback when BeautifulSoup not available)."""
        import re
        
        # Remove script and style content
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Replace block elements with newlines
        text = re.sub(r'<(br|p|div|h[1-6]|li|tr)[^>]*>', '\n', text, flags=re.IGNORECASE)
        
        # Remove all remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        import html
        text = html.unescape(text)
        
        # Clean up whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _extract_metadata(self, html_content: str) -> Dict[str, any]:
        """Extract metadata from HTML (title, meta tags)."""
        import re
        metadata = {}
        
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            import html
            metadata['title'] = html.unescape(title_match.group(1).strip())
        
        # Extract meta description
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', 
                               html_content, re.IGNORECASE)
        if desc_match:
            import html
            metadata['description'] = html.unescape(desc_match.group(1).strip())
        
        return metadata
    
    def _extract_sections(self, text_content: str) -> List[DocumentSection]:
        """Create sections from extracted text."""
        lines = text_content.split('\n')
        
        # For HTML, we create a single section with all content
        # More sophisticated section detection could be added later
        return [DocumentSection(
            title=None,
            content=text_content,
            level=0,
            start_line=0,
            end_line=len(lines) - 1
        )]


class TextParser(BaseParser):
    """
    Parser for plain text files (.txt).
    
    Reads text content directly with encoding detection.
    
    Requirements satisfied:
    - 1.4: Read content directly
    """
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            ParsedDocument with the text content
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = self._read_with_encoding_detection(path)
        sections = self._extract_sections(content)
        
        return ParsedDocument(
            content=content,
            doc_type=DocumentType.TEXT,
            metadata={},
            sections=sections,
            source_path=str(path.absolute())
        )
    
    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file."""
        ext = Path(file_path).suffix.lower()
        return ext == '.txt'
    
    def _read_with_encoding_detection(self, path: Path) -> str:
        """Read file with encoding detection."""
        # Try common encodings in order of likelihood
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
        
        for encoding in encodings:
            try:
                content = path.read_text(encoding=encoding)
                return content
            except (UnicodeDecodeError, PermissionError) as e:
                if isinstance(e, PermissionError):
                    raise PermissionError(f"Permission denied: {path}")
                continue
        
        # If all encodings fail, read as binary and decode with errors='replace'
        try:
            content = path.read_bytes().decode('utf-8', errors='replace')
            return content
        except PermissionError:
            raise PermissionError(f"Permission denied: {path}")
    
    def _extract_sections(self, content: str) -> List[DocumentSection]:
        """Create sections from text content based on blank line separation."""
        lines = content.split('\n')
        sections = []
        
        current_content_lines = []
        current_start = 0
        
        for i, line in enumerate(lines):
            if line.strip() == '' and current_content_lines:
                # Check if we have substantial content (not just whitespace)
                content_text = '\n'.join(current_content_lines).strip()
                if content_text:
                    sections.append(DocumentSection(
                        title=None,
                        content=content_text,
                        level=0,
                        start_line=current_start,
                        end_line=i - 1
                    ))
                current_content_lines = []
                current_start = i + 1
            else:
                current_content_lines.append(line)
        
        # Add remaining content
        if current_content_lines:
            content_text = '\n'.join(current_content_lines).strip()
            if content_text:
                sections.append(DocumentSection(
                    title=None,
                    content=content_text,
                    level=0,
                    start_line=current_start,
                    end_line=len(lines) - 1
                ))
        
        # If no sections, create one with all content
        if not sections:
            sections.append(DocumentSection(
                title=None,
                content=content.strip(),
                level=0,
                start_line=0,
                end_line=len(lines) - 1
            ))
        
        return sections


class JSONParser(BaseParser):
    """
    Parser for JSON files (.json).
    
    Recursively extracts string values from JSON structures.
    
    Requirements satisfied:
    - 1.5: Extract text values from JSON structure
    """
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a JSON file and extract text values.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            ParsedDocument with extracted text content
        """
        import json
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            content = path.read_text(encoding='utf-8')
        except PermissionError:
            raise PermissionError(f"Permission denied: {file_path}")
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        
        # Extract all string values recursively
        text_values = self._extract_strings(data)
        extracted_content = '\n'.join(text_values)
        
        metadata = {'original_structure': type(data).__name__}
        if isinstance(data, dict):
            # Extract common metadata fields
            for key in ['title', 'name', 'description', 'id']:
                if key in data and isinstance(data[key], str):
                    metadata[key] = data[key]
        
        sections = [DocumentSection(
            title=None,
            content=extracted_content,
            level=0,
            start_line=0,
            end_line=len(text_values) - 1
        )]
        
        return ParsedDocument(
            content=extracted_content,
            doc_type=DocumentType.JSON,
            metadata=metadata,
            sections=sections,
            source_path=str(path.absolute())
        )
    
    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file."""
        ext = Path(file_path).suffix.lower()
        return ext == '.json'
    
    def _extract_strings(self, data: any, depth: int = 0) -> List[str]:
        """Recursively extract string values from JSON data."""
        strings = []
        
        if isinstance(data, str):
            # Only include non-empty strings
            if data.strip():
                strings.append(data)
        elif isinstance(data, dict):
            for key, value in data.items():
                # Include key if it's descriptive (not just an ID)
                if isinstance(key, str) and len(key) > 2 and not key.startswith('_'):
                    strings.extend(self._extract_strings(value, depth + 1))
                else:
                    strings.extend(self._extract_strings(value, depth + 1))
        elif isinstance(data, list):
            for item in data:
                strings.extend(self._extract_strings(item, depth + 1))
        # Skip numbers, booleans, None
        
        return strings


class PDFParser(BaseParser):
    """
    Parser for PDF files (.pdf).
    
    Extracts text content from PDF documents using PyPDF2.
    Handles multi-page documents by extracting text from all pages.
    
    Requirements satisfied:
    - 1.2: Extract text content from all pages
    """
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a PDF file and extract text content from all pages.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ParsedDocument with extracted text content
            
        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be read
            ValueError: If PDF parsing fails
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Try to import PyPDF2
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                raise ImportError(
                    "PyPDF2 is required for PDF parsing. "
                    "Install it with: pip install PyPDF2"
                )
            
            # Open and read the PDF
            try:
                reader = PdfReader(str(path))
            except PermissionError:
                raise PermissionError(f"Permission denied: {file_path}")
            except Exception as e:
                raise ValueError(f"Failed to parse PDF {file_path}: {e}")
            
            # Extract text from all pages
            pages_text = []
            sections = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages_text.append(page_text)
                        
                        # Create a section for each page
                        sections.append(DocumentSection(
                            title=f"Page {page_num + 1}",
                            content=page_text.strip(),
                            level=1,
                            start_line=page_num,
                            end_line=page_num
                        ))
                except Exception as e:
                    # Log warning but continue with other pages
                    pages_text.append(f"[Error extracting page {page_num + 1}: {e}]")
            
            # Combine all pages
            content = '\n\n'.join(pages_text)
            
            # Extract metadata
            metadata = self._extract_metadata(reader)
            metadata['page_count'] = len(reader.pages)
            
            # If no sections were created, create one with all content
            if not sections:
                sections.append(DocumentSection(
                    title=None,
                    content=content.strip() if content else "",
                    level=0,
                    start_line=0,
                    end_line=0
                ))
            
            return ParsedDocument(
                content=content,
                doc_type=DocumentType.PDF,
                metadata=metadata,
                sections=sections,
                source_path=str(path.absolute())
            )
            
        except (FileNotFoundError, PermissionError, ImportError, ValueError):
            raise
        except Exception as e:
            raise ValueError(f"Failed to parse PDF {file_path}: {e}")
    
    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file."""
        ext = Path(file_path).suffix.lower()
        return ext == '.pdf'
    
    def _extract_metadata(self, reader) -> Dict[str, any]:
        """Extract metadata from PDF document."""
        metadata = {}
        
        try:
            if reader.metadata:
                # Common PDF metadata fields
                if reader.metadata.title:
                    metadata['title'] = str(reader.metadata.title)
                if reader.metadata.author:
                    metadata['author'] = str(reader.metadata.author)
                if reader.metadata.subject:
                    metadata['subject'] = str(reader.metadata.subject)
                if reader.metadata.creator:
                    metadata['creator'] = str(reader.metadata.creator)
        except Exception:
            # Metadata extraction is optional, don't fail if it doesn't work
            pass
        
        return metadata


class CodeParser(BaseParser):
    """
    Parser for code files (.py, .js, .ts, .java, .go, .rs, .c, .cpp, .rb).
    
    Extracts code content with language detection based on file extension.
    
    Requirements satisfied:
    - 1.6: Extract code content with language detection
    """
    
    # Mapping of file extensions to language names
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.rb': 'ruby',
    }
    
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a code file and extract its content.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            ParsedDocument with code content and language metadata
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            content = path.read_text(encoding='utf-8')
        except PermissionError:
            raise PermissionError(f"Permission denied: {file_path}")
        except UnicodeDecodeError:
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    content = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Unable to decode file: {file_path}")
        
        ext = path.suffix.lower()
        language = self.LANGUAGE_MAP.get(ext, 'unknown')
        
        metadata = {
            'language': language,
            'extension': ext,
            'filename': path.name,
        }
        
        sections = self._extract_sections(content, language)
        
        return ParsedDocument(
            content=content,
            doc_type=DocumentType.CODE,
            metadata=metadata,
            sections=sections,
            source_path=str(path.absolute())
        )
    
    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file."""
        ext = Path(file_path).suffix.lower()
        return ext in self.LANGUAGE_MAP
    
    def _extract_sections(self, content: str, language: str) -> List[DocumentSection]:
        """Extract sections based on code structure (functions, classes)."""
        lines = content.split('\n')
        sections = []
        
        # Simple heuristic-based section detection
        # For more accurate parsing, a proper AST parser would be needed
        
        current_section_start = 0
        current_section_title = None
        current_section_content_lines = []
        
        for i, line in enumerate(lines):
            # Detect function/class definitions based on language
            section_title = self._detect_definition(line, language)
            
            if section_title:
                # Save previous section if exists
                if current_section_content_lines:
                    sections.append(DocumentSection(
                        title=current_section_title,
                        content='\n'.join(current_section_content_lines),
                        level=1 if current_section_title else 0,
                        start_line=current_section_start,
                        end_line=i - 1
                    ))
                
                current_section_start = i
                current_section_title = section_title
                current_section_content_lines = [line]
            else:
                current_section_content_lines.append(line)
        
        # Add the last section
        if current_section_content_lines:
            sections.append(DocumentSection(
                title=current_section_title,
                content='\n'.join(current_section_content_lines),
                level=1 if current_section_title else 0,
                start_line=current_section_start,
                end_line=len(lines) - 1
            ))
        
        # If no sections found, create one section with all content
        if not sections:
            sections.append(DocumentSection(
                title=None,
                content=content,
                level=0,
                start_line=0,
                end_line=len(lines) - 1
            ))
        
        return sections
    
    def _detect_definition(self, line: str, language: str) -> Optional[str]:
        """Detect function/class definitions in a line of code."""
        import re
        
        stripped = line.strip()
        
        if language == 'python':
            # Python: def function_name or class ClassName
            match = re.match(r'^(def|class)\s+(\w+)', stripped)
            if match:
                return f"{match.group(1)} {match.group(2)}"
        
        elif language in ['javascript', 'typescript']:
            # JS/TS: function name, class Name, const name = function/arrow
            match = re.match(r'^(function|class)\s+(\w+)', stripped)
            if match:
                return f"{match.group(1)} {match.group(2)}"
            match = re.match(r'^(const|let|var)\s+(\w+)\s*=\s*(function|\(|async)', stripped)
            if match:
                return f"function {match.group(2)}"
        
        elif language == 'java':
            # Java: public/private/protected class/void/type methodName
            match = re.match(r'^(public|private|protected)?\s*(static)?\s*(class|void|\w+)\s+(\w+)\s*[\({]', stripped)
            if match:
                return f"{match.group(3)} {match.group(4)}"
        
        elif language == 'go':
            # Go: func (receiver) name or func name
            match = re.match(r'^func\s+(\([^)]+\)\s+)?(\w+)', stripped)
            if match:
                return f"func {match.group(2)}"
            match = re.match(r'^type\s+(\w+)\s+(struct|interface)', stripped)
            if match:
                return f"type {match.group(1)}"
        
        elif language == 'rust':
            # Rust: fn name, struct Name, impl Name
            match = re.match(r'^(pub\s+)?(fn|struct|enum|impl|trait)\s+(\w+)', stripped)
            if match:
                return f"{match.group(2)} {match.group(3)}"
        
        elif language in ['c', 'cpp']:
            # C/C++: type function_name( or class Name
            match = re.match(r'^(class|struct)\s+(\w+)', stripped)
            if match:
                return f"{match.group(1)} {match.group(2)}"
            # Function detection is more complex in C/C++, skip for now
        
        elif language == 'ruby':
            # Ruby: def method_name, class ClassName, module ModuleName
            match = re.match(r'^(def|class|module)\s+(\w+)', stripped)
            if match:
                return f"{match.group(1)} {match.group(2)}"
        
        return None


# ============================================================================
# Parser Registration
# ============================================================================

def register_default_parsers() -> None:
    """
    Register all default parsers with the factory.
    
    This function should be called once at module initialization to make
    all parsers available through the DocumentParserFactory.
    """
    # Markdown parser
    DocumentParserFactory.register(['.md', '.markdown'], MarkdownParser())
    
    # HTML parser
    DocumentParserFactory.register(['.html', '.htm'], HTMLParser())
    
    # Text parser
    DocumentParserFactory.register(['.txt'], TextParser())
    
    # JSON parser
    DocumentParserFactory.register(['.json'], JSONParser())
    
    # PDF parser
    DocumentParserFactory.register(['.pdf'], PDFParser())
    
    # Code parser - register for all supported extensions
    code_parser = CodeParser()
    DocumentParserFactory.register(
        ['.py', '.js', '.ts', '.java', '.go', '.rs', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.rb'],
        code_parser
    )


# Register parsers when module is imported
register_default_parsers()
