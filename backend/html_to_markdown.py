"""HTML to Markdown processing for Weaviate storage.

This module provides functionality to convert HTML content to Markdown format
using the jinaai/ReaderLM-v2 model. This improves readability and reduces noise
in data stored in Weaviate.
"""
import re
import logging
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patterns for HTML cleaning
SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"

# Global variables to hold the loaded model and tokenizer
_tokenizer = None
_model = None
_device = "cpu"  # Default to CPU


def initialize_model(device: str = "cpu") -> None:
    """Initialize the model and tokenizer for HTML to Markdown conversion.
    
    Args:
        device: The device to use for inference ("cpu" or "cuda")
    """
    global _tokenizer, _model, _device
    
    if _model is not None and _tokenizer is not None:
        # Already initialized
        return
    
    _device = device
    logger.info(f"Initializing ReaderLM model on {_device}")
    
    try:
        _tokenizer = AutoTokenizer.from_pretrained("jinaai/ReaderLM-v2")
        _model = AutoModelForCausalLM.from_pretrained("jinaai/ReaderLM-v2").to(_device)
        logger.info("ReaderLM model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize HTML to Markdown model: {e}")
        raise


def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    """Replace SVG content with a placeholder.
    
    Args:
        html: The HTML string containing SVG elements
        new_content: The placeholder text to use
        
    Returns:
        HTML string with SVG content replaced
    """
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    """Replace base64 encoded images with a simple image tag.
    
    Args:
        html: The HTML string containing base64 images
        new_image_src: The source URL to use in the replacement img tag
        
    Returns:
        HTML string with base64 images replaced
    """
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False) -> str:
    """Clean HTML by removing scripts, styles, and other non-content elements.
    
    Args:
        html: The HTML string to clean
        clean_svg: Whether to replace SVG content with placeholder
        clean_base64: Whether to replace base64 encoded images
        
    Returns:
        Cleaned HTML string
    """
    html = re.sub(
        SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
    return html


def create_prompt(
    text: str, instruction: str = None, schema: str = None
) -> str:
    """Create a prompt for the model with optional instruction and JSON schema.
    
    Args:
        text: The HTML text to process
        instruction: Optional custom instruction
        schema: Optional JSON schema for structured extraction
        
    Returns:
        Formatted prompt string
    """
    global _tokenizer
    
    if _tokenizer is None:
        initialize_model()
    
    if not instruction:
        instruction = "Extract the main content from the given HTML and convert it to Markdown format."
    if schema:
        instruction = "Extract the specified information from a list of news threads and present it in a structured JSON format."
        prompt = f"{instruction}\n```html\n{text}\n```\nThe JSON schema is as follows:```json\n{schema}\n```"
    else:
        prompt = f"{instruction}\n```html\n{text}\n```"

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def html_to_markdown(html: str, max_new_tokens: int = 1024) -> str:
    """Convert HTML to Markdown using the ReaderLM model.
    
    Args:
        html: The HTML string to convert
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Markdown formatted content
    """
    global _tokenizer, _model, _device
    
    if _model is None or _tokenizer is None:
        initialize_model()
    
    # Clean the HTML first
    html = clean_html(html, clean_svg=True, clean_base64=True)
    
    # Skip conversion if the input is very small or empty
    if len(html.strip()) < 10:
        return html.strip()
    
    try:
        input_prompt = create_prompt(html)
        
        inputs = _tokenizer.encode(input_prompt, return_tensors="pt").to(_device)
        outputs = _model.generate(
            inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=0, 
            do_sample=False, 
            repetition_penalty=1.08
        )
        
        markdown_result = _tokenizer.decode(outputs[0])
        
        # Extract just the generated markdown from the response
        try:
            # First try to find markdown code block
            if "```markdown" in markdown_result:
                markdown = markdown_result.split("```markdown")[1].split("```")[0].strip()
            # Otherwise look for assistant response
            elif "ASSISTANT:" in markdown_result:
                markdown = markdown_result.split("ASSISTANT:")[1].strip()
            else:
                # Just return everything after the prompt
                end_of_prompt = input_prompt.strip()
                markdown = markdown_result.replace(end_of_prompt, "").strip()
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to extract markdown from model output: {e}")
            # Fall back to returning the whole output
            markdown = markdown_result
        
        return markdown
    
    except Exception as e:
        logger.error(f"HTML to Markdown conversion failed: {e}")
        # Return the original HTML as fallback
        return html


def process_documents(documents: list) -> list:
    """Process a list of documents by converting HTML content to Markdown.
    
    Args:
        documents: List of langchain Document objects
        
    Returns:
        List of documents with HTML converted to Markdown
    """
    processed_docs = []
    
    for doc in documents:
        # Only convert if content looks like HTML
        if doc.page_content and "<" in doc.page_content and ">" in doc.page_content:
            try:
                doc.page_content = html_to_markdown(doc.page_content)
            except Exception as e:
                logger.warning(f"Error processing document: {e}")
                # Keep original content if conversion fails
        
        processed_docs.append(doc)
    
    return processed_docs


# Simple test function
if __name__ == "__main__":
    test_html = """
    <html>
    <head>
        <title>Test Page</title>
        <style>body { font-family: Arial; }</style>
    </head>
    <body>
        <h1>Hello, World!</h1>
        <p>This is a <b>test</b> of the HTML to Markdown conversion.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </body>
    </html>
    """
    
    # Initialize model
    initialize_model()
    
    # Convert to markdown
    markdown = html_to_markdown(test_html)
    print("Converted Markdown:")
    print(markdown)