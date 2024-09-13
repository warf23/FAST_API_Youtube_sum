"""
FastAPI Summarization API

This module provides a FastAPI application for text summarization of web content.
It offers two endpoints: one for basic summarization and another for advanced summarization
with customizable parameters.

The API uses the Groq language model via LangChain for generating summaries.

Dependencies:
- FastAPI
- Pydantic
- LangChain
- Validators

Usage:
Run the script and access the API documentation at http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import validators

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Constants
LANGUAGE_CODES: Dict[str, str] = {
    'English': 'en', 'Arabic': 'ar', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
    'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko'
}

FOCUS_AREAS: Dict[str, str] = {
    "general": "provide a balanced overview of all main points",
    "technical": "emphasize technical details, methodologies, and specifications",
    "business": "highlight business implications, market analysis, and strategic insights",
    "academic": "focus on research findings, theoretical frameworks, and scholarly implications"
}

# Pydantic models for request validation
class SummarizeRequest(BaseModel):
    groq_api_key: str
    url: str
    language: str = "English"

class AdvancedSummarizeRequest(SummarizeRequest):
    word_count: int = 300
    focus: str = "general"

# Prompt templates
BASIC_PROMPT_TEMPLATE = """
Please provide a concise and informative summary in the language {language} of the content found at the following URL. 
The summary should be approximately 300 words and should highlight the main points, key arguments, and any significant 
conclusions or insights presented in the content. Ensure that the summary is clear and easy to understand for someone 
who has not accessed the original content.

URL Content:
{text}
"""

ADVANCED_PROMPT_TEMPLATE = """
Please provide a {focus} summary in the language {language} of the content found at the following URL. 
The summary should be approximately {word_count} words and should highlight the main points, key arguments, 
and any significant conclusions or insights presented in the content. Ensure that the summary is clear and 
easy to understand for someone who has not accessed the original content.

Focus areas for the summary:
- If the focus is "general", {general_focus}
- If the focus is "technical", {technical_focus}
- If the focus is "business", {business_focus}
- If the focus is "academic", {academic_focus}

URL Content:
{text}
"""

# FastAPI app instance
app = FastAPI(
    title="Web Content Summarization API",
    description="An API for summarizing web content using advanced language models.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_and_process_url(url: str, language: str) -> str:
    """
    Load content from a given URL and process it for summarization.

    Args:
        url (str): The URL to load content from.
        language (str): The language code for content processing.

    Returns:
        str: Processed text content ready for summarization.
    """
    if "youtube.com" in url:
        loader = YoutubeLoader.from_youtube_url(url, language=LANGUAGE_CODES[language], add_video_info=True)
    else:
        loader = UnstructuredURLLoader(
            urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0"}
        )

    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    return " ".join([doc.page_content for doc in texts])

def create_summarization_chain(model: ChatGroq, prompt: PromptTemplate) -> Any:
    """
    Create a LangChain chain for text summarization.

    Args:
        model (ChatGroq): The initialized Groq language model.
        prompt (PromptTemplate): The prompt template for summarization.

    Returns:
        Any: A LangChain chain for text summarization.
    """
    return (
        {"text": RunnablePassthrough(), "language": lambda x: x["language"]}
        | prompt
        | model
        | StrOutputParser()
    )

@app.post("/summary", summary="Generate a basic summary of web content")
async def summarize(request: SummarizeRequest):
    """
    Generate a basic summary of web content.

    This endpoint takes a URL and language, fetches the content, and generates a summary
    using a language model.

    Args:
        request (SummarizeRequest): The request body containing the URL and language.

    Returns:
        dict: A dictionary containing the generated summary.

    Raises:
        HTTPException: If there's an error in processing the request or generating the summary.
    """
    if not validators.url(request.url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    if request.language not in LANGUAGE_CODES:
        raise HTTPException(status_code=400, detail="Invalid language")

    try:
        model = ChatGroq(groq_api_key=request.groq_api_key, model_name="llama-3.1-70b-versatile")
        combined_text = load_and_process_url(request.url, request.language)

        prompt = PromptTemplate(template=BASIC_PROMPT_TEMPLATE, input_variables=["text", "language"])
        chain = create_summarization_chain(model, prompt)

        output = chain.invoke({"text": combined_text, "language": request.language})
        return {"summary": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@app.post("/summary/advanced", summary="Generate an advanced summary of web content")
async def advanced_summarize(request: AdvancedSummarizeRequest):
    """
    Generate an advanced summary of web content with customizable parameters.

    This endpoint takes a URL, language, word count, and focus area. It fetches the content
    and generates a customized summary using a language model.

    Args:
        request (AdvancedSummarizeRequest): The request body containing URL, language, word count, and focus area.

    Returns:
        dict: A dictionary containing the generated summary.

    Raises:
        HTTPException: If there's an error in processing the request or generating the summary.
    """
    if not validators.url(request.url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    if request.language not in LANGUAGE_CODES:
        raise HTTPException(status_code=400, detail="Invalid language")

    if request.focus not in FOCUS_AREAS:
        raise HTTPException(status_code=400, detail="Invalid focus area")

    try:
        model = ChatGroq(groq_api_key=request.groq_api_key, model_name="llama-3.1-70b-versatile")
        combined_text = load_and_process_url(request.url, request.language)

        prompt = PromptTemplate(
            template=ADVANCED_PROMPT_TEMPLATE,
            input_variables=["text", "language", "word_count", "focus"],
            partial_variables=FOCUS_AREAS
        )
        chain = create_summarization_chain(model, prompt)

        output = chain.invoke({
            "text": combined_text,
            "language": request.language,
            "word_count": request.word_count,
            "focus": request.focus
        })
        return {"summary": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)