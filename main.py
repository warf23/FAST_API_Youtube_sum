import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key

class SummarizeRequest(BaseModel):
    url: HttpUrl
    language: str = "English"

prompt_template = """
Please provide a concise and informative summary in the language {language} of the content found at the following URL. The summary should be approximately 300 words and should highlight the main points, key arguments, and any significant conclusions or insights presented in the content. Ensure that the summary is clear and easy to understand for someone who has not accessed the original content.

URL Content:
{text}

Summary:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])

language_codes = {'English': 'en', 'Arabic': 'ar', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 
                'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko'}

@app.post("/api/summarize")
async def summarize(request: SummarizeRequest, api_key: str = Depends(get_api_key)):
    if request.language not in language_codes:
        raise HTTPException(status_code=400, detail="Invalid language")

    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ API key not configured")

        model = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

        # Load content from URL
        url = str(request.url)
        logger.info(f"Attempting to load content from URL: {url}")

        if "youtube.com" in url or "youtu.be" in url:
            loader = YoutubeLoader.from_youtube_url(
                url,
                language=language_codes[request.language],
                add_video_info=True
            )
        else:
            loader = WebBaseLoader(url)

        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} document(s) from the URL")

        if not docs:
            raise HTTPException(status_code=400, detail="Unable to extract content from the provided URL")

        # Combine the documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
        combined_text = " ".join([doc.page_content for doc in texts])

        logger.info(f"Combined text length: {len(combined_text)} characters")

        if len(combined_text) < 100:  # Arbitrary threshold, adjust as needed
            raise HTTPException(status_code=400, detail="Insufficient content extracted from the URL")

        # Create the chain
        chain = (
            {"text": RunnablePassthrough(), "language": lambda _: request.language}
            | prompt
            | model
            | StrOutputParser()
        )

        # Run the chain
        logger.info("Running the summarization chain")
        output = chain.invoke(combined_text)

        logger.info(f"Summary generated successfully. Length: {len(output)} characters")

        return {"summary": output}

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}