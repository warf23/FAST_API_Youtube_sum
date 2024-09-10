import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import validators
from fastapi.middleware.cors import CORSMiddleware

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
Please provide a concise and informative summary of the physics content from the following text. 

Focus on the following aspects:
1. Key physics concepts and principles introduced
2. Important formulas or equations, if any
3. Main theoretical explanations or arguments (if there are equations or formulas added)
4. Significant experimental results or observations, if mentioned
5. Practical applications or real-world examples of the concepts
6. Any historical context or important physicists mentioned
7. Add all the equations with bold format 

The summary should be approximately 250-300 words in {language}, written in clear and accessible language. Ensure that the summary highlights the most relevant information and provides a solid foundation for understanding the topic.

Text to summarize:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])

language_codes = {'English': 'en', 'Arabic': 'ar', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 
                'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko'}

@app.post("/summarize")
async def summarize(request: SummarizeRequest, api_key: str = Depends(get_api_key)):
    if request.language not in language_codes:
        raise HTTPException(status_code=400, detail="Invalid language")

    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ API key not configured")

        llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-groq-70b-8192-tool-use-preview")

        if "youtube.com" in str(request.url):
            loader = YoutubeLoader.from_youtube_url(
                str(request.url),
                language=language_codes[request.language],
                add_video_info=True
            )
        else:
            loader = UnstructuredURLLoader(
                urls=[str(request.url)],
                ssl_verify=False,
                headers={"User-Agent": "Mozilla/5.0"}
            )

        docs = loader.load()

        summarize_chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
        output = summarize_chain.run(input_documents=docs, language=request.language)

        return {"summary": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}