from fastapi import FastAPI
import uvicorn
#import vertexai
#from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
#from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, ToolConfig

GCP_PROJECT="fifth-dynamics-452002-t1"
GCP_LOCATION="us-central1"

#vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
#llm = GenerativeModel("gemini-1.5-flash-002")

app=FastAPI()

@app.post("/rag/str")
def rag_string(question: str):
    print(f"RAG RECEIVED: {question}")
    return f'RAG RECEIVED: {question}' #llm.generate_content(question,stream=False).text.strip()

#if __name__ == "__main__":
#    uvicorn.run(app, host='127.0.0.1', port=8003)