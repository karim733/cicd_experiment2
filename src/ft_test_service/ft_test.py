from fastapi import FastAPI
import uvicorn
#import vertexai
#from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
#from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, ToolConfig
from pydantic import BaseModel, Field
from typing import List, Dict
from fastapi import FastAPI

GCP_PROJECT="fifth-dynamics-452002-t1"
GCP_LOCATION="us-central1"




#vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
#llm = GenerativeModel("gemini-1.5-flash-002")

app=FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict] 

@app.post("/sft/inference")
def sft_inference(req : ChatRequest):
    print(f"FT RECEIVED: {req.messages}")
    return req.messages #llm.generate_content(req.messages[-1].value,stream=False).text.strip()

#if __name__ == "__main__":
#    uvicorn.run(app, host='127.0.0.1', port=8002)