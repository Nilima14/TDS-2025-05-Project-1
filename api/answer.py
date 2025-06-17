# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "argparse",
#     "fastapi",
#     "httpx",
#     "markdownify",
#     "numpy",
#     "semantic_text_splitter",
#     "tqdm",
#     "uvicorn",
#     "google-genai",
#     "pillow",
# ]
# ///

import os
import time
import base64
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig

app = FastAPI()


# ------------------- Rate Limiter -------------------
class RateLimiter:
    def __init__(self, requests_per_minute=60, requests_per_second=2):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.request_times = []
        self.last_request_time = 0

    def wait_if_needed(self):
        current_time = time.time()

        # Per-second rate limiting
        time_since_last = current_time - self.last_request_time
        if time_since_last < (1.0 / self.requests_per_second):
            time.sleep((1.0 / self.requests_per_second) - time_since_last)

        # Per-minute rate limiting
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            current_time = time.time()
            self.request_times = [t for t in self.request_times if current_time - t < 60]

        self.request_times.append(current_time)
        self.last_request_time = current_time


rate_limiter = RateLimiter(requests_per_minute=5, requests_per_second=2)


# ------------------- Pydantic Model -------------------
class QuestionRequest(BaseModel):
    question: str
    image: str | None = None  # base64 image string (optional)


# ------------------- Helper Functions -------------------
def get_embedding(text: str, max_retries: int = 3) -> list[float]:
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            result = client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                time.sleep(2 ** attempt)
            elif attempt == max_retries - 1:
                raise
            else:
                time.sleep(1)
    raise Exception("Max retries exceeded for embedding")


def get_image_description(base64_image: str) -> str:
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    image_data = base64.b64decode(base64_image.split(",")[-1])
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_data)

    uploaded = client.files.upload(file=temp_path)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            uploaded,
            "Describe the content of this image in detail, especially anything that helps answer questions."
        ]
    )

    os.remove(temp_path)
    return response.text


def load_embeddings():
    data = np.load("embeddings.npz", allow_pickle=True)
    return data["chunks"], data["embeddings"]


def generate_llm_response(question: str, context: str) -> str:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    system_prompt = """> You are a knowledgeable and concise teaching assistant. Use only the information provided in the context to answer the question.
>
> * Format your response using **Markdown**.
> * Use code blocks (` ``` `) for any code or command-line instructions.
> * Use bullet points or numbered lists for clarity where appropriate.
> * Always include a brief introduction or heading if needed.
>
> ⚠️ **Important:** If the context does not contain enough information to answer the question, reply exactly with:
>
> ```
> I don't know
> ```
>
> Do not attempt to guess, fabricate, or add external information."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            system_prompt,
            f"Context: {context}",
            f"Question: {question}"
        ],
        config=GenerateContentConfig(
            max_output_tokens=512,
            temperature=0.5,
            top_p=0.95,
            top_k=40
        )
    )
    return response.text


# ------------------- Core Answer Logic -------------------
def answer(question: str, image: str = None):
    chunks, embeddings = load_embeddings()

    if image:
        image_description = get_image_description(image)
        question += f" {image_description}"

    question_embedding = get_embedding(question)

    similarities = np.dot(embeddings, question_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    top_indices = np.argsort(similarities)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_indices]

    response = generate_llm_response(question, "\n".join(top_chunks))
    return {
        "question": question,
        "response": response,
        "top_chunks": top_chunks
    }


# ------------------- FastAPI Endpoint -------------------
@app.post("/api/")
async def api_answer(payload: QuestionRequest):
    try:
        return answer(payload.question, payload.image)
    except Exception as e:
        return {"error": str(e)}


# ------------------- Local Development Run -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("answer:app", host="0.0.0.0", port=10000, reload=True)
