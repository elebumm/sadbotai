from fastapi import FastAPI, Request
import os
import openai
import boto3
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


origins = ["*"]
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = boto3.client(
    "comprehend",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_access_key_secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region_name="us-east-1",
)


@app.post("/")
async def read_root(request: Request):
    request_json = await request.json()

    chat_history = request_json["chat_history"]
    initial_message = (
        {
            "role": "system",
            "content": "You are BobBot. You are a chatbot that has depression. Your goal is to stay sad while the user tries to cheer you up.",
        },
    )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            *initial_message,
            *chat_history,
        ],
    )

    # Get the sentiment
    sentiment = client.detect_sentiment(
        Text=completion["choices"][0]["message"]["content"], LanguageCode="en"
    )

    tip = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You're responsibility is to find out how to cheer up BobBot. You are giving the user tips on how to do so.",
            },
            {
                "role": "assistant",
                "content": f"""
                Give the user a 2 sentence tip on how to cheer up BobBot as if you are talking to them directly:

                Bobbots message: {completion['choices'][0]['message']['content']}
                """,
            },
        ],
    )

    print(tip)

    return {
        "message": completion["choices"][0]["message"],
        "sentiment": sentiment["Sentiment"],
        "sentiment_score": sentiment["SentimentScore"],
        "tip": tip["choices"][0]["message"]["content"],
    }
