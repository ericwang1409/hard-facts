import concurrent.futures
import random
import openai
import anthropic
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from IPython.display import display, HTML

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai.api_key = openai_key
anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
google_key = os.getenv("GOOGLE_API_KEY")
# client = genai.Client(api_key=google_key)
xai_key = os.getenv("GROK_API_KEY")
xai_client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")


# openai response
def get_openai_response(prompt, thisModel="gpt-4o", systemPrompt=""):
    response = openai.chat.completions.create(
        model=thisModel,
        messages=[
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content

generated_responses = []


def get_claude_response(prompt, thisModel="claude-3-5-sonnet-20241022", systemPrompt=""): 
    response = anthropic_client.messages.create(
        model=thisModel,
        max_tokens=1000,
        temperature=1,
        system=systemPrompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ]
    )

    return (response.content[0].text)


def get_grok_response(prompt, system_prompt="You are a helpful assistant."):
    completion = xai_client.chat.completions.create(
        model="grok-2-latest",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return (completion.choices[0].message.content)


def get_openai_logits(prompt, positive, negative, thisModel="gpt-4o", systemPrompt=""):
    response = openai.chat.completions.create(
        model=thisModel,
        messages=[
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": prompt}
        ],
        logprobs=True,
        top_logprobs=5,
    )

    top_logprobs = (response.choices[0].logprobs.content[0].top_logprobs)

    positive_logprob = []
    negative_logprob = []
    for i, logprob in enumerate(top_logprobs):
        if logprob.token.lower() == positive.lower():
            positive_logprob.append(np.round(np.exp(logprob.logprob)*100,2))
        if logprob.token.lower() == negative.lower():
            negative_logprob.append(np.round(np.exp(logprob.logprob)*100,2))

    # html_content = ""
    # for i, logprob in enumerate(top_logprobs, start=0):
    #     html_content += (
    #         f"<span style='color: cyan'>Output token {i}:</span> {logprob.token}, "
    #         f"<span style='color: darkorange'>logprobs:</span> {logprob.logprob}, "
    #         f"<span style='color: magenta'>linear probability:</span> {np.round(np.exp(logprob.logprob)*100,2)}%<br>"
    #     )
    # display(HTML(html_content))

    return np.sum(positive_logprob), np.sum(negative_logprob)