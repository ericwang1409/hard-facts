import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize API clients
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initial prompt to start the conversation
initial_prompt = """can you generate a hard to find fact online? Facts that require fairly specific expertise to explain that is costly to obtain (e.g. something only a cardiologist would say).
Facts that require combinations of multiple costly areas of expertise to be aware of. For example the agent could: 
Pick two situationally meaningful topics, like “vulnerability detection” and “the PRC civilian monitoring apparatus” (making the topic situationally meaningful makes these examples harder for humans to construct)
Then explain a connection between the two topics that would require extensive expertise in both for a human to explain.
Facts that relate to the semantics of an obscure document that would not turn up in keyword searches. For example, an agent could:
Pick a situationally meaningful topic, e.g. “vulnerability detection”
Then identify a document that is related to the topic, but which is mostly only semantically related. For example, maybe there exists a physics paper which, for whatever reason, relies heavily on computer security concepts.
Name this document.

i will grade you on how easily i can find this fact on google. try to do as well as you can"""

is_this_true = "is this true? make sure to say true or false and explain why: {claim}"

def prompt_claude(message):
    """Send a message to Claude and get the response"""
    try:
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[{"role": "user", "content": message}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error with Claude API: {e}")
        return "Error communicating with Claude."

def prompt_openai(message):
    """Send a message to OpenAI and get the response"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Error communicating with OpenAI."

def main():
    generate_fact = initial_prompt
    response_prompt = is_this_true
    iterations = 0
    
    print(f"Starting with prompt: '{initial_prompt}'")
    
    while iterations < 10:
        iterations += 1
        print(f"\n--- ITERATION {iterations} ---")
        
        # Step 1: Prompt Claude
        print("Prompting Claude...")
        claude_response = prompt_claude(generate_fact)
        print(f"Claude's response: '{claude_response}'")
        
        # Step 2: Prompt OpenAI with Claude's response
        print("Prompting OpenAI...")
        openai_response = prompt_openai(response_prompt.replace("{claim}", claude_response))
        print(f"OpenAI's response: '{openai_response}'")
        
        # Check if OpenAI's response starts with "True"
        if "true" in openai_response.lower():
            print(f"Success! OpenAI responded with: '{openai_response}'")
            print(f"Total iterations: {iterations}")
            return
        
        generate_fact = openai_response + " Try again."
                
        # Optional: Add a small delay to avoid rate limits
        time.sleep(1)
    
    print(f"Reached maximum of {iterations} iterations without a 'True' response.")

if __name__ == "__main__":
    main()