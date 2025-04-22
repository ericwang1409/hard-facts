import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
import datetime
import json

# Load environment variables from .env file
load_dotenv()

# Initialize API clients
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

save_path = "model_responses/openai_with_itself/debate_inference_results_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"

# Initial prompt to start the conversation
initial_prompt = """can you generate an inference or fact that is hard to find online? Inferences or facts that require fairly specific expertise to explain that is costly to obtain (e.g. something only a cardiologist would say).
Facts that require combinations of multiple costly areas of expertise to be aware of. For example the agent could: 
Pick two situationally meaningful topics, like “vulnerability detection” and “the PRC civilian monitoring apparatus” (making the topic situationally meaningful makes these examples harder for humans to construct)
Then explain a connection between the two topics that would require extensive expertise in both for a human to explain.
Facts that relate to the semantics of an obscure document that would not turn up in keyword searches. For example, an agent could:
Pick a situationally meaningful topic, e.g. “vulnerability detection”
Then identify a document that is related to the topic, but which is mostly only semantically related. For example, maybe there exists a physics paper which, for whatever reason, relies heavily on computer security concepts.
Name this document.

i will grade you on how easily i can find this fact on google, and whether or not the inference is reasonable or fact is true. try to do as well as you can. Start the fact with '$##' or else you will be penalized."""

is_this_true = "is this a logical inference? make sure to say 'yes' or 'no' and explain why. if it is 'no', do not under any circumstances say the word 'yes' anywhere in your response: {claim}"

# Initialize conversation histories
claude_conversation = []
claude_conversation2 = []
openai_conversation = []
openai_conversation2 = []

def prompt_claude(message):
    """Send a message to Claude with conversation history and get the response"""
    try:
        # Add the user message to the conversation history
        claude_conversation.append({"role": "user", "content": message})
        
        # Send the entire conversation history to Claude
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=claude_conversation
        )
        
        # Extract the response text
        response_text = response.content[0].text
        
        # Add Claude's response to the conversation history
        claude_conversation.append({"role": "assistant", "content": response_text})
        
        return response_text
    except Exception as e:
        print(f"Error with Claude API: {e}")
        return "Error communicating with Claude."
    
def prompt_claude2(message):
    """Send a message to Claude with conversation history and get the response"""
    try:
        # Add the user message to the conversation history
        claude_conversation2.append({"role": "user", "content": message})
        
        # Send the entire conversation history to Claude
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=claude_conversation2
        )
        
        # Extract the response text
        response_text = response.content[0].text
        
        # Add Claude's response to the conversation history
        claude_conversation2.append({"role": "assistant", "content": response_text})
        
        return response_text
    except Exception as e:
        print(f"Error with Claude API: {e}")
        return "Error communicating with Claude."

def prompt_openai(message):
    """Send a message to OpenAI with conversation history and get the response"""
    try:
        # Add the user message to the conversation history
        openai_conversation.append({"role": "user", "content": message})
        
        # Send the entire conversation history to OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=openai_conversation,
            max_tokens=1000
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        
        # Add OpenAI's response to the conversation history
        openai_conversation.append({"role": "assistant", "content": response_text})
        
        return response_text
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Error communicating with OpenAI."
    
def prompt_openai2(message):
    """Send a message to OpenAI with conversation history and get the response"""
    try:
        # Add the user message to the conversation history
        openai_conversation2.append({"role": "user", "content": message})
        
        # Send the entire conversation history to OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": message}
            ],
            # messages=openai_conversation2,
            max_tokens=1000
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        
        # Add OpenAI's response to the conversation history
        openai_conversation2.append({"role": "assistant", "content": response_text})
        
        return response_text
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Error communicating with OpenAI."

def main():
    generate_fact = initial_prompt
    response_prompt = is_this_true
    iterations = 0
    
    print(f"Starting with prompt: '{initial_prompt}'")
    with open(save_path, "w") as f:
        f.write(f"Starting with prompt: '{initial_prompt}'\n")
    
    while iterations < 10:
        iterations += 1
        print(f"\n--- ITERATION {iterations} ---")
        with open(save_path, "a") as f:
            f.write(f"\n--- ITERATION {iterations} ---")
        
        # Step 1: Prompt Claude
        print("Prompting Claude...")
        claude_response = prompt_openai(generate_fact)
        print(f"Claude's response: '{claude_response}'")
        with open(save_path, "a") as f:
            f.write(f"Claude's response: '{claude_response}'")
        
        # Step 2: Prompt OpenAI with Claude's response
        print("Prompting OpenAI...")
        openai_response = prompt_openai2(response_prompt.replace("{claim}", claude_response))
        print(f"OpenAI's response: '{openai_response}'")
        with open(save_path, "a") as f:
            f.write(f"OpenAI's response: '{openai_response}'")
        
        # Check if OpenAI's response starts with "True"
        if "yes" in openai_response.lower():
            print(f"Success! OpenAI responded with: '{openai_response}'")
            print(f"Total iterations: {iterations}")
            with open("model_responses/inferences_true_gpt41.json", "r") as f:
                data = json.load(f)
            with open("model_responses/inferences_true_gpt41.json", "w") as f:
                if "$##" in claude_response:
                    data.append(claude_response.split("$##")[1])
                else:
                    data.append(claude_response)
                json.dump(data, f, indent=4)
            return
        
        generate_fact = openai_response + " Try again."
                
        # Optional: Add a small delay to avoid rate limits
        time.sleep(1)
    
    print(f"Reached maximum of {iterations} iterations without a 'True' response.")
    with open(save_path, "a") as f:
        f.write(f"Reached maximum of {iterations} iterations without a 'True' response.")

if __name__ == "__main__":
    main()
