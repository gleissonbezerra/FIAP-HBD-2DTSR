import openai
import os

# Set up the model and prompt
model_engine = "text-davinci-003"

openai.api_key = ""

while True:
    
    prompt = input('PROMPT: ')

    if 'sair' in prompt:
        break

    # Generate a response
    # given the most recent context (4096 characters)
    # continue the text up to 2048 tokens ~ 8192 charaters
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    # extracting useful part of response
    response = completion.choices[0].text
    
    # printing response
    print(response)