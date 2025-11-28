import os
from groq import Groq

# Set your API key (keep it SECRET)
os.environ["GROQ_API_KEY"] = "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"

# Initialize client
client = Groq(api_key=os.environ["GROQ_API_KEY"])
print("hi")
def real_llm(prompt):
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    return completion.choices[0].message.content

real_llm("Say 'Hello Darshan!' in JSON.")

