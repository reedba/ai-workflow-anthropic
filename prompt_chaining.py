import openai
import os

# Set your OpenAI API key (replace with your actual key or set as environment variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
openai.api_key = OPENAI_API_KEY

# Step 1: Summarize a text
text_to_summarize = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic cognitive functions that humans associate with the human mind, such as learning and problem-solving.
"""

summarize_prompt = f"Summarize the following text in 2-3 sentences:\n{text_to_summarize}"

summary_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": summarize_prompt}
    ],
    max_tokens=100
)
summary = summary_response["choices"][0]["message"]["content"].strip()
print("Summary:\n", summary)

# Step 2: Use the summary to generate follow-up questions
question_prompt = f"Based on the following summary, generate three insightful follow-up questions someone might ask to learn more.\nSummary: {summary}"

questions_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question_prompt}
    ],
    max_tokens=100
)
questions = questions_response["choices"][0]["message"]["content"].strip()
print("\nFollow-up Questions:\n", questions)

# This demonstrates prompt chaining: the output of the first prompt (summary) is used as input for the second prompt (question generation).
