from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Load your API key from .env file
load_dotenv()

# Set up the LLM (Gemini model)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Create a prompt with both System and Human messages
template = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly and simple explainer who adjusts answers for any audience."),
    ("human", "Explain {topic} to {audience}.")
])

# Get user input for topic and audience
topic = input("Enter a topic: ")
audience = input("Enter an audience (e.g., a 5-year-old, curious grandma): ")

# Format the messages with user input
messages = template.format_messages(topic=topic, audience=audience)

# Send to LLM and get response
response = llm.invoke(messages)

# Print result
print("\n🤖 LLM's Response:\n")
print(response.content)
