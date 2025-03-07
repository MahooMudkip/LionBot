import os
import random
import time
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from indexing_pipeline import document_store

os.environ["MISTRAL_API_KEY"] = "AArfArswpHyhIsvX3xbBZg66pvioLSz8"

retriever = InMemoryBM25Retriever(document_store=document_store)
generator = MistralChatGenerator(model="mistral-small", api_key=Secret.from_env_var("MISTRAL_API_KEY"), streaming_callback=print_streaming_chunk)

template = """
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Please answer the question based on the given information from local documentation.
Do not answer any questions that are not related to the context or the content.

{{question}}
"""

message = [ChatMessage.from_user(template)]

prompt_builder = ChatPromptBuilder(template=message)

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)

rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

query = ""

print("\nHi! I am the LionBot. Ask me a question!\n\n[Type \exit to leave at anytime.]")

while True:
    query = input("\nQuery: ")

    if query == "\exit":
        print("Thanks for chatting! See you next time!\n")
        break
    elif query == "":
        reply = random.randint(0,3)
        if reply == 0:
            print("That's not a query...\n")
        elif reply == 1:
            print("You do know how to ask a question, right?\n")
        elif reply == 2:
            print("You're not fooling me with that...\n")
        elif reply == 3:
            print("You think an empty query is funny? Watch this.\n")
            time.sleep(3)
            print("\n"*25)
            print("Do it again and I'll flush your screen even more.\n")
        continue
            

    res = rag_pipeline.run(
    {
        "retriever": {"query": query},
        "prompt_builder": {"question": query}
    })
    print()