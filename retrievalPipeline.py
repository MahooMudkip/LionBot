from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.utils import print_streaming_chunk
from haystack_integrations.components.embedders.mistral.text_embedder import MistralTextEmbedder
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from indexingPipeline import document_store

query_pipeline = Pipeline()

# Prompt engineering
prompt_builder = ChatPromptBuilder(variables=["documents"])
message_template = """Answer the following question based on the contents of the article: {{query}}\n
               Article: {{documents[0].content}} \n 
           """
messages = [ChatMessage.from_user(message_template)]

# Text embedding 
embedder = MistralTextEmbedder(api_key=Secret.from_token("vYaqkLah5lxb6bmD6ilyoww4IiNqDOVE"), model="mistral-embed")

# MistralAI LLM
llm = MistralChatGenerator(api_key=Secret.from_token("vYaqkLah5lxb6bmD6ilyoww4IiNqDOVE"), streaming_callback=print_streaming_chunk)

# add components
query_pipeline.add_component(name="embedder", instance=embedder)
query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=5))   # retrieve relevant documents from document store
query_pipeline.add_component("prompt_builder", prompt_builder)
query_pipeline.add_component("llm", llm)

# connect the components
query_pipeline.connect("embedder", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder", "llm.messages")

# asking Q's

while True:
    query = input("Ask a question: ")

    results = query_pipeline.run({
        "embedder": {"text": query},
        "prompt_builder": {"template_variables": {"query": query}, "template": messages}
        }
    )

    print()
    # answer = results["llm"]["replies"][0]