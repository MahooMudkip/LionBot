import random, os, time
from haystack.dataclasses import ChatMessage
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import Document
from haystack.components.retrievers import InMemoryEmbeddingRetriever, InMemoryBM25Retriever, SentenceWindowRetriever
# Use ChromaRetriever when scaling?
from haystack_integrations.components.embedders.mistral.document_embedder import MistralDocumentEmbedder
from haystack_integrations.components.embedders.mistral.text_embedder import MistralTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from haystack.components.builders import PromptBuilder, ChatPromptBuilder
from haystack.components.generators.utils import print_streaming_chunk
from haystack.utils import Secret

os.environ["MISTRAL_API_KEY"] = "AArfArswpHyhIsvX3xbBZg66pvioLSz8"
# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["OPENAI_API_KEY"] = "sk-proj-7_pi270Un1AQwP_z5Q_l_T9A0ANNXmaIJseV6-y4GMglZ)jw_aHahyS6jeN1Pq1KsZPzgoNnT4T3BlbkFJ3WOe9dd3Aid7mGZebSpoROvF9fXmQpaii9pLI13EmlSrKmZGQRvsnKvCjWqk3HyFqkxy06B34A"
# os.environ["AZURE_OPENAI_AD_TOKEN"] = ""

rag_pipeline = Pipeline()
document_store = InMemoryDocumentStore()
generator = MistralChatGenerator(model="mistral-small", api_key=Secret.from_env_var("MISTRAL_API_KEY"), streaming_callback=print_streaming_chunk) # or lambda chunk: print(chunk.content, end="", flush=True)
bm25_retriever = InMemoryBM25Retriever(document_store=document_store)

# Sample hardcoded knowledge base
knowledge = [Document(content="Enigma's first album is called \"MCMXC a.D.\", released in 1990"),
                                Document(content="Enigma's second album is called \"The Cross of Changes\", released in 1993"),
                                Document(content="Enigma's third album is called \"Le Roi Est Mort, Vive Le Roi!\", released in 1996"),
                                Document(content="Enigma's fourth album is called \"The Screen Behind the Mirror\", released in 1999"),
                                Document(content="Enigma's fifth album is called \"Voyageur\", released in 2003"),
                                Document(content="Enigma's sixth album is called \"A Posteriori\", released in 2006"),
                                Document(content="Enigma's seventh album is called \"Seven Lives, Many Faces\", released in 2008"),
                                Document(content="Enigma's eigth and final album is called \"The Fall of a Rebel Angel\", released in 2016")]



# Basic pipeline: 
# Query -> Embedder -> Retriever -> PromptBuilder -> Generator -> Response
#             |            |                             |
#             |            |                             V
#             |            V                            LLM
#             V         Vector DB
#     Embedding function

document_embedder = MistralDocumentEmbedder()
documents_with_embeddings = document_embedder.run(knowledge)
document_store.write_documents(documents_with_embeddings["documents"])
text_embedder = MistralTextEmbedder(api_key=Secret.from_env_var("MISTRAL_API_KEY"))
retriever = InMemoryEmbeddingRetriever(document_store=document_store)


# prompt_template = """
#     Answer the following question based on the contents of the article: {{query}}
#     Article: {{documents[0].content}}
# """

prompt_template = """
    Given the following context, answer the question.
    Do not provide an answer to any questions that are not related to the context.
    Context:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    Question: {{query}}
    Answer:
"""

chat_prompt_template = [ChatMessage.from_user(prompt_template)]

# prompt_builder = PromptBuilder(template=prompt_template)
chat_prompt_builder = ChatPromptBuilder(template=chat_prompt_template, variables=["documents"])

# rag_pipeline.add_component("embedder", embedder)
rag_pipeline.add_component("embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", chat_prompt_builder)
rag_pipeline.add_component("generator", generator)

rag_pipeline.connect("embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator.messages")

# rag_pipeline.draw(path="/Users/gn30jo/Library/CloudStorage/OneDrive-ING/Desktop/LionBot/lionbot_prototype.png")

query = ""

print("\nHi! I am the LionBot. Ask me a question!\n\nType \exit to leave at anytime.\n")

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

    # res=rag_pipeline.run({
    #     "embedder": {
    #         "text": query
    #     },
    #     "prompt_builder": {
    #         "query": query
    #     }
    # })

    # print(res)

    res = rag_pipeline.run(
    {
        "embedder": {"text": query},
        "prompt_builder": {"template_variables": {"query": query}, "template": chat_prompt_template}
    })
    print()