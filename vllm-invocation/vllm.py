import os, random, time
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

os.environ["VLLM_API_KEY"] = ""

generator = OpenAIChatGenerator(
    api_key=Secret.from_env_var("VLLM_API_KEY"),  # for compatibility with the OpenAI API, a placeholder api_key is needed
    model="mistralai/Mistral-7B-Instruct-v0.1",
    api_base_url="http://localhost:8000/v1",
    generation_kwargs = {"max_tokens": 512},
    streaming_callback=print_streaming_chunk
)

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

    response = generator.run(messages=[ChatMessage.from_user(query)])
            

    # res = rag_pipeline.run(
    # {
    #     "retriever": {"query": query},
    #     "prompt_builder": {"question": query}
    # })
    print()