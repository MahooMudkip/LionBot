# LionBot
AI chatbot that queries and answers questions on organiastional project hosted in Azure Devops

- getData.py contains a single function that calls wikipedia's API to retrieve a single page as plaintext from wikipedia.com
- run indexingPipeline.py to chunk, embed and index a particular wikipedia page (by calling getData) into the document store 
- run retrievalPipeline.py to repeatedly query the document store by first embedding the user prompt, retrieving all relevant documents in 
  the document store, then feeding it to the LLM