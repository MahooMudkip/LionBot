from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack_integrations.components.embedders.mistral.document_embedder import MistralDocumentEmbedder
from haystack.utils import Secret
from getData import loadData

indexing_pipeline = Pipeline()

# Setup storage
document_store = InMemoryDocumentStore()

# Chunking
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=200))

# Document Embedding
embedder = MistralDocumentEmbedder(api_key=Secret.from_token("vYaqkLah5lxb6bmD6ilyoww4IiNqDOVE"), model="mistral-embed")

# add components
indexing_pipeline.add_component(name="embedder", instance=embedder)
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

# connect the components
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

indexing_pipeline.run({"splitter":{"documents":loadData()}})
print(indexing_pipeline)
