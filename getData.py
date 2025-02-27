import wikipedia
from haystack.dataclasses import Document

def loadData():
    title = "Australia"
    page = wikipedia.page(title=title, auto_suggest=False)
    raw_docs = [Document(content=page.content, meta={"title": page.title, "url":page.url})]
    
    return raw_docs
