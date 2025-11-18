embedding_id = uuid.uuid1()
embedding = OpenAiEmbeddings().embed(text)
return Vector(embedding_id=embedding_id, embedding=embedding)