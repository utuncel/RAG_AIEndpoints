import argparse
import time
import os
from langchain import hub
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_community.embeddings.ovhcloud import OVHCloudEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.runnables import RunnablePassthrough

_OVH_AI_ENDPOINTS_ACCESS_TOKEN = os.environ.get('OVHCLOUD_API_KEY')

def chat_completion(new_message: str):
  model = ChatMistralAI(model="Mixtral-8x22B-Instruct-v0.1",
                        api_key=_OVH_AI_ENDPOINTS_ACCESS_TOKEN,
                        endpoint='https://mixtral-8x22b-instruct-v01.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/',
                        max_tokens=1500, 
                        streaming=True)

  loader = PyPDFDirectoryLoader(
      path ="./rag-files/",
      glob="**/*",
  )
  pages = loader.load_and_split()

  vectorstore = Chroma.from_documents(documents=pages,
                                      embedding=OVHCloudEmbeddings(model_name="multilingual-e5-base", access_token=_OVH_AI_ENDPOINTS_ACCESS_TOKEN))

  prompt = hub.pull("rlm/rag-prompt")

  rag_chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
  )

  print("🤖: ")
  for r in rag_chain.stream({"question", new_message}):
    print(r.content, end="", flush=True)
    time.sleep(0.150)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--question', type=str)
  args = parser.parse_args()
  chat_completion(args.question)

if __name__ == '__main__':
    main()
