import ast
import openai
import pandas as pd
import tiktoken
from scipy import spatial

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
openai.api_key = "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" #this key can be obtained from  https://openai.com/ 

embeddings_path = "summer_olympics_2008.csv"

df = pd.read_csv(embeddings_path)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

def strings_ranked_by_relatedness(query, df, relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),top_n = 100):
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text, model = GPT_MODEL):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(query, df, model, token_budget):
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the dataset on the 2008 Summer Olympics to answer the subsequent question. If the answer cannot be found in the dataset, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (num_tokens(message + next_article + question, model=model) > token_budget):
            break
        else:
            message += next_article
    return message + question


def ask(query, df = df, model = GPT_MODEL, token_budget = 4096 - 50, print_message = False):
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the 2008 Summer Olympics."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.5
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

ask('How many records were set at the 2008 Summer Olympics?')
ask('Which country won the most gold MEDALS')
ask('Which Canadian competitor won the frozen hot dog eating competition?')