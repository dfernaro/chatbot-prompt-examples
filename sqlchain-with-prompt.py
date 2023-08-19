from langchain import SQLDatabase, SQLDatabaseChain
from langchain.chat_models import AzureChatOpenAI
from config.config import (
    OPENAI_API_BASE,
    OPENAI_API_DEPLOYMENT_NAME,
    OPENAI_API_KEY,
    OPENAI_API_MODEL_NAME,
    OPENAI_API_VERSION,
)

# Initial configuration
db = SQLDatabase.from_uri("sqlite:///chinook.db")
llm = AzureChatOpenAI(
    deployment_name=OPENAI_API_DEPLOYMENT_NAME,
    model_name=OPENAI_API_MODEL_NAME,
    openai_api_base=OPENAI_API_BASE,
    openai_api_version=OPENAI_API_VERSION,
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    verbose=True,
)
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True, top_k=3)

# Prompts
PROMPT_VERSION_1 = """
Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

{question}
"""

PROMPT_VERSION_2 = """
Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Additional Information:
- Humans means users

{question}
"""

PROMPT_VERSION_3 = """
Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Additional Information:
- Humans means users

Restrictions:
- Customers table is not accessible

{question}
"""

# Questions
try:
    question = PROMPT_VERSION_1.format(question="How many users are there?")
    print(db_chain.run(question))
except Exception as e:
    print(e)
