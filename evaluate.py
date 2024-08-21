from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from ragas.metrics import answer_similarity
from ragas import evaluate

EVALUATE_MODEL = "llama3-8b-8192"
EMBEDDING_MODEL = "all-MiniLM-L12-v2"
LLM_ANSWER_FILE = "answers.json"
EVAL_SAVE_FILE = "evaluation_result.csv"

llm = ChatGroq(model=EVALUATE_MODEL, temperature=0)
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL, 
    model_kwargs={
        'trust_remote_code':True,
        'device': 'cuda'
    }
)


dataset = Dataset.from_json(LLM_ANSWER_FILE)
result = evaluate(
    dataset,
    metrics=[answer_similarity],
    llm=llm,
    embeddings=embeddings
)
df = result.to_pandas()
df.to_csv(EVAL_SAVE_FILE, index=False, sep="\t")