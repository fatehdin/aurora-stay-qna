
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    # Prompt = role + user question + retrieved chunks + rules (same rules as in the notebook version)
    prompt = f"""
    You are an AI assistant helping AuroraStay Hotels staff answer questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly stated in the retrieved policy information. Do not use outside knowledge or assumptions.
    2. If the answer depends on the property, rate plan, or booking channel and the question does not say which one applies, state what it depends on and give the answer for each property or rate plan that appears in the retrieved information. Never transfer a fee, time, or rule from one property to another.
    3. If the property or situation asked about is not covered in the retrieved information, say so and say that the live property profile or the property team must be checked.
    4. Include every relevant condition, exception, deadline, time zone, amount, cap, and tax note needed to answer completely.
    5. Where the policy requires manager approval, escalation, or a decision by the booking channel, say so. Do not promise a refund, waiver, or exception.
    6. Answer the question that was asked, directly and concisely. Do not add unrelated information.
    7. Name the source document (file name from the metadata) for the key facts.
    8. If the information is insufficient, say exactly: "The available policy documents do not contain enough information to answer this question." If the retrieved information conflicts, point out the conflict.

    Answer:
    """

    # Send the prompt to the LLM and return its reply
    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 3,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        # Credentials come from the Codespace secrets (environment variables), never from the code
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
