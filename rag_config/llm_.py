from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import os
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
# langsmith


def load_llm(model='llama'):
    load_dotenv()
    open_ai_key = os.getenv("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG_ID")
    groq_api = os.getenv("GROQ_API_KEY")
    print(f"Groq Key: {groq_api}")
    if model == 'llama':
        active_model = ChatOllama(
                    base_url="http://ollama:11434",  # Ollama server endpoint
                    model="llama3.1:8b",
                    temperature=0,
                )
    elif model == 'llama3.1':
        active_model = ChatOllama(
                    base_url="http://ollama:11434",  # Ollama server endpoint
                    model="llama3.1:70b",
                    temperature=0,
                )
    elif model == 'gpt4':
        active_model = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            timeout=None,
            openai_api_key=open_ai_key,
            organization=org_id
        )
    elif model == 'gpt-4o':
        active_model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            timeout=None,
            openai_api_key=open_ai_key,
            organization=org_id
        )
    elif model == 'gpt-4o-mini':
        active_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            timeout=None,
            openai_api_key=open_ai_key,
            organization=org_id
        )
    elif model == 'gpt3.5':
        active_model = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0,
            timeout=None,
            max_retries=2,
            openai_api_key=open_ai_key,
            organization=org_id
        )
    elif model == 'gemma':
        active_model = ChatGroq(temperature=0,groq_api_key=groq_api, model="gemma-7b-it",max_retries=3)
    elif model == 'mixtral':
        active_model = ChatGroq(temperature=0,groq_api_key=groq_api, model="mixtral-8x7b-32768",max_retries=3)
    else:
        active_model = ChatOllama(
                    base_url="http://ollama:11434",  # Ollama server endpoint
                    model="llama3.1:70b",
                    temperature=0,
                )

    return active_model



BASE_PROMPT_0="""Read the following abstracts from PubMed:{context} 
                Given your INTERNAL CLINICAL AND MEDICALDOMAIN KNOWLEDGE and provided abstracts, is the following claim true or false? 
                Claim: {question}
                if the abstract does not provide enough information to answer the question, search in medical databases and search engines.
                if the claim is true, select "True". If the claim is false, select "False". If the claim cannot be substantiated, select "Cannot be substantiated".
                Explain your answer by providing evidence from the context that supports your reasoning.
                """
def ask_llm(question,context, model='llama'):
    try:

        BASE_PROMPT=f"""You are a very smart physician specialízing in drugs and their efficacy and indications, when prompted with a drug fact you will check and justify the truth value of the fact using context given to you combined with your immense knowledge that the stated information is merely a factual assertion.
        Regarding drugs and evidence methods regarding their mechanisms of action, clinical trials and evidence based medicine.
        STATED FACT:{question}. Fact check the fact with justification for your opinion using the following lines of evidence and have a seperate section for each of these:
        -Mechanisms of action (go into detail for mechanisms of action if possible)
        Evidence-based medicine-comparisons with other treatments: Provide references to clinical trials or studies that compare the drug in question to other treatments.
        Use the given context {context} as your foundation of knowledge for the justification if you cannot find the information within the given context you MUST USE YOUR VAST INTERNAL KNOWLEDGE base and go into detail don't just state facts provide an in-depth justification(e.g.. point to mechavare al, clinical trials you know) tor that given section but state you used your internal knowledge base for this eVidence category.
        BUT DON'T HALLUCINATE If you do not know state this and do not justitly using that evidence you are a medical professional and they don't lie 
        #Extra instruction: DO THE FOLLOWING FIRST!
        ## Relevant Problems: Recall one example if possible of similar justification problems that are relevant to the initial problem: {question}.
        ## Solve the Initial Problem:
        Q: Copy and paste the initial problem here.
        A: Explain the solution for each separate category as listed above using the context given and if needed your internal knowledge base then enclose the ultimate answer in Fact) here. In the end, one sentence to summarize your justification outcome and state the truth value for the tact here YOU MUST GIVE A TRUTH VALUE FOR THE STATED FACT{question} using the evidence to reason the truth of the supposed fact.
        """
        system = "You are a helpful assistant expert in medical domain"
        template = ChatPromptTemplate.from_messages([("system", system), ("human", BASE_PROMPT)], template_format='mustache')
        # print(template.format_messages())   
        chain = template | model     
        result = chain.invoke({"question": question,"context":context})
        print(f"type of result: {type(result)}")
        result =  result.content
        print(result)
        return result
    except Exception as e:
        print(f"Error in asking llm: {e}")
        raise e