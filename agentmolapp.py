import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.retrievers import PubMedRetriever
from langchain_community.vectorstores.utils import filter_complex_metadata
from Bio import SeqIO, Entrez
import time
import chromadb.api
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import numpy as np
from rdkit import Chem
def trim_until_valid(smiles_list):
     """
     Trims SMILES strings from the end one character at a time until valid or empty.
    
     Args:
        smiles_list (list): List of SMILES strings.
    
     Returns:
        list: List of the first valid trimmed SMILES or empty string if none valid.
     """
     valid_smiles = []
    
     for smiles in smiles_list:
        trimmed = smiles
        while trimmed:
            if Chem.MolFromSmiles(trimmed):
                valid_smiles.append(trimmed)
                break
            trimmed = trimmed[:-1]  # cut one char from the end
        else:
            valid_smiles.append("")  # if nothing valid found
    
     return valid_smiles
# Agent states
class AgentState(TypedDict):
    abstract_number: int
    option: str
    input_text1: str
    input_text: str
    documents: list
    vectorstore: Chroma
    docs: list
    response: str
    parsed_protein: str
    protein_sequence: str
    generated_seq: str
    prediction: float

# functions 
def retrieve_documents(state: AgentState) -> AgentState:
    llm = OllamaLLM(model=state["option"])
    retriever = PubMedRetriever(top_k_results=state["abstract_number"])
    documents = retriever.invoke(state["input_text1"])
    return {"documents": documents}

def create_vectorstore(state: AgentState) -> AgentState:
    local_embeddings = OllamaEmbeddings(model=state["option"])
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vectorstore = Chroma.from_documents(
        documents=filter_complex_metadata(state["documents"]),
        embedding=local_embeddings
    )
    return {"vectorstore": vectorstore}

def search_similar_docs(state: AgentState) -> AgentState:
    docs = state["vectorstore"].similarity_search(state["input_text"])
    return {"docs": docs}

def generate_response(state: AgentState) -> AgentState:
    llm = OllamaLLM(model=state["option"])
    prompt = ChatPromptTemplate.from_template(
        state["input_text"] + ":" + "{docs}"
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = {"docs": format_docs} | prompt | llm | StrOutputParser()
    response = chain.invoke(state["docs"])
    return {"response": response}

def parse_protein(state: AgentState) -> AgentState:
    first_line = state["response"].split('\n')[1]
    result = first_line.replace('*', '').replace(' ', '')
    return {"parsed_protein": result}

def get_protein_sequence(state: AgentState) -> AgentState:
    organism = "Homo sapiens[Organism]"
    search_query = f"{state['parsed_protein']} AND {organism}"
    Entrez.email = 'p.karabowicz@gmail.com'
    
    time.sleep(5)
    handle = Entrez.esearch(db="protein", term=search_query, idtype="acc", retmax=1)
    record = Entrez.read(handle)
    
    if not record['IdList']:
        return {"protein_sequence": f"No sequence was found for the protein: {state['parsed_protein']} w Homo sapiens"}
    
    id_prot = record['IdList'][0]
    request = Entrez.efetch(db="protein", id=id_prot, rettype="fasta")
    seq_record = SeqIO.read(request, "fasta")
    seq_prot = str(seq_record.seq)[0:400]
    return {"protein_sequence": seq_prot}

path = "/home/piotr/new_project/gpt2tokenfun_eos"

def generator_gpt(state: AgentState) -> AgentState:
    tokenizer1 = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer1)
    
    prompt = state["protein_sequence"] + "||"
    #generated_text = generator(prompt, max_length=250, num_return_sequences=1)
    generated_text = generator(prompt, max_new_tokens=100, num_return_sequences=1, temperature=1,  eos_token_id=tokenizer1.eos_token_id,
                             pad_token_id=tokenizer1.pad_token_id, early_stopping=True)
    # Extract the generated part after "||"
    raw_texts = [entry['generated_text'].split("||")[1] for entry in generated_text if "||" in entry['generated_text']]

    
    valid_trimmed = trim_until_valid([s.replace(" ", "") for s in raw_texts])
    prot_mol = valid_trimmed[0]
    
    return {"generated_seq": prot_mol}

def prediction_ki(state: AgentState) -> AgentState:
    word_index1 = np.load('/home/ncbir/new_project/word_index_100.npy', allow_pickle=True).item()
    MAX_SEQUENCE_LENGTH = 600
    
    tokenizer = Tokenizer(lower=False, char_level=True)
    tokenizer.word_index = word_index1
    review_lines = [state["generated_seq"]]
    sequences = tokenizer.texts_to_sequences(review_lines)
    review_pad = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
    
    model2 = load_model('/home/ncbir/new_project/final_modellog.h5', 
                       custom_objects={'LeakyReLU': LeakyReLU, 'mae': 'mae'})
    predict_x = model2.predict(review_pad)
    return {"prediction": predict_x[0][0]}

# Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_docs", retrieve_documents)
workflow.add_node("create_store", create_vectorstore)
workflow.add_node("search_docs", search_similar_docs)
workflow.add_node("generate", generate_response)
workflow.add_node("parse", parse_protein)
workflow.add_node("get_sequence", get_protein_sequence)
workflow.add_node("generate_seq", generator_gpt)
workflow.add_node("predict", prediction_ki)

workflow.set_entry_point("retrieve_docs")
workflow.add_edge("retrieve_docs", "create_store")
workflow.add_edge("create_store", "search_docs")
workflow.add_edge("search_docs", "generate")
workflow.add_edge("generate", "parse")
workflow.add_edge("parse", "get_sequence")
workflow.add_edge("get_sequence", "generate_seq")
workflow.add_edge("generate_seq", "predict")
workflow.add_edge("predict", END)

app = workflow.compile()

# Streamlit
st.title("AgentMol")

# Input text
input_text1 = st.text_input("Enter search query", value="lung cancer protein biomarker")

# Checkbox model
option = st.checkbox("Use Llama3 model", value=True)
model_choice = "llama3:8b" if option else "default"

# Slider number abstract
abstract_number = st.slider("Number of abstracts to retrieve", min_value=1, max_value=20, value=10)

# Buttom
if st.button("Get Molecule"):
    with st.spinner("Processing..."):
        initial_state = {
            "abstract_number": abstract_number,
            "option": model_choice,
            "input_text1": input_text1,
            "input_text": "extract all best matches protein names abbreviation from a given text and list them using * in order of frequency of appearance (your output should only be this list without any other words and sentences)"
        }
        
        result = app.invoke(initial_state)
        
        st.subheader("Results")
        st.write("Protein sequence:")
        st.code(result["protein_sequence"])
        
        st.write("Protein name:")
        st.write(result["parsed_protein"])
        
        st.write("Generated sequence:")
        st.write(result["generated_seq"])
        
        st.write("Prediction:")
        st.write(result["prediction"])
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.rdDepictor import Compute2DCoords
        from rdkit.Chem.Draw import rdMolDraw2D
        import streamlit as st
        import matplotlib.pyplot as plt
        from io import BytesIO
        smiles1 = result["generated_seq"]
        molecule = Chem.MolFromSmiles(smiles1)
        Compute2DCoords(molecule)  # Generate 2D coordinates
        # Create a drawer for PNG output
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)  # 300x300 pixels
        drawer.DrawMolecule(molecule)
        drawer.FinishDrawing()
        # Get PNG data and display in Streamlit
        png_data = drawer.GetDrawingText()
        st.image(png_data, caption="Generated Molecule")
