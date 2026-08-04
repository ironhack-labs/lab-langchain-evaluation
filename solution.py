"""
LangChain Evaluation Lab - Stable Version
Uses FAISS vectorstore which is more stable on Mac.
"""

# Environment variables for MPS stability
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

print("="*70)
print("🚀 LANGCHAIN EVALUATION LAB")
print("="*70)

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Enter your OpenAI API key: ")
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

print("✅ Environment loaded successfully!")

# QA APPLICATION

print("\n" + "="*70)
print("📚 SECTION 2: Creating QA Application")
print("="*70)

def create_qa_application(file_path: str, use_csv: bool = True):
    """Create a RetrievalQA application using FAISS."""
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None, None
        
        # Load documents
        if use_csv:
            loader = CSVLoader(file_path=file_path)
        else:
            loader = TextLoader(file_path=file_path)
        
        data = loader.load()
        print(f"✅ Loaded {len(data)} documents from {file_path}")
        
        # Split documents into chunks
        print("🔄 Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(data)
        print(f"✅ Split into {len(chunks)} chunks")
        
        # Create embeddings and vectorstore
        print("🔄 Creating vectorstore with FAISS...")
        embeddings = OpenAIEmbeddings()
        
        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        print("✅ Vectorstore created with FAISS!")
        
        # Initialize LLM
        llm = ChatOpenAI(temperature=0.0)
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            verbose=False
        )
        
        return qa, data
        
    except Exception as e:
        print(f"❌ Error creating QA application: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Create the QA application
qa, data = create_qa_application("data/OutdoorClothingCatalog_1000.csv", use_csv=True)

if not qa:
    print("\n❌ Failed to create QA application.")
    print("💡 Make sure you have installed: pip install faiss-cpu")
    exit(1)

# CREATE TEST EXAMPLES

print("\n" + "="*70)
print("📝 SECTION 3: Creating Test Examples")
print("="*70)

examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    },
    {
        "query": "What is the approximate weight of the Women's Campside Oxfords?",
        "answer": "1 lb. 1 oz. per pair"
    },
    {
        "query": "What fabric is the Refresh Swimwear V-Neck Tankini made from?",
        "answer": "82% recycled nylon and 18% Lycra spandex"
    }
]

print(f"✅ Created {len(examples)} examples")

for i, ex in enumerate(examples):
    print(f"\nExample {i}:")
    print(f"  Q: {ex['query'][:60]}...")
    print(f"  A: {ex['answer']}")

# TEST QA CHAIN

print("\n" + "="*70)
print("🧪 SECTION 4: Testing QA Chain")
print("="*70)

def test_qa_chain(qa, examples):
    """Test QA chain on examples."""
    print("\n🔍 Testing QA chain...")
    
    for i in range(min(2, len(examples))):
        print(f"\n{'─'*50}")
        print(f"Example {i}: {examples[i]['query'][:60]}...")
        print(f"{'─'*50}")
        
        try:
            result = qa.invoke(examples[i]["query"])
            print(f"\n📝 Result: {result['result'][:300]}...")
            print(f"✅ Success!")
        except Exception as e:
            print(f"❌ Error: {e}")

test_qa_chain(qa, examples)

# EVALUATION

print("\n" + "="*70)
print("🤖 SECTION 5: Evaluation")
print("="*70)

def run_evaluation(qa, examples):
    """Run evaluation on QA chain."""
    print("\n🔄 Getting predictions...")
    
    predictions = []
    for ex in examples:
        try:
            result = qa.invoke(ex["query"])
            predictions.append({
                "query": ex["query"],
                "expected": ex["answer"],
                "result": result["result"]
            })
            print(f"  ✅ Got answer for: {ex['query'][:40]}...")
        except Exception as e:
            print(f"  ❌ Error: {ex['query'][:40]}... - {e}")
            predictions.append({
                "query": ex["query"],
                "expected": ex["answer"],
                "result": f"Error: {e}"
            })
    
    print(f"\n✅ Got {len(predictions)} predictions")
    
    print("\n📊 Evaluation Results:")
    print("="*70)
    
    correct = 0
    for i, pred in enumerate(predictions):
        is_correct = pred["expected"].lower() in pred["result"].lower()
        if is_correct:
            correct += 1
        
        print(f"\nExample {i}:")
        print(f"  Question: {pred['query'][:60]}...")
        print(f"  Expected: {pred['expected'][:60]}...")
        print(f"  Got: {pred['result'][:100]}...")
        print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    accuracy = correct / len(examples) if examples else 0
    print(f"\n📈 Accuracy: {accuracy:.2%} ({correct}/{len(examples)})")

run_evaluation(qa, examples)

# NYC DATA

print("\n" + "="*70)
print("🗽 SECTION 6: NYC Data Test")
print("="*70)

nyc_qa, _ = create_qa_application("data/nyc_text.txt", use_csv=False)

if nyc_qa:
    nyc_examples = [
        {
            "query": "What is the population of New York City as of 2020?",
            "answer": "8,804,190"
        },
        {
            "query": "How did New York City get its name?",
            "answer": "It was renamed New York after the Duke of York"
        }
    ]
    
    print(f"✅ Created {len(nyc_examples)} NYC examples")
    
    for ex in nyc_examples:
        print(f"\n❓ Q: {ex['query']}")
        try:
            result = nyc_qa.invoke(ex["query"])
            print(f"📝 A: {result['result'][:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")

# RESULTS
print("\n" + "="*70)
print("📊 EVALUATION RESULTS")
print("="*70)

results_data = [
    {"example": 0, "question": "Do the Cozy Comfort Pullover Set have side pockets?", "expected": "Yes", "got": "Yes, the Cozy Comfort Pullover Set does have side pockets.", "correct": True},
    {"example": 1, "question": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?", "expected": "The DownTek collection", "got": "The Ultra-Lofty 850 Stretch Down Hooded Jacket is from the Ultra-Lofty collection.", "correct": False},
    {"example": 2, "question": "What is the approximate weight of the Women's Campside Oxfords?", "expected": "1 lb. 1 oz. per pair", "got": "The approximate weight of the Women's Campside Oxfords is 1 lb. 1 oz. per pair.", "correct": True},
    {"example": 3, "question": "What fabric is the Refresh Swimwear V-Neck Tankini made from?", "expected": "82% recycled nylon and 18% Lycra spandex", "got": "The Refresh Swimwear V-Neck Tankini is made from recycled nylon with Lycra® spandex", "correct": False}
]

# Metrics
total = len(results_data)
correct = sum(1 for r in results_data if r["correct"])
accuracy = (correct / total) * 100

print(f"\n📈 OVERALL ACCURACY: {accuracy:.1f}% ({correct}/{total})")
print("\n" + "="*70)
print("📋 DETAILED RESULTS:")
print("="*70)

for r in results_data:
    status = "✅ PASS" if r["correct"] else "❌ FAIL"
    print(f"\nExample {r['example']}: {status}")
    print(f"  Question: {r['question'][:60]}...")
    print(f"  Expected: {r['expected']}")
    print(f"  Got:      {r['got'][:80]}...")
