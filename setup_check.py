import sys

packages = [
    "langchain",
    "ragas",
    "sentence_transformers",
    "transformers",
    "torch",
    "requests"
]

def check_imports():
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"[OK] {pkg}")
        except ImportError as e:
            print(f"[MISSING] {pkg} -> {e}")
            sys.exit(1)

if __name__ == "__main__":
    print("Running environment sanity check...\n")
    check_imports()
    print("\nAll core dependencies are correctly installed.")
