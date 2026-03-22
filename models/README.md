# models/

This directory stores trained model artifacts.

These files are not committed to the repository because they are binary blobs
that change frequently during training. Download them from the releases page
or train your own from labeled logs (see `data/README.md`).

## Files expected here

```
models/
  fault_classifier.pkl    # Trained XGBoost + isotonic calibration pipeline
  rag_index/              # ChromaDB persistent vector store (built by rag_pipeline.py)
    chroma.sqlite3
    ...
```

## Training the fault classifier

Once you have labeled logs in `data/raw/labeled/`:

```bash
pip install -r requirements.txt

# Step 1: extract features from all labeled logs
python python/build_dataset.py \
    --labels data/sample_labels.json \
    --output data/features_train.csv

# Step 2: train the classifier
python python/train.py \
    --dataset data/features_train.csv \
    --output models/fault_classifier.pkl

# Step 3: evaluate on a held-out split
python python/train.py \
    --dataset data/features_train.csv \
    --output models/fault_classifier.pkl \
    --eval
```

## Building the RAG index

```bash
# Build the rag_corpus.jsonl first (scrapes ArduPilot wiki + discuss posts)
python python/build_corpus.py --output data/rag_corpus.jsonl

# Build the vector index
python -c "
from rag_pipeline import FixSuggestionPipeline
p = FixSuggestionPipeline()
p.build_index('data/rag_corpus.jsonl')
print('Index built in models/rag_index/')
"
```

## Without a trained model

`diagnose.py` falls back to `heuristic_classify()` in `fault_classifier.py`
when no `.pkl` file is found. The heuristic classifier uses rule-based
thresholds from the ArduPilot wiki and will give reasonable results on
common fault types even with no training data.
