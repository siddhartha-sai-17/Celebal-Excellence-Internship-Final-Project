# Fashion Product Recommendation System - Developer Manual

This guide describes local development setups, pipelines, and test suites for developers maintaining the visual recommendation engine.

---

## 🛠️ Development Environment Setup

### 1. Virtual Environment & Dependencies
Initialize a virtual environment using Python 3.11:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install pytest pytest-cov
```

### 2. Configuration Settings
All variables (e.g. subset ratios, model parameters, relative directories, image dimensions) are managed centrally in [settings.py](file:///C:/Users/banda/Desktop/Vision%20Product%20Recommendation/config/settings.py).

---

## ⚙️ Execution Pipeline (`main.py`)

The orchestration module `main.py` runs preprocessing, training, extraction, indexing, and evaluating from the command line.

### Commands

* **Validate Dataset & Build Balanced Subset**:
  ```bash
  python main.py --action prepare_dataset
  ```
  * Extracts raw styles metadata and samples balanced subsets (250 images per category).
  * Validates files and saves report to `evaluation/reports/dataset_validation_report.json`.

* **Train Category Classification Model (Transfer Learning)**:
  ```bash
  python main.py --action train_transfer_learning
  ```
  * Stage 1: Trains frozen backbone top head.
  * Stage 2: Unfreezes backbone and fine-tunes with small learning rate.
  * Checkpoints are saved under `models/checkpoints/`.

* **Train Siamese Metric Learning Model**:
  ```bash
  python main.py --action train_siamese
  ```
  * Pairs are built using positive samples (same articleType) and negative samples (different articleTypes).
  * Employs Margin Contrastive Loss. Pre-loads images into RAM to achieve a 22x training speedup.

* **Pre-Compute Image Embeddings**:
  ```bash
  python main.py --action extract_embeddings --model baseline
  python main.py --action extract_embeddings --model transfer_learning
  python main.py --action extract_embeddings --model siamese
  ```
  * Computes L2-normalized embedding tensors for all subset images and exports `.npy` matrices to `embeddings/`.

* **Build FAISS Index File**:
  ```bash
  python main.py --action build_faiss
  ```
  * Serializes flat inner-product (IP) indices to `faiss_index/` for immediate similarity search queries.

* **Execute System Benchmarks**:
  ```bash
  python main.py --action benchmark
  ```
  * Measures latency profile, FPS, RAM utilization, and dumps results into `evaluation/reports/recommendation_benchmark.json`.

---

## 🧪 Testing Suite

We use `pytest` for unit and integration testing.

* **Run all tests**:
  ```bash
  venv\Scripts\pytest.exe
  ```
* **Run coverage metrics**:
  ```bash
  venv\Scripts\pytest.exe --cov=./ --cov-report=term-missing
  ```

---

## 🔍 Code Guidelines
* Maintain docstrings describing function arguments and returns.
* Ensure all database files or image file reads resolve paths relative to `settings.BASE_DIR` dynamically to prevent cross-OS path failures.
* Utilize the centralized logger from `utils.logger` for debugging and error tracing.
