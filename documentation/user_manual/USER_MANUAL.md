# Fashion Product Recommendation System - User Manual

Welcome to the **Fashion Product Recommendation System**, a premium deep learning application for retrieving visually similar apparel, footwear, watches, and handbags.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have:
* Python 3.10 or 3.11 installed.
* Raw dataset placed under `dataset/images/` and `dataset/styles.csv`.

### Installation & Run
1. Run the setup script to install all python requirements.
2. Launch the Streamlit dashboard by double-clicking `run_app.bat` or executing:
   ```bash
   venv\Scripts\streamlit.exe run app/streamlit_app.py
   ```
3. Open your browser and navigate to the local URL (usually `http://localhost:8501`).

---

## 🖥️ Application Features & Walkthrough

### 1. Home Dashboard
The entry point shows the **Project Overview**, **Dataset Analytics**, and **System Health Telemetry**.
* **Objective Cards**: Displays short summaries of the three backend model methods.
* **Dataset Analytics**: Shows the verified record count, subset distribution, and health checklist.
* **Tech Stack**: Lists the software libraries and architectures used.

### 2. Image Search Page
Perform visual search by uploading an image.
* **Upload Widget**: Drag & drop or browse a `.jpg` or `.png` file.
* **Sidebar Controls**:
  * **Embedding Model**: Choose between **Baseline CNN**, **Transfer Learning**, or **Siamese Network**.
  * **Top-K Results**: Slider (5 to 50) to limit recommendations shown.
  * **Similarity Threshold**: Slider to filter recommendations below a certain similarity percentage.
  * **FAISS Acceleration**: Toggle to switch between sub-millisecond FAISS indexing and exact Cosine calculations.
  * **Metadata Filters**: Restrict suggestions to specific Genders, Categories, Colours, Seasons, or Usages.
* **Search Execution**: Click **Generate Recommendations** to see similarity score tags, product names, categories, and matching ranks.
* **Explainable AI (Grad-CAM)**: Expand this panel to visualize a heatmap overlay on your query image showing which parts of the design (e.g. collars, buckles, watch face, shoe heel) the neural network focused on.

### 3. Model Comparison Page
Upload a query and compare the retrieved matches from all three models side-by-side.
* Helps analyze how baseline features compare against fine-tuned classification and Siamese metric learning embedding spaces.

### 4. Performance Page
Diagnostics dashboard showing real-time resource utilization.
* **Telemetry**: CPU usage, RAM allocation, disk capacity, active threads, and GPU availability.
* **Overview Tab**: Key responsiveness latency (ms) and throughput (FPS) per model.
* **Latency Component Breakdown Tab**: Grouped and stacked charts separating CNN inference times from similarity search latency.
* **Radar Tab**: polar model comparison across normalised axes.

### 5. Evaluation Metrics Page
Visualizes accuracy and classification proxy evaluations.
* **Retrieval Accuracy Tab**: Precision@K, Recall@K, and Mean Average Precision (mAP) side-by-side comparison.
* **Rank & Quality Tab**: NDCG, Hit Rate, and Reciprocal Rank metrics.
* **Model Training Logs Tab**: Loss and Accuracy training/validation history curves loaded dynamically from checkpoints.

### 6. Dataset Explorer Page
Interactive data distribution graphs.
* Displays charts and statistics for categories, genders, top colours, seasons, and usages.
* Includes a **Raw Data Tab** to search, sort, and filter records inside the training subset.

### 7. System Logs Page
Coloured console log entry reader.
* Filters logs by level (INFO/WARNING/ERROR/DEBUG), supports text search, handles line limits, and contains a log download button.

---

## 🛠️ Troubleshooting

* **Missing Thumbnails / Skeleton Blocks**: Ensure raw images exist in `dataset/images/`.
* **Latency Spikes**: Turn on **FAISS Acceleration** in the sidebar.
* **Model Checkpoint Missing**: Ensure you ran the model training pipeline (`main.py`) before starting the app.
