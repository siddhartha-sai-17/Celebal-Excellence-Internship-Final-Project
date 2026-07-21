# 📁 Repository File Contributions & Architecture Map

This guide outlines the file and folder layout of the **Fashion Product Recommendation System**, describing how each file contributes to the overall pipeline and how the directories interact during training, search, and frontend rendering.

---

## 🗺️ 1. Complete Architecture Mapping

The repository is divided into three functional layers:
1.  **Training & Preprocessing Layer (`siamese/`, `training/`, `utils/`)**: Extracts, normalizes, and trains deep learning metric spaces (Siamese / Category Classifier).
2.  **Inference & Search Engine Layer (`recommendation/`, `config/`)**: Indexes product vectors, evaluates distance metrics, applies metadata filtering, and returns results.
3.  **UI & Presentation Layer (`app/`)**: Renders the Streamlit frontend layout, graphs metrics, tracks system logs, and overlays Grad-CAM attention.

---

## 📂 2. Directory & File Contribution Details

### 🎨 A. UI & Presentation Layer (`app/`)
This folder contains the Streamlit frontend components and layout pages.

*   📂 **`app/`**
    *   📄 **`streamlit_app.py`**: The entrypoint of the application. It acts as the routing hub and displays the initial dashboard, system health checks, and summary metrics.
    *   📂 **`app/components/`**
        *   📄 **`header.py`**: Renders a premium, consistent page title with a custom CSS flexbox alignment to keep emoji icons and text aligned on all browsers.
        *   📄 **`sidebar.py`**: Builds the left-hand control panel containing the model selector (CNN/Transfer/Siamese), sliders for thresholds and Top-K matches, and checkboxes for e-commerce metadata filters.
        *   📄 **`upload_widget.py`**: Creates a stylized drag-and-drop file uploader area with custom flex alignment.
    *   📂 **`app/pages/`**
        *   📄 **`1_Home.py`**: Renders the landing hub, explaining the project goals, objectives, and pipelines.
        *   📄 **`2_Image_Search.py`**: The core page. Takes the uploaded query, triggers the recommendation engine, returns the matching item cards, and handles the backpropagation callback for Grad-CAM.
        *   📄 **`3_Model_Comparison.py`**: Compares matches across all three models side-by-side using the same query image.
        *   📄 **`4_Performance.py`**: Displays performance logs, benchmarking charts, search speeds, and latency graphs.
        *   📄 **`5_Evaluation_Metrics.py`**: Visualizes evaluation metrics (Precision, Recall, MAP) based on pre-compiled test outputs.
        *   📄 **`6_About.py`**: Explains the mathematical concepts (Siamese Networks, Contrastive Loss, Grad-CAM).
        *   📄 **`7_Dataset_Explorer.py`**: Allows users to filter and browse the subset e-commerce catalog interactively.
        *   📄 **`8_System_Logs.py`**: Displays real-time validation logs and trace records.
    *   📂 **`app/utils/`**
        *   📄 **`theme.py`**: Houses the premium dark glassmorphism stylesheet, styling input overlays, table rows, and scrollbars.

---

### 🔍 B. Inference & Recommendation Layer (`recommendation/`)
This folder acts as the backend search engine, evaluating similarities and returning final matches.

*   📂 **`recommendation/`**
    *   📄 **`recommendation_service.py`**: The primary coordinator. It initializes models, manages loaded embedding files, coordinates FAISS indexes, and processes requests from the frontend pages.
    *   📄 **`recommendation_engine.py`**: Validates input images, maps model selections, and delegates embedding extraction.
    *   📄 **`faiss_engine.py`**: Builds and maintains the FAISS indexes (e.g. `IndexFlatIP`). Handles high-dimensional vector search.
    *   📄 **`ranking_engine.py`**: Takes search results and applies user parameters (minimum similarity threshold, gender/category filters).
    *   📄 **`similarity_engine.py`**: Standard similarity fallbacks (calculates cosine similarities).
    *   📄 **`embedding_database.py`**: Handles local read/write IO operations for pre-computed vector files.
    *   📄 **`metadata_loader.py`**: Reads the e-commerce database (`subset_metadata.csv`), ensuring clean data types and categorical attributes.
    *   📄 **`evaluation_engine.py`**: Runs evaluations (Top-K precision and recall) against validation splits.

---

### 🧠 C. Training & Metric Learning Layer (`siamese/` & `training/`)
These folders contain training routines for fine-tuning weights and optimizing embedding distance representations.

*   📂 **`siamese/`**
    *   📄 **`model.py`**: Defines the Siamese Network architecture, wrapping the ResNet50 backbone with dense, dropout, and L2 normalization layers.
    *   📄 **`losses.py`**: Defines **Contrastive Loss** used to penalize distance between matching pairs.
    *   📄 **`dataset.py`**: Pipeline that reads e-commerce groups and creates multi-threaded batch sequences of training pairs.
    *   📄 **`evaluator.py`**: Evaluates model similarity performance on validation datasets during training.
    *   📄 **`metrics.py`**: Computes validation accuracy based on distance threshold.
    *   📄 **`inference.py`**: Interface for extracting features using fine-tuned Siamese weights.
*   📂 **`training/`**
    *   📄 **`trainer.py`**: Full training pipeline for classification (Transfer Learning) and Siamese training.
    *   📄 **`siamese_training.py`**: Specific trainer wrapper calling Contrastive Loss optimizers.
    *   📄 **`callbacks.py`**: Handlers for saving best model checkpoints, logging metrics, and logging training histories.

---

### 🛠️ D. Helper Utilities (`utils/` & `config/`)
Shared libraries and parameters referenced across the repository.

*   📂 **`utils/`**
    *   📄 **`image_utils.py`**: Contains standard image processing logic (resizing, tensor conversions) and the **Grad-CAM** calculation pipeline.
    *   📄 **`logger.py`**: Global logger settings.
    *   📄 **`timer.py`**: Simple execution times metrics tool used to capture search latencies.
*   📂 **`config/`**
    *   📄 **`settings.py`**: Centralizes global configurations (image size, model directories, categories, cache paths).

---

## ⚙️ 3. How the Files Work Together

### 1. The Startup Verification Loop
When the Streamlit dashboard starts, `app/streamlit_app.py` triggers `app/utils/startup.py`.
1.  It checks configurations from `config/settings.py`.
2.  If missing, it automatically creates directories and downloads dependencies.
3.  It validates metadata via `recommendation/metadata_loader.py`.
4.  It initializes model checkpoints inside `models/`.

### 2. The Search Execution Loop
When a user uploads an image on `2_Image_Search.py`:
1.  The image is sent to `utils/image_utils.py` for resizing and tensor preprocessing.
2.  `recommendation/recommendation_service.py` feeds the image tensor to the selected model (`models/checkpoints/` weight files) to extract a `512`-dim embedding.
3.  The embedding is sent to `recommendation/faiss_engine.py` to fetch similar product IDs.
4.  `recommendation/ranking_engine.py` matches these IDs to metadata loaded by `metadata_loader.py`, filters them by sidebar criteria (e.g. Color, Category, Threshold), and returns the results to the UI.

### 3. The Grad-CAM loop
If the user clicks "Generate Activation Heatmap" on the UI:
1.  `app/pages/2_Image_Search.py` calls `generate_gradcam()` in `utils/image_utils.py`.
2.  The function dynamically traces backpropagation calculations on the model layer.
3.  `overlay_heatmap_on_image()` overlays the heatmap onto the image and renders it on the dashboard.
