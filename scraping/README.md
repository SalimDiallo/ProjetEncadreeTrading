# trading_application



# 🛢️ Petrol Trading Backtesting Platform - Data Module
This repository contains the **Data Module** for a comprehensive trading backtesting platform designed to evaluate investment strategies using technical indicators, Machine Learning (Random Forest/SVM), and NLP-based sentiment analysis.

## 📌 Module Overview
The Data Module acts as the automated "refinery" for the platform. [It handles the end-to-end lifecycle of financial and textual data, ensuring that the **Module ML** and **Module NLP** receive high-quality, reproducible inputs.

### Key Features
* **Automated Extraction**: Specialized collectors for the EIA (Energy Information Administration) API and financial news scraping[cite: 12, 25, 75].
* **Immutable Archiving**: Every API call and web scrape is archived as a raw JSON/HTML file in `data/raw/` before processing to ensure total reproducibility.
* **High-Performance Storage**: Cleaned data is stored in **Parquet** format, which preserves schema types (dates, floats) and offers superior speed over CSV.
* **Modular Design**: Built using an abstract collector pattern, making it trivial to add new assets or news sources.

---

## 🚀 Getting Started (Any OS)

Follow these steps to set up the data pipeline on Linux, macOS, or Windows.

### 1. Requirements
Ensure you have **Python 3.9+** installed. Install the required engineering stack:
```bash
pip install pandas requests python-dotenv pyarrow beautifulsoup4
```

### 2. Accessing the Data (EIA.gov)
The price data is sourced from the US Energy Information Administration.
1.  Go to the [EIA Open Data Registration](https://www.eia.gov/opendata/register.php).
2.  Register your email to receive your free **API Key**.
3.  Check your email and keep the key ready.

### 3. Environment Setup
The module uses a `.env` file to handle secrets securely. 
1.  Create a file named `.env` in the root directory.
2.  Add your key:
    ```text
    EIA_API_KEY=your_secret_key_here
    ```

### 4. Project Structure
The code is structured as a Python package to ensure clean imports:
```text
.
├── main.py                 # Orchestrator script
├── .env                    # Secret API keys (do not commit!)
└── data_module/
    ├── config.py           # Paths and asset constants
    ├── collectors/         # EIA and News scrapers
    └── data/               # Local data storage
        ├── raw/            # Archived JSON/HTML
        └── processed/      # Ready-to-use Parquet files
```

---

## 🛠️ Usage
To fetch the latest petrol prices and news headlines, run the orchestrator from the project root:

```bash
python main.py
```

### Data Verification
Once the pipeline finishes, you can verify the integrity of your data in your terminal:
```bash
python -c "import pandas as pd; print(pd.read_parquet('data_module/data/processed/petrol_wti_daily.parquet').head())"
```

## 🎯 Success Criteria Met
* **Reliability**: Automated raw data archiving for consistent backtesting.
* **Performance**: Utilizes modular processing and Parquet binary storage.
* *Transparency**: Clear separation between raw acquisition and processed outputs.

---



