# 🧮 Customer Lifetime Value Prediction Dashboard

This is a complete end-to-end Data Analytics + Machine Learning + Dashboarding project focused on **Customer Lifetime Value (CLV) prediction and visualization** using historical retail data.

👉 **Live Dashboard**: [https://clv-dashboard-w3ma.onrender.com](https://clv-dashboard-w3ma.onrender.com)

---

## 📊 Project Objective

The goal of this project is to:
- Predict **Customer Lifetime Value (CLV)** using historical transactions
- Identify **high-value vs low-value customers**
- Discover key behavioral patterns (Recency, Frequency, Monetary)
- Help businesses **target the right customers** for upselling, loyalty, or reactivation

---

## 📁 Dataset

- **Name**: [Online Retail II Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- **Source**: UCI Machine Learning Repository
- **Size**: ~1 million transactions (2009–2011) for a UK-based online retailer
- **Columns Used**:
  - `Invoice`, `StockCode`, `Quantity`, `Price`, `Customer ID`, `InvoiceDate`, `Country`

---

## 🧠 Solution Approach

1. **Data Cleaning**:
   - Removed nulls, negative quantities, and invalid customers
2. **Feature Engineering**:
   - Built **RFM metrics**: Recency, Frequency, Monetary
3. **Modeling**:
   - Trained a **Random Forest Regressor** to predict CLV
   - Achieved **R² = 0.999**, **RMSE ≈ 0.0015**
4. **Post-Model Insights**:
   - Created CLV segments, Pareto analysis, and behavioral breakdowns

---

## 🛠️ Tech Stack

| Layer             | Tools Used                            |
|------------------|----------------------------------------|
| Data Analysis     | Python, Pandas, NumPy                  |
| Machine Learning  | Scikit-learn (Random Forest)           |
| Visualization     | Plotly, Dash                           |
| Deployment        | Render (Free Python web hosting)       |

---

## 📈 Dashboard Insights

The dashboard includes:

- 🔷 **CLV Segments** — See how customers are distributed by predicted value
- 📉 **Recency vs CLV** — Recent buyers often bring higher lifetime value
- 📐 **Pareto Analysis** — Top 20% customers bring 80%+ revenue
- 💎 **High-Frequency VIPs** — Loyal repeat buyers = core business
- 🏆 **Top 10 Customers** — Your biggest future contributors

> Bonus: Each chart includes an insight box summarizing its takeaway 🎯

---

## 📌 Key Highlights

- ✅ Extracted RFM metrics from 1M+ rows of transaction data
- 🔁 Built a regression pipeline to predict individual CLV
- 📊 Designed a clean, interactive dashboard using Plotly Dash
- 🚀 Deployed on Render for public sharing (great for interviews & resumes!)

---

## 🖼️ Preview Screenshot

![Dashboard Preview](dashboard-screenshot.png)

_(Upload `dashboard-screenshot.png` in your repo to show this image)_

---

## 📬 Author

- **Sahil Vachher**
- [LinkedIn](https://www.linkedin.com/in/sahilvachher) (replace with yours if different)
- Built for resume projects, product/data science interviews, and end-to-end ML storytelling

---

## 💡 Want to Replicate?

```bash
git clone https://github.com/yourusername/clv-dashboard.git
cd clv-dashboard
pip install -r requirements.txt
python app.py
