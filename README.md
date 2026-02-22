# 🎬 BERT-Based Movie Recommendation System

## 📌 Project Overview
This project implements an advanced **Movie Recommendation System** based on deep learning and Natural Language Processing techniques. The system enhances recommendation quality by integrating **Bidirectional Encoder Representations from Transformers (BERT)** with traditional text-vectorization approaches.

The goal is to suggest movies that are highly relevant to a selected movie by understanding both:

- contextual meaning of descriptions
- thematic similarity of metadata

The approach combines **BERT embeddings** and **TF-IDF similarity** to produce accurate and meaningful recommendations. :contentReference[oaicite:0]{index=0}

---

## 🎯 Problem Statement
Modern streaming platforms contain massive amounts of content, making it difficult for users to discover relevant movies. Recommendation systems help solve this by predicting user preferences and suggesting suitable movies.

Traditional recommendation approaches include:
- collaborative filtering
- content-based filtering
- hybrid systems :contentReference[oaicite:1]{index=1}

However, traditional methods struggle to capture deep semantic meaning from text. This project addresses that limitation using transformer-based contextual embeddings.

---

## 🎯 Objectives
- Build a recommendation system using contextual NLP models
- Improve recommendation accuracy using deep learning
- Compare multiple NLP similarity techniques
- Combine semantic and keyword-based similarity
- Provide relevant ranked movie suggestions

---

## 🧠 Core Idea
The system uses **BERT to analyze movie descriptions** and determine similarity based on context, while **TF-IDF analyzes tags and metadata** to capture thematic similarity. These methods are combined into a unified recommendation score. :contentReference[oaicite:2]{index=2}

---

## 📂 Dataset Used
The system uses two datasets:

### 1. Movies Dataset
Contains:
- title
- overview
- genres
- keywords

### 2. Credits Dataset
Contains:
- cast
- crew

Both datasets are merged using movie title as a common key to create a unified dataset for processing. :contentReference[oaicite:3]{index=3}

---

## ⚙️ Methodology

### Step 1 — Data Integration
Two datasets are merged to combine movie content information with cast and crew details. :contentReference[oaicite:4]{index=4}

---

### Step 2 — Data Preprocessing
Processing steps include:

- Extract genres and keywords from JSON format
- Select top three cast members
- Extract director from crew
- Normalize text
- Remove inconsistencies

These steps improve data quality and consistency. :contentReference[oaicite:5]{index=5}

---

### Step 3 — Feature Engineering
Relevant movie attributes are combined into a single **tags column** consisting of:

```
genres + keywords + cast + crew
```

This column represents each movie’s overall identity. :contentReference[oaicite:6]{index=6}

---

### Step 4 — BERT Embedding Generation
The movie overview text is:

1. Tokenized  
2. Passed into a pretrained BERT model  
3. Converted into contextual embeddings  

The embedding of the **[CLS] token** is used as the semantic representation of the movie description. :contentReference[oaicite:7]{index=7}

---

### Step 5 — TF-IDF Vectorization
The tags column is vectorized using TF-IDF, which highlights important words relative to the entire dataset. :contentReference[oaicite:8]{index=8}

---

### Step 6 — Similarity Calculation
Similarity between movies is computed using **cosine similarity**:

```
similarity(A,B) = (A · B) / (||A|| × ||B||)
```

This is applied to both:
- BERT embeddings
- TF-IDF vectors :contentReference[oaicite:9]{index=9}

---

### Step 7 — Hybrid Recommendation Score
Final similarity is computed by combining both similarity matrices:

```
Combined Score = α × BERT + (1 − α) × TF-IDF
```

This produces more balanced and accurate recommendations. :contentReference[oaicite:10]{index=10}

---

## 🏗 System Architecture
```
Datasets
   ↓
Merge & Preprocess
   ↓
Feature Engineering
   ↓
BERT Embeddings + TF-IDF Vectors
   ↓
Similarity Computation
   ↓
Score Combination
   ↓
Ranked Recommendations
```

---

## 📊 Models Compared
The project evaluates multiple NLP approaches:

| Model | Purpose |
|------|--------|
BERT | Contextual semantic similarity |
Word2Vec | Word relationship similarity |
BoW | Word frequency similarity |
TF-IDF | Keyword importance |

Results show that **BERT achieves the highest precision, recall, and F1-score** among all methods. :contentReference[oaicite:11]{index=11}

---

## 📈 Performance Results

### Precision / Recall / F1
| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
BERT | 0.85 | 0.80 | 0.82 |
Word2Vec | 0.80 | 0.78 | 0.79 |
BoW | 0.75 | 0.70 | 0.72 |

BERT shows the best balance between precision and recall. :contentReference[oaicite:12]{index=12}

---

### AUC Score Comparison
| Model | AUC |
|------|-----|
BERT | 0.9432 |
Word2Vec | 0.8698 |
BoW | 0.8315 |

Higher AUC confirms better classification and recommendation ability. :contentReference[oaicite:13]{index=13}

---

## 🚀 Features
- Context-aware recommendation
- Hybrid similarity scoring
- Multiple NLP models
- Metadata + semantic analysis
- Modular pipeline design
- Scalable architecture

---

## 🛠 Requirements
Install dependencies:

```bash
pip install pandas numpy scikit-learn transformers torch
```

Environment:
- Python 3.8+
- Jupyter Notebook / Colab
- GPU recommended

---

## ▶️ How to Run
1. Download datasets
2. Place them in project folder
3. Open notebook
4. Run cells sequentially
5. Provide movie title input
6. View recommended movies

---

## 🎓 Learning Outcomes
This project demonstrates practical understanding of:

- NLP pipelines
- Transformer models
- Semantic similarity
- Feature engineering
- Hybrid recommender systems
- Machine learning evaluation metrics

---

## 🔮 Future Scope
Future improvements may include:

- optimizing performance for large datasets
- improving scalability
- testing across diverse datasets
- incorporating user interaction data
- extending system to other domains

These enhancements can further improve accuracy and usability. :contentReference[oaicite:14]{index=14}

---

## 🏁 Conclusion
This project demonstrates that integrating **BERT with traditional recommendation techniques** significantly improves recommendation quality. By combining contextual understanding with keyword-based similarity, the system delivers accurate, relevant, and meaningful movie suggestions.

---

⭐ **Run Notebook → Generate Embeddings → Compute Similarity → Get Recommendations**
