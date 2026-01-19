# IDTA Coursework 2 - Project Plan

## Overview
Sentiment analysis on labelled sentences using text preprocessing, classification, and topic modeling.

**Dataset**: Amazon cells reviews (1000 sentences: 500 positive, 500 negative)  
**Output**: Report (max 3000 words) + Python code (.ipynb)

---

## Task 1: Text Preprocessing (30%)
**Word Count Target**: ~750 words

### Implementation
1. Load Amazon dataset (tab-separated: sentence \t label)
2. Apply preprocessing steps in order:
   - **Remove punctuation** (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
   - **Remove numbers** (0-9 and digit combinations)
   - **Convert to lowercase** (standardization)
   - **Remove stop words** (using NLTK stopwords list)
   - **Lemmatization** (using NLTK WordNetLemmatizer)

### Deliverables
- Show **3 examples** for each preprocessing step
- Create a table showing original → processed text for each step
- Display intermediate results to show transformation chain

---

## Task 2: Bag-of-Words Classification (30%)
**Word Count Target**: ~750 words

### Implementation
1. **Feature Extraction**
   - Use TfidfVectorizer or CountVectorizer
   - Set max_features=1000-5000 (for speed)

2. **Train-Test Split**
   - 80/20 or 70/30 split
   - Set random_state for reproducibility

3. **Three Algorithms** (simple & fast):
   - **Logistic Regression** (fast, baseline)
   - **Naive Bayes** (MultinomialNB - ideal for text)
   - **Random Forest** (n_estimators=100, max_depth=10)

4. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix

### Deliverables
- Comparison table with all metrics
- Confusion matrices (visualized)
- Discussion: which algorithm performed best and why

---

## Task 3: BERT Classification (10%)
**Word Count Target**: ~500 words

### Implementation
1. **Model**: DistilBERT (distilbert-base-uncased)
   - Lighter and faster than full BERT
   - Good performance for sentiment analysis

2. **Fine-tuning**
   - Use Hugging Face Transformers
   - Train for 2-3 epochs (for speed)
   - Batch size: 16
   - Learning rate: 2e-5

3. **Evaluation**
   - Same metrics as Task 2
   - Compare with traditional algorithms

### Deliverables
- Performance comparison with Task 2 algorithms
- Discussion on BERT vs traditional methods
- Training time comparison

---

## Task 4: Topic Detection (30%)
**Word Count Target**: ~1000 words

### Implementation
1. **Algorithm**: Latent Dirichlet Allocation (LDA)
   - Set n_topics=10
   - Use preprocessed text from Task 1

2. **Visualization**
   - Top 10-15 words per topic
   - Topic distribution across documents
   - Word clouds for each topic (optional)

3. **Quality Assessment**
   - Coherence score
   - Topic interpretation
   - Overlap between topics

### Deliverables
- Table: 10 topics with top words
- Description/interpretation of each topic
- Quality assessment and discussion
- Identify which topics relate to positive/negative sentiment

---

## Implementation Strategy

### Python Libraries
```python
# Preprocessing
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Classification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# BERT
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Topic Modeling
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

### File Structure
```
IDTACoursework2_2025-26/
├── plan.md (this file)
├── assignment.ipynb (main code)
├── report.docx (final report)
└── sentiment labelled sentences/
    └── amazon_cells_labelled.txt (data)
```

---

## Report Structure

### Task 1 Section (~750 words)
- Brief intro to preprocessing necessity
- Table with examples for each step
- Analysis of text changes
- Impact discussion

### Task 2 Section (~750 words)
- Brief methodology
- Results table (all 3 algorithms)
- Confusion matrices
- Comparative analysis
- Discussion on advantages/disadvantages

### Task 3 Section (~500 words)
- BERT fine-tuning approach
- Results comparison with Task 2
- Performance vs computational cost
- When to use BERT vs traditional methods

### Task 4 Section (~1000 words)
- LDA methodology
- Topic presentation (table + interpretation)
- Quality assessment
- Relationship to sentiment
- Insights and patterns

---

## Time Estimates
- Task 1: 3-4 hours (coding + analysis)
- Task 2: 4-5 hours (implementation + comparison)
- Task 3: 3-4 hours (BERT setup + training)
- Task 4: 4-5 hours (topic modeling + interpretation)
- Report writing: 6-8 hours
- **Total**: ~25-30 hours

---

## Key Tips
1. **Keep models simple** - focus on analysis not complexity
2. **Use visualizations** - they don't count toward word limit
3. **Be concise** - avoid generic descriptions
4. **Critical analysis** - discuss why results occurred
5. **Set random seeds** - ensure reproducibility
6. **Document everything** - comments in code
7. **Save checkpoints** - don't lose progress

---

## Submission Checklist
- [ ] Report (max 3000 words, excluding tables/figures)
- [ ] All 4 tasks completed
- [ ] Tables and figures properly labeled
- [ ] APA 7 references (if external sources used)
- [ ] Code (.ipynb file)
- [ ] Code is commented and organized
- [ ] Zip file with all code
- [ ] Proofread for grammar/spelling
