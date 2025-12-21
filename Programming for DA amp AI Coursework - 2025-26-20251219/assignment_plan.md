# Stroke Dataset Analysis - Assignment Plan

**Dataset:** Healthcare Stroke Dataset (5110 patients, 12 attributes)

**Target Tasks:**
- Classification: Predict stroke occurrence
- Regression: Predict BMI values
- Clustering: Identify patient groups

---

## 1. Descriptive Analytics (10%)

### 1.1 Statistical Analysis
**Methods:**
- Calculate descriptive statistics for all 12 attributes (mean, median, std, min, max, quartiles)
- Analyze distributions (normal, skewed, etc.)
- Identify data types (numerical vs categorical)
- Compute correlation matrices for numerical features

### 1.2 Visualizations (4 required)
**Visualization 1:** Box plot
- Display outliers in numerical features (age, BMI, glucose level)

**Visualization 2:** Correlation heatmap
- Show relationships between multiple numerical variables

**Visualization 3:** Scatter plot matrix
- Visualize pairwise relationships (age vs BMI vs glucose, colored by stroke)

**Visualization 4:** Grouped bar charts / violin plots
- Show relationships between categorical and numerical variables (e.g., stroke vs work_type vs age)

---

## 2. Data Preparation (30%)

### 2.1 General Data Preparation

**Method 1: Missing Value Handling**
- Identify missing values in BMI and other attributes
- Use imputation techniques: mean/median for numerical, mode for categorical
- Alternative: K-NN imputation or regression-based imputation

**Method 2: Outlier Detection & Treatment**
- Apply IQR method or Z-score to identify outliers in age, BMI, glucose
- Treatment options: capping, removal, or transformation

**Method 3: Feature Encoding**
- One-Hot Encoding for nominal categorical variables (gender, work_type, smoking_status)
- Label Encoding for binary variables (ever_married, Residence_type)

**Method 4: Feature Scaling**
- Normalization (Min-Max scaling) for distance-based algorithms
- Standardization (Z-score) for algorithms sensitive to feature magnitude

### 2.2 Task-Specific Preparation

**For Classification (Stroke Prediction):**
- Address class imbalance using SMOTE or class weights (249 stroke vs 4861 non-stroke)
- Feature selection using correlation analysis or feature importance
- Train-test split (70-30 or 80-20) with stratification

**For Regression (BMI Prediction):**
- Remove rows where BMI is target from missing value analysis
- Create separate train-test split
- Ensure continuous target variable normalization if needed

**For Clustering:**
- Drop ID and target variable (stroke)
- Apply dimensionality reduction if needed (PCA)
- Ensure all features are numerical and scaled

---

## 3. Classification Task (25%)

### Algorithms (minimum 3):

**Algorithm 1: Logistic Regression**
- Baseline linear classifier
- Interpretable coefficients
- Suitable for binary classification

**Algorithm 2: Random Forest Classifier**
- Ensemble method, handles non-linearity
- Feature importance extraction
- Robust to outliers and imbalanced data
- Limit n_estimators (e.g., 100) for faster execution

**Algorithm 3: Decision Tree Classifier**
- Fast execution and easy to interpret
- Handles both numerical and categorical data
- Visualizable tree structure for understanding decisions
- Good baseline for comparison with ensemble methods

### Evaluation Metrics:
- Confusion matrix (TP, TN, FP, FN)
- Accuracy, Precision, Recall, F1-score (overall and per class)
- ROC-AUC curve (important for imbalanced data)
- Cross-validation scores

### Comparison Approach:
- Compare performance metrics across algorithms
- Analyze per-class performance (stroke vs non-stroke)
- Discuss trade-offs between precision and recall
- Consider computational efficiency and interpretability

---

## 4. Regression Task (15%)

### Algorithms (minimum 2):

**Algorithm 1: Linear Regression**
- Baseline model
- Assumes linear relationships
- Simple and interpretable

**Algorithm 2: K-Nearest Neighbors (KNN) Regressor**
- Fast training time (lazy learner)
- No assumptions about data distribution
- Handles non-linear relationships
- Works well with small to medium datasets

### Evaluation Metrics:
- Mean Squared Error (MSE) / Root MSE
- Mean Absolute Error (MAE)
- R-squared (R²) coefficient
- Residual analysis plots

### Comparison Approach:
- Compare error metrics (lower is better)
- Compare R² values (higher is better)
- Analyze residual distributions
- Discuss model complexity vs performance

---

## 5. Clustering Task (20%)

### Algorithms (minimum 2)
**Algorithm 1: K-Means Clustering**
- Fast partitioning method
- Determine optimal k using Elbow method or Silhouette score
- Centroid-based, works well with spherical clusters
- Efficient for large datasetste score
- Centroid-based, works well with spherical clusters

**Algorithm 2: DBSCAN (Density-Based Clustering)**
- Fast density-based method
- Automatically identifies number of clusters
- Handles noise and outliers well
- No need to specify k in advance
- Different linkage methods (ward, complete, average)

### Evaluation Metrics:
- Silhouette score (cluster quality)
- Davies-Bouldin index (lower is better)
- Calinski-Harabasz score (higher is better)
- Within-cluster sum of squares (WCSS)

### Analysis Approach:
- Report cluster sizes (coverage)
- Analyze mean values for each cluster
- Identify cluster characteristics (patient profiles)
- Compare cluster quality metrics between algorithms
- Visualize clusters using PCA or t-SNE

---

## Libraries & Tools

**Data Manipulation:**
- pandas (data loading, preprocessing)
- numpy (numerical operations)

**Visualization:**
- matplotlib (basic plots)
- seaborn (statistical visualizations)

**Machine Learning:**
- scikit-learn (all algorithms, preprocessing, evaluation)
- imbalanced-learn (SMOTE for class imbalance)

**Statistical Analysis:**
- scipy (statistical tests)
