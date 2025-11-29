# Heart Attack Prediction - Code Implementation Plan

## Project Overview
**Objective:** Develop AI-based models to predict heart attacks using health indicators (BMI, age, smoking status, etc.)

**Dataset:** heart_attack.csv

**Implementation:** Single Jupyter Notebook (.ipynb file)

**Models to Implement:** 
1. KNN (K-Nearest Neighbors)
2. SVM (Support Vector Machine)
3. Decision Trees
4. Gradient Boosting Tree (must include as ANN-based/ensemble model)

---

## Phase 1: Data Preparation & Exploration [20%]

### 1.1 Initial Data Loading
**Required Actions:**
- Import all necessary libraries (pandas, numpy, matplotlib, seaborn, sklearn, tensorflow/keras or xgboost)
- Set random seeds for reproducibility (numpy, tensorflow if used)
- Suppress warnings for clean output
- Load the heart_attack.csv dataset into a DataFrame
- Display dataset shape (number of rows and columns)
- Show first 5-10 rows of the dataset
- Display column names and data types using .info()
- Show statistical summary using .describe()

### 1.2 Exploratory Data Analysis (EDA)
**Required Actions:**
- Display count of missing values for each column
- Calculate total missing values in the dataset
- Create visualizations:
  - Histograms or distribution plots for all numerical features (use subplots for clean layout)
  - Box plots to identify outliers
  - Count plot or bar chart for target variable distribution
- Create correlation matrix heatmap using seaborn
- Analyze which features are most correlated with target variable
- Check for duplicate rows

### 1.3 Data Quality & Preprocessing
**Required Actions:**
- Handle missing values (use appropriate strategy: mean/median imputation, drop, or fill)
- Handle outliers if detected (document decision to keep or remove)
- Encode categorical variables if present (use LabelEncoder or OneHotEncoder)
- Check class balance in target variable
- If imbalanced, document the imbalance ratio
- Consider using class weights or SMOTE if severe imbalance exists

### 1.4 Feature Scaling & Train-Test Split
**Required Actions:**
- Separate features (X) and target variable (y)
- Apply StandardScaler to normalize all features
- Split data into training and testing sets (80-20 or 70-30 split)
- Use stratified split to maintain class distribution (stratify=y)
- Set random_state for reproducibility
- Display sizes of training and testing sets
- Verify class distribution is maintained in both sets

---

## Phase 2: Model Development [20%]

### 2.1 Model 1: K-Nearest Neighbors (KNN)
**Required Actions:**
- Import KNeighborsClassifier from sklearn
- Initialize KNN model with initial k value (start with k=5)
- Train the model on X_train and y_train using .fit()
- Make predictions on X_test using .predict()
- Store predictions for evaluation
- Optionally: Test different k values (3, 5, 7, 9, 11) and compare performance
- Optionally: Use cross-validation to find optimal k

### 2.2 Model 2: Support Vector Machine (SVM)
**Required Actions:**
- Import SVC from sklearn.svm
- Initialize SVM model with kernel='rbf' (or test linear, rbf, poly)
- Set probability=True to enable probability predictions for ROC curve
- Train the model on X_train and y_train using .fit()
- Make predictions on X_test using .predict()
- Store predictions for evaluation
- Optionally: Test different kernels and compare performance
- Note: SVM works best with scaled data (already done in preprocessing)

### 2.3 Model 3: Decision Trees
**Required Actions:**
- Import DecisionTreeClassifier from sklearn.tree
- Initialize Decision Tree model with random_state for reproducibility
- Consider setting max_depth to prevent overfitting (e.g., max_depth=5 or 10)
- Train the model on X_train and y_train using .fit()
- Make predictions on X_test using .predict()
- Store predictions for evaluation
- Optionally: Visualize the decision tree structure
- Optionally: Display feature importance

### 2.4 Model 4: Gradient Boosting Tree
**Required Actions:**
- Import GradientBoostingClassifier from sklearn.ensemble (or XGBoost if preferred)
- Initialize Gradient Boosting model with parameters:
  - n_estimators (number of boosting stages, e.g., 100)
  - learning_rate (e.g., 0.1)
  - max_depth (e.g., 3)
  - random_state for reproducibility
- Train the model on X_train and y_train using .fit()
- Make predictions on X_test using .predict()
- Store predictions for evaluation
- Optionally: Display feature importance from the model
- Note: This is an advanced ensemble method that combines multiple weak learners

---

## Phase 3: Model Evaluation [20%]

### 3.1 Create Evaluation Function
**Required Actions:**
- Create a reusable function named `evaluate_model(model, X_test, y_test, model_name)`
- This function should:
  - Make predictions using model.predict()
  - Calculate accuracy using accuracy_score()
  - Calculate precision using precision_score()
  - Calculate recall using recall_score()
  - Calculate F1-score using f1_score()
  - Generate confusion matrix using confusion_matrix()
  - Get probability predictions using predict_proba() or decision_function()
  - Calculate ROC curve using roc_curve()
  - Calculate AUC score using auc() or roc_auc_score()
  - Print all metrics in a formatted way
  - Plot confusion matrix as a heatmap
  - Return dictionary with all metrics including FPR and TPR for ROC plotting

### 3.2 Evaluate All Models
**Required Actions:**
- Call the evaluation function for each of the 4 models:
  - KNN model
  - SVM model
  - Decision Trees model
  - Gradient Boosting model
- Store results from each model in a list
- Each evaluation should display:
  - All calculated metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
  - Confusion matrix heatmap with proper labels
- Save metric dictionaries for comparison table

### 3.3 Optional: Cross-Validation
**Optional Actions:**
- Use cross_val_score with cv=5 or cv=10 for each model
- Calculate mean and standard deviation of cross-validation scores
- Display results to show model stability
- This helps verify that models generalize well

---

## Phase 4: Model Comparison [20%]

### 4.1 Create Comparison Table
**Required Actions:**
- Create a DataFrame from the list of result dictionaries
- Include columns: Model, Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Set Model as index
- Display the complete comparison table
- Sort by different metrics to see rankings:
  - Sort by Accuracy to see most accurate model
  - Sort by Recall (important for medical diagnosis - don't miss heart attacks)
  - Sort by F1-Score for balanced performance

### 4.2 Visualize Model Comparison
**Required Actions:**
- Create bar chart comparing all metrics across 4 models
  - Use the comparison DataFrame
  - Plot with appropriate figure size
  - Add title, labels, and legend
  - Rotate x-axis labels if needed for readability
- Create overlayed ROC curves for all 4 models
  - Plot TPR vs FPR for each model on same figure
  - Label each curve with model name and AUC score
  - Add diagonal line for random classifier (y=x)
  - Add grid, legend, title, and axis labels
  - This visualization shows which model best separates classes

### 4.3 Identify Best Model
**Required Actions:**
- Based on comparison table and visualizations, identify the best performing model
- Consider multiple factors:
  - Highest accuracy for overall performance
  - Highest recall for medical context (catching all heart attack cases is critical)
  - Highest F1-score for balanced precision and recall
  - Highest AUC for overall classification ability
- Print a clear statement identifying the best model
- Justify the selection with specific metric values
- Note: For heart attack prediction, recall is often prioritized over precision

---

## Phase 5: Model Improvement [20%]

### 5.1 Improvement Strategy 1: Hyperparameter Tuning
**Required Actions:**
- Select the best performing model from Phase 4
- Define a parameter grid (dictionary) with hyperparameters to test:
  - For KNN: {'n_neighbors': [3, 5, 7, 9, 11, 15], 'weights': ['uniform', 'distance']}
  - For SVM: {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1], 'kernel': ['rbf', 'linear']}
  - For Decision Tree: {'max_depth': [3, 5, 10, 15, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
  - For Gradient Boosting: {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
- Import and use GridSearchCV from sklearn.model_selection
- Set cv=5 for 5-fold cross-validation
- Fit GridSearchCV on training data
- Print the best parameters found
- Get the best estimator (optimized model)
- Evaluate the tuned model on test data using the evaluation function
- Create comparison showing:
  - Metrics BEFORE tuning (original model)
  - Metrics AFTER tuning (optimized model)
  - Improvement in each metric
- Display this as a comparison table or side-by-side metrics

### 5.2 Improvement Strategy 2: Feature Selection/Engineering
**Required Actions:**
- Choose ONE of these feature-based approaches:

**Option A: Feature Selection using Feature Importance**
- Use the best tree-based model (Decision Tree or Gradient Boosting)
- Get feature importances using model.feature_importances_
- Create bar plot showing importance of each feature
- Select top N most important features (e.g., top 5-10 features)
- Retrain the best model using only selected features
- Evaluate performance and compare with original (all features)

**Option B: Feature Selection using Recursive Feature Elimination (RFE)**
- Import RFE from sklearn.feature_selection
- Use RFE with the best model to select top features
- Specify number of features to select
- Fit RFE on training data
- Transform both training and testing data
- Retrain the best model on selected features
- Evaluate performance and compare with original

**Option C: Feature Engineering - Create Interaction Features**
- Create new features by combining existing ones (e.g., BMI * Age, or Age^2)
- Add polynomial features using PolynomialFeatures
- Retrain the best model with augmented feature set
- Evaluate performance and compare with original

**Evidence to Provide:**
- List of features selected or created
- Feature importance plot (if applicable)
- Metrics BEFORE feature selection/engineering
- Metrics AFTER feature selection/engineering
- Comparison table showing improvement
- State whether improvement was achieved and by how much

---

## Single Notebook Structure

### Section 1: Introduction & Setup (Markdown + Code)
**Purpose:** Set up the environment and load data
- Add markdown title and project description
- Import all required libraries
- Set random seeds for reproducibility (np.random.seed(42), etc.)
- Suppress warnings
- Load heart_attack.csv into DataFrame
- Display dataset shape, first few rows, data types, and statistical summary

### Section 2: Exploratory Data Analysis (Code + Visualizations)
**Purpose:** Understand the data and identify patterns
- Check for missing values (count and display)
- Check for duplicate rows
- Create distribution plots for all numerical features (use subplots)
- Create target variable distribution plot (bar chart or count plot)
- Generate correlation heatmap using seaborn
- Identify which features correlate most with target

### Section 3: Data Preprocessing (Code)
**Purpose:** Prepare data for machine learning models
- Handle missing values (impute or drop based on analysis)
- Handle outliers if severe outliers detected
- Encode categorical variables if any exist
- Check class balance in target variable (print value counts and percentages)
- Separate features (X) and target (y)
- Apply StandardScaler to features
- Perform train-test split (80-20, stratified, random_state=42)
- Display training and testing set sizes and class distributions

### Section 4: Model Evaluation Function (Code)
**Purpose:** Create reusable evaluation function
- Define function `evaluate_model(model, X_test, y_test, model_name)`
- Calculate: accuracy, precision, recall, f1-score, confusion matrix, ROC-AUC
- Plot confusion matrix heatmap
- Return dictionary with all metrics and ROC data (FPR, TPR)

### Section 5: Model 1 - KNN (Code)
**Purpose:** Train and evaluate K-Nearest Neighbors
- Import KNeighborsClassifier
- Initialize KNN with n_neighbors=5
- Train on X_train, y_train
- Evaluate using the evaluation function
- Store results in a list

### Section 6: Model 2 - SVM (Code)
**Purpose:** Train and evaluate Support Vector Machine
- Import SVC
- Initialize SVM with kernel='rbf', probability=True
- Train on X_train, y_train
- Evaluate using the evaluation function
- Store results in the list

### Section 7: Model 3 - Decision Trees (Code)
**Purpose:** Train and evaluate Decision Tree
- Import DecisionTreeClassifier
- Initialize Decision Tree with random_state=42, max_depth=10
- Train on X_train, y_train
- Evaluate using the evaluation function
- Store results in the list

### Section 8: Model 4 - Gradient Boosting (Code)
**Purpose:** Train and evaluate Gradient Boosting
- Import GradientBoostingClassifier
- Initialize with n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
- Train on X_train, y_train
- Evaluate using the evaluation function
- Store results in the list

### Section 9: Model Comparison (Code + Visualizations)
**Purpose:** Compare all models and identify the best one
- Create comparison DataFrame from results list
- Display full comparison table
- Create bar chart comparing all metrics across models
- Create overlayed ROC curves for all 4 models
- Identify and print the best performing model with justification

### Section 10: Improvement Strategy 1 - Hyperparameter Tuning (Code)
**Purpose:** Optimize the best model's hyperparameters
- Select the best model from comparison
- Define parameter grid for that model
- Use GridSearchCV with cv=5
- Fit on training data and find best parameters
- Evaluate tuned model
- Create before/after comparison showing improvement

### Section 11: Improvement Strategy 2 - Feature Selection/Engineering (Code)
**Purpose:** Improve model through better feature usage
- Choose one approach: Feature Importance, RFE, or Feature Engineering
- Implement the chosen approach
- Retrain best model with modified features
- Evaluate improved model
- Create before/after comparison showing improvement

### Section 12: Final Summary (Markdown + Code)
**Purpose:** Summarize all findings
- Create final summary table with all models including improved versions
- Identify the final best model
- Print key findings and insights
- Document final performance metrics

---

## Required Libraries

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations and array operations

### Visualization Libraries
- **matplotlib.pyplot** - Creating plots and charts
- **seaborn** - Statistical data visualization (heatmaps, distribution plots)

### Preprocessing & Model Selection
- **sklearn.preprocessing.StandardScaler** - Feature scaling/normalization
- **sklearn.model_selection.train_test_split** - Splitting data into train/test sets
- **sklearn.model_selection.GridSearchCV** - Hyperparameter tuning with cross-validation

### Evaluation Metrics
- **sklearn.metrics.accuracy_score** - Calculate accuracy
- **sklearn.metrics.precision_score** - Calculate precision
- **sklearn.metrics.recall_score** - Calculate recall (sensitivity)
- **sklearn.metrics.f1_score** - Calculate F1-score
- **sklearn.metrics.confusion_matrix** - Generate confusion matrix
- **sklearn.metrics.roc_curve** - Generate ROC curve data
- **sklearn.metrics.auc** or **roc_auc_score** - Calculate AUC score

### Machine Learning Models
- **sklearn.neighbors.KNeighborsClassifier** - KNN algorithm
- **sklearn.svm.SVC** - Support Vector Machine
- **sklearn.tree.DecisionTreeClassifier** - Decision Trees
- **sklearn.ensemble.GradientBoostingClassifier** - Gradient Boosting

### Optional Libraries (for advanced improvements)
- **sklearn.feature_selection.RFE** - Recursive Feature Elimination
- **imblearn.over_sampling.SMOTE** - Handle class imbalance (if needed)
- **warnings** - Suppress warning messages for clean output

---

## Code Quality Guidelines

### Best Practices
- Clear and descriptive variable names
- Markdown cells explaining each section
- Comments for complex operations
- Reproducibility (set random seeds: np.random.seed(42), tf.random.set_seed(42))
- Clean, readable code structure
- Proper error handling

### Visualization Standards
- Clear titles and labels on all plots
- Legends where needed
- Consistent color schemes
- Appropriate figure sizes
- Professional appearance

### Code Testing
- [ ] All cells run without errors
- [ ] Visualizations display correctly
- [ ] Results are reproducible
- [ ] All metrics calculated correctly
- [ ] Comparison tables are accurate

---

## Key Implementation Requirements

### Critical Variables to Define
- **target_col** - Name of the target column in dataset (adjust based on actual column name in heart_attack.csv)
- **X** - Feature matrix (all columns except target)
- **y** - Target variable (column to predict)
- **X_train, X_test** - Training and testing features
- **y_train, y_test** - Training and testing target values
- **scaler** - StandardScaler object fitted on training data
- **results_list** - Empty list to store evaluation results from all models

### Essential Function to Create
**evaluate_model(model, X_test, y_test, model_name)**
- Input parameters:
  - model: Trained model object
  - X_test: Test features
  - y_test: Test target values
  - model_name: String name for display
- Must calculate: accuracy, precision, recall, f1-score, confusion matrix, ROC curve data, AUC
- Must visualize: confusion matrix heatmap
- Must return: Dictionary containing all metrics plus FPR and TPR arrays for ROC plotting

### Model Training Pattern (For Each Model)
1. Import the model class
2. Initialize model with appropriate parameters
3. Fit model on X_train and y_train
4. Call evaluate_model() function
5. Append returned dictionary to results_list

### Comparison and Visualization Requirements
- Convert results_list to pandas DataFrame
- Display comparison table with all metrics
- Create bar chart comparing metrics across models
- Create overlayed ROC curves showing all models on one plot
- Identify best model based on metrics (prioritize recall for medical context)

### Hyperparameter Tuning Pattern
- Define param_grid dictionary with parameters to test
- Create GridSearchCV object with the base model, param_grid, and cv=5
- Fit GridSearchCV on training data
- Extract best parameters and best estimator
- Evaluate tuned model and compare with baseline

### Feature Selection/Engineering Pattern
- Choose one approach (Feature Importance, RFE, or Polynomial Features)
- Apply transformation to select or create features
- Retrain the best model with new feature set
- Evaluate and compare performance with original model

---

## Important Implementation Notes

### Critical Considerations
1. **Medical Context:** For heart attack prediction, **recall (sensitivity)** is the most critical metric
   - Missing a heart attack case (false negative) is far worse than a false alarm (false positive)
   - Prioritize recall when selecting the best model

2. **Reproducibility:** 
   - Set random_state=42 in all operations (train_test_split, model initialization, GridSearchCV)
   - Set numpy seed: np.random.seed(42)
   - This ensures results can be reproduced exactly

3. **Data-Specific Adjustments:**
   - Examine the actual column names in heart_attack.csv
   - Identify the target column name (might be 'target', 'output', 'heart_attack', etc.)
   - Check if categorical variables exist and encode them if needed
   - Verify if missing values need imputation or can be dropped

4. **Class Imbalance:**
   - If one class has significantly fewer samples, consider using:
     - class_weight='balanced' parameter in models that support it
     - SMOTE for oversampling minority class
     - Stratified split is already handled in train_test_split

5. **Model-Specific Notes:**
   - SVM: Must set probability=True to get probability predictions for ROC curve
   - Decision Trees: May overfit without max_depth constraint
   - Gradient Boosting: Computationally intensive but usually high performance
   - All models benefit from scaled features (already handled by StandardScaler)

---

## Expected Deliverable - Complete Notebook with:

### Data Analysis Outputs
- Dataset dimensions and structure
- Missing value analysis
- Distribution plots for all features
- Correlation heatmap
- Class balance visualization

### Model Training Outputs (4 Models)
- KNN model trained and evaluated
- SVM model trained and evaluated
- Decision Tree model trained and evaluated
- Gradient Boosting model trained and evaluated
- Each with: confusion matrix, all metrics printed, stored results

### Comparison Outputs
- Comprehensive comparison table (DataFrame) with 5 metrics for 4 models
- Bar chart comparing all metrics across models
- Overlayed ROC curves showing all 4 models
- Clear statement identifying best performing model

### Improvement Outputs (2 Strategies)
- Strategy 1: Hyperparameter tuning results
  - Best parameters found
  - Before/after metrics comparison
  - Evidence of improvement (or explanation if no improvement)
- Strategy 2: Feature selection/engineering results
  - Features selected or created
  - Before/after metrics comparison
  - Evidence of improvement (or explanation if no improvement)

### Final Summary
- Complete results table including original and improved models
- Final best model identification
- Key insights from the analysis
- Performance metrics of the final selected model

---

## Success Criteria

The notebook should demonstrate:
- ✓ Clean, well-structured code with proper comments
- ✓ All 4 required models implemented and evaluated
- ✓ Comprehensive evaluation with 5+ metrics per model
- ✓ Professional visualizations (confusion matrices, ROC curves, comparison charts)
- ✓ Evidence-based model comparison and selection
- ✓ Two distinct improvement strategies with documented results
- ✓ Clear markdown explanations for each section
- ✓ Reproducible results (random seeds set)
- ✓ All cells execute without errors
- ✓ Logical flow from data loading to final model selection
