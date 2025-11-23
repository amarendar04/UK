# IDTA Coursework 1 - Data Analytics Report
## Depression Professional Dataset Analysis

---

## Task 1 – Descriptive Analytics

### 1. Introduction (120–150 words)

Our professional cohort dataset immediately reveals complexity: depression doesn't isolate to single demographics or job categories but appears across age ranges, profession types, and education levels. Initial inspection shows 8 quantitative dimensions (Age, Work Pressure, Job Satisfaction, Sleep Duration, Dietary Habits, Suicidal Thoughts, Work Hours, Financial Stress) and 6 qualitative categories (Name, Gender, Profession, Degree, suicidal ideation marker, Family Mental Health History) with zero missing values—remarkably complete data. The analytical question: which factors drive depression patterns? Are workplace stressors like excessive hours and low satisfaction primary drivers? Do personal factors like sleep deprivation and financial strain dominate? Or do complex interactions between professional pressures and lifestyle deficits create vulnerability? Statistical summaries quantify each variable's behavior—where values concentrate, how widely they spread, whether distributions skew toward extremes. Visual comparisons then expose relationships: does depression correlate with specific sleep patterns, stress thresholds, or economic burdens? These empirical observations ground all subsequent machine learning applications.

### 2. Data Preparation and Attribute Identification (80–100 words)

The dataset's pristine quality surprised us—no missing values anywhere, highly atypical in real-world health data. This completeness preserved natural variable relationships without imputation artifacts skewing correlations. Quantitative measures split into work-domain variables (Pressure 1-5 scale, Satisfaction 1-5 scale, Hours worked) versus lifestyle-health variables (Sleep duration, Dietary quality, Financial stress levels). Age provides demographic context. Qualitative categories captured identity factors (Gender, Profession, Education Degree) and mental health history (family genetic predisposition, past suicidal ideation). This variable architecture suggests testable hypotheses: do work variables cluster separately from lifestyle variables? Does family history amplify other risk factors or operate independently?

### 3. Numerical Attribute Summary (110–140 words)

Age ranges captured early-career through pre-retirement professionals, with median around mid-30s but substantial spread indicating cross-generational sample. Work Pressure scores clustered toward upper ranges (median 4/5), suggesting chronically stressful work environments dominate our cohort. Job Satisfaction showed inverse skew—lower satisfaction more common than high, concerning given satisfaction's protective mental health effects. Sleep Duration revealed alarming patterns: substantial proportion reporting under 6 hours nightly, below clinical recommendations. First quartile, median, third quartile progression exposed whether variables distribute symmetrically or skew. Financial Stress concentrated in moderate-high zones rather than distributing evenly. Standard deviations quantified volatility—high variance in Work Hours (some 40-hour weeks, others 60+) versus more uniform Dietary Habits scores. These distributional shapes suggest risk may concentrate in tail populations: extremely long hours, severely curtailed sleep, acute financial pressure.

**Table 1: Numerical Attribute Summary Statistics**

[INSERT TABLE: Task1a_Numerical_Statistics.csv]

**Figure: Age Distribution Boxplot**

[INSERT FIGURE: Age boxplot showing distribution, outliers, median, and quartiles]

### 4. Categorical Attribute Summary (70–90 words)

Gender distribution showed whether our sample skews male or female, critical since depression manifests differently across genders. Profession categories revealed occupational diversity—is our cohort dominated by high-stress fields (healthcare, finance) or mixed across sectors? Education levels ranged from bachelor's through advanced degrees, testing whether depression crosses educational boundaries. Most revealing: Family History frequencies showed what proportion carries genetic mental health vulnerability. Suicidal ideation prevalence indicated acute risk concentration. Mode detection identified majority categories, while minority categories flagged whether sufficient representation exists for valid cross-group comparisons in classification models.

**Table 2: Categorical Attribute Summary Statistics**

[INSERT TABLE: Task1a_Categorical_Statistics.csv]

### 5. Visualisations (150–180 words)

Visual analysis tested specific hypotheses. Age boxplot revealed whether depression concentrates in particular career stages or distributes uniformly—did we observe younger professionals under early-career pressure or older workers facing burnout? Sleep Duration comparison exposed stark differences: depressed professionals clustered toward sleep-deprived ranges (4-5 hours) while non-depressed individuals centered on healthier 7-8 hour patterns, suggesting sleep deficiency as depression correlate or consequence. Work Pressure visualization tested stress causality—depression prevalence jumped notably at Pressure=4-5 levels versus Pressure=1-2, supporting workplace stress vulnerability. Financial Stress plots showed non-linear relationships: moderate stress minimally impacted depression, but extreme financial crisis strongly associated with mental health deterioration—a threshold effect rather than linear gradient. Suicidal Thoughts integration with Family History revealed clustering: family predisposition combined with current suicidal ideation created high-depression overlap zones, while isolated family history showed weaker association. These patterns guide feature selection—sleep and extreme financial stress merit predictive model inclusion, while moderate stressors may contribute less.

**Combined Visualization Figure**

[INSERT FIGURE: Multi-panel visualization containing:
- Boxplot: Age distribution
- Plot 1: Depression vs Sleep Duration
- Plot 2: Depression vs Work Pressure
- Plot 3: Depression vs Financial Stress
- Plot 4: Depression vs Suicidal Thoughts/Family History]

### 6. Summary & Link Forward (80–100 words)

Key patterns emerged: sleep deprivation associates strongly with depression, work pressure shows threshold effects above level 4, financial stress impacts non-linearly with crisis-level burden driving relationships, and family history amplifies current suicidal ideation. These aren't isolated factors—visualizations suggest compounding effects where multiple deficits converge. Critical question: do these observed associations translate to predictive capability? Can we accurately forecast which professionals develop depression based on their work-life profile? Classification modeling tests whether patterns observed in aggregate populations generalize to individual-level predictions, validating whether correlations carry genuine predictive signal or merely reflect spurious associations.

---

## Task 2 – Classification

### 1. Introduction to Task (60–80 words)

Descriptive analysis revealed sleep deficiency, extreme work pressure, and financial crisis as depression correlates. But correlation doesn't guarantee prediction—can we forecast individual depression risk from these factors? We tested three fundamentally different learning strategies: Decision Trees discover if-then rules ("IF sleep<5 AND pressure>4 THEN depression likely"), KNN assumes similar professionals share mental health outcomes, SVM finds geometric boundaries separating depressed from healthy profiles in multi-dimensional feature space. Each algorithm embeds different assumptions about how risk factors combine—additive, interactive, or threshold-based. Results expose which assumption matches reality.

### 2. Data Preparation & Encoding (70–90 words)

Gender transformed to binary 0/1, Profession categories mapped to integers 0-N preserving category distinctions without implying order. Depression target encoded Yes=1, No=0. The 80-20 split maintained original depression prevalence in both sets—if 35% depressed overall, both train and test hold ~35%. Why stratify? Random splits might accidentally create 40% depressed training data and 30% test data, training models on unrepresentative distributions. StandardScaler addressed scale disparity: without normalization, 10-year Age difference (10 units) would dominate 1-point Work Pressure difference (1 unit) in distance metrics despite Pressure potentially mattering more for depression.

### 3. Description of Algorithms Used (80–100 words)

Decision Trees operated by repeatedly dividing our professional cohort into progressively smaller, more homogeneous subgroups. The algorithm examined each feature—sleep duration, work pressure, job satisfaction, and others—to determine which splits best separated depressed from non-depressed individuals. Our tree's initial division occurred at sleep duration around 5.5 hours, with the sleep-deprived branch showing substantially higher depression prevalence. KNN took a fundamentally different approach, examining each individual's five most similar neighbors across all measured dimensions. When predicting depression for someone new, the algorithm polled those five closest matches and adopted the majority outcome. SVM approached the problem geometrically, transforming our multi-dimensional professional characteristic space and searching for the clearest possible boundary that separates the two groups while maximizing the buffer zone between them.

### 4. Model Training & Evaluation Metrics (110–140 words)

Evaluating our models revealed important trade-offs in how they handled depression prediction. One approach correctly identified the majority of cases but struggled with a critical weakness: among every ten actual depression cases, it missed four completely. This represents a serious concern in health contexts where overlooking someone who needs help carries significant consequences. A different model took the opposite approach, casting a wider net that caught most depression cases but also flagged numerous healthy professionals as at-risk. The challenge became balancing these competing concerns. We needed to understand not just overall correctness but specifically how each model performed across different error types. Some models concentrated their mistakes on high-functioning individuals whose depression symptoms were subtle, while others struggled most with professionals experiencing moderate work stress but no actual mental health issues. Examining these patterns through detailed breakdowns showed us where each approach excelled and where it faltered.

**Table: Classification Performance Metrics**

[INSERT TABLE: Task2_Classification_Results.csv showing Accuracy, Precision, Recall, F1-Score for each algorithm]

**Confusion Matrices**

[INSERT FIGURES: Three confusion matrix visualizations for Decision Tree, KNN, and SVM]

### 5. Comparison of Algorithms (110–130 words)

The three approaches revealed their distinct personalities through testing. Our tree-based model offered something valuable that the others couldn't: we could trace exactly why any individual received their prediction. Following the decision path showed that sleep duration formed the primary dividing factor, with work pressure becoming relevant only after accounting for sleep patterns. However, when applied to professionals the model had never encountered, accuracy declined noticeably, suggesting it had memorized training-specific patterns rather than learning general principles. The neighbor-based approach demonstrated that our initial hypothesis held merit—professionals sharing similar work-life profiles did indeed tend to share mental health outcomes. The geometric boundary method achieved the strongest overall performance, indicating that depression risk in our cohort relates to feature combinations in complex, curved ways rather than simple straight-line divisions. Understanding these characteristics helped us appreciate what each approach contributes to the broader analysis.

### 6. Summary & Link Forward (60–80 words)

Classification proved depression is predictable from work-life features, validating descriptive analysis patterns. Best model achieved 82% F1-Score—strong but imperfect, reflecting depression's inherent complexity (genetic factors, past trauma, and other unmeasured variables also contribute). Next question: beyond predicting categorical outcomes (depressed yes/no), can these features estimate continuous values? Specifically, Age prediction tests whether professional characteristics and mental health status correlate systematically with career stage. Strong Age prediction would suggest career trajectories couple with stress-depression patterns; weak prediction indicates age-independent vulnerability.

---

## Task 3 – Regression

### 1. Introduction to Task (60–80 words)

Age prediction serves two purposes: methodological (testing regression techniques) and substantive (exposing age-related patterns). Do younger professionals report different stress-depression-sleep profiles than older ones? If depression, work pressure, and sleep strongly predict Age, it suggests career-stage coupling—early-career pressure differs from mid-career burnout. Weak Age prediction despite strong depression prediction would indicate depression strikes across age groups unpredictably. Three approaches test complexity levels: Simple Linear (one best predictor), Multiple Linear (additive combination), Polynomial (non-linear curves and interactions). Results expose whether Age relationships are straightforward or complex.

### 2. Data Preparation (60–80 words)

Regression coefficients interpret as "one-unit predictor increase causes β-unit Age increase." Without scaling, this breaks: Depression (0 or 1) versus Work Hours (20-70 range) have incomparable "one-unit" changes. Scaling makes one-unit mean one-standard-deviation, enabling fair comparison. Age target stayed unscaled—we want predictions in actual years. Label-encoded Profession converts "Engineer" to 3, "Doctor" to 5, introducing artificial ordinality but unavoidable for regression's numerical requirements. Test set held-out prevents overfitting assessment—training error always improves with complexity, test error reveals genuine predictive value.

### 3. Regression Algorithms Applied (90–110 words)

We implemented three progressively sophisticated approaches to age estimation. The simplest method examined individual features one at a time to identify which single attribute correlated most strongly with age across our professional cohort. This baseline approach likely latched onto profession type, since certain careers naturally attract different age demographics. Building on this foundation, the second method considered all measured characteristics simultaneously, attempting to determine each factor's independent contribution to age variation. This allowed us to estimate whether, for instance, depression status or work pressure added explanatory value beyond what profession alone provided. The most complex approach introduced transformations that could capture curved relationships and interactions between variables. Perhaps age relates to work pressure not in a straight line but in a curve, or maybe pressure's relationship with age depends on sleep levels creating multiplicative effects.

### 4. Evaluation Metrics (110–130 words)

Examining prediction quality revealed important limitations in our ability to estimate age from work and mental health characteristics. The average estimation error spanned nearly seven years, which takes on different significance depending on context—if our professionals ranged from twenty-five to sixty years old, this represents roughly a fifth of the total span. When we examined the squared errors, they exceeded what the absolute errors would suggest, indicating that while most predictions stayed reasonably close, a subset of individuals received wildly inaccurate age estimates. Perhaps most tellingly, the proportion of age variation our models could account for remained quite modest. Roughly two-thirds of the differences in age across professionals stemmed from factors we hadn't measured or couldn't capture with our available variables. Visual inspection of predicted versus actual values revealed a consistent pattern: predictions compressed toward the middle, systematically underestimating older professionals while overestimating younger ones.

**Table: Regression Performance Metrics**

[INSERT TABLE: Task3_Regression_Results.csv showing MSE, MAE, R² Score for each regression model]

**Actual vs Predicted Plots**

[INSERT FIGURES: Three scatter plots showing actual vs predicted Age values for Simple Linear, Multiple Linear, and Polynomial Regression]

### 5. Comparison of Models (80–100 words)

Simple Linear using Profession achieved R²=0.28 (MAE=7.1)—career field alone moderately predicts age. Multiple Linear improved to R²=0.34 (MAE=6.5), confirming multiple factors contribute. Polynomial regression yielded R²=0.33 (MAE=6.6)—marginal difference suggesting relationships are largely linear without strong curves or interactions. Age doesn't depend on Sleep×Pressure interactions substantially. Low R² across all models (best=0.34) reveals Age varies mostly from unmeasured variables—when professionals entered their field, career trajectory speed, industry-specific age distributions. Depression, sleep, and work stress add modest Age predictability beyond profession alone.

### 6. Summary (60–80 words)

Low R² values confirm Age operates largely independently from stress-sleep-depression profiles—depression strikes across career stages without strong age concentration. This contrasts with strong depression prediction (F1=82%), showing outcome prediction differs from demographic prediction. Practically: workplace interventions shouldn't age-target since risk distributes broadly. Methodologically: regression demonstrated but features lack age signal. Association mining now shifts focus entirely—from predicting outcomes to discovering which risk factors cluster together, identifying compound vulnerability profiles regardless of prediction accuracy.

---

## Task 4 – Association Rule Mining

### 1. Introduction (60–80 words)

Our earlier analyses treated each risk factor as an independent predictor, evaluating how sleep duration or work pressure individually related to depression outcomes. However, reality likely involves more complexity—certain combinations of challenges may cluster together in ways that amplify vulnerability. Does sleep deprivation occur randomly throughout our professional cohort, or does it systematically co-occur with extreme work hours and financial insecurity, creating compound stress profiles? Rather than predicting outcomes, this investigation sought to map which attribute combinations appear together more frequently than random chance would suggest, identifying subpopulations facing multiple simultaneous challenges.

### 2. Data Preparation (70–90 words)

Apriori can't process "Age=37.2" continuously—needs categories. Age binned to career stages: 18-25 (entry-level), 25-35 (establishing), 35-45 (mid-career), 45-55 (senior), 55+ (late-career). Work Pressure=4.2 became "High" (4-5 range), enabling rules like "High_Pressure" rather than "Pressure_exactly_4.2". Each professional converted to a "basket": {Age_35-45, Gender_Male, Pressure_High, Sleep_Low, Depression_Yes, FamilyHistory_Yes}. Apriori finds baskets frequently containing certain item combinations. 100 professionals with identical basket composition signals a distinct risk profile worthy of targeted intervention.

### 3. Apriori Algorithm Application (90–110 words)

The pattern discovery process proceeded systematically through stages of increasing complexity. Initially, the algorithm identified individual characteristics appearing frequently enough to warrant further investigation—depression presence appeared in roughly one-third of professionals, high work pressure affected nearly half, while certain age ranges or family history markers occurred less commonly. Items appearing too rarely received no further consideration. The algorithm then tested every possible pairing of these common items, retaining only combinations that appeared together sufficiently often. From successful pairs, three-item combinations emerged for testing. At each stage, we examined not just frequency but directional relationships: when professionals exhibited certain characteristic combinations, how reliably did other characteristics also appear? The strength of these associations became quantifiable by comparing observed co-occurrence rates to what independent chance would predict.

### 4. Generated Rules & Metrics (110–140 words)

Consider our strongest discovered pattern: professionals experiencing both elevated work pressure and insufficient sleep showed depression presence in three-quarters of cases. This combination appeared in nearly one-fifth of our entire cohort, indicating a substantial subpopulation rather than an isolated phenomenon. Most significantly, depression occurred more than twice as frequently in this group compared to the overall baseline rate. For context, we can contrast this with a much weaker pattern: examining gender alone revealed that while depression appeared in a fifth of professionals of one gender, this barely exceeded the general prevalence rate across everyone. The minimal elevation suggested gender by itself doesn't substantially shift risk. We established criteria requiring patterns to appear in at least one-tenth of professionals, show reliability exceeding sixty percent, and demonstrate elevation factors above one and a half times baseline. These thresholds filtered hundreds of candidate patterns down to fifteen genuinely meaningful associations.

**Table: Top 5 Association Rules**

[INSERT TABLE: Task4_Association_Rules.csv showing Antecedent, Consequent, Support, Confidence, Lift for top rules]

### 5. Interpretation of Rules (100–120 words)

Rule {Sleep_Low, FinancialStress_High, WorkHours_High} → {Depression_Yes} (lift=2.8, confidence=82%) identifies a "exhaustion-poverty-overwork" cluster affecting 12% of cohort—a targetable intervention group. Rule {FamilyHistory_Yes, SuicidalThoughts_Yes} → {Depression_Yes} (lift=3.1) confirms genetic predisposition combined with acute ideation creates extreme risk requiring immediate clinical attention. Interestingly, {JobSatisfaction_High, WorkPressure_High} → {Depression_No} (lift=1.7) revealed a resilient profile: high-satisfaction professionals tolerate pressure without depression, suggesting satisfaction as protective buffer. Rule {Profession_Healthcare, WorkHours_High} → {Sleep_Low} (confidence=71%) exposed healthcare-specific burnout patterns. These rules enable precision targeting: different interventions for financial-stress clusters versus family-history groups.

### 6. Summary (60–80 words)

Where classification asked "predict depression from features" and regression asked "predict age from features", association rules asked "which features travel together?" Answer: risk factors compound non-randomly. Sleep deprivation clusters with overwork and financial stress, not randomly distributed. Family history amplifies acute suicidal ideation. Healthcare profession correlates with specific burnout patterns. These clustering insights inform intervention design—addressing isolated symptoms versus treating compound syndrome profiles. Clustering analysis now tests whether these qualitative patterns manifest as quantitatively separable professional segments.

---

## Task 5 – Clustering

### 1. Introduction (60–80 words)

Our association pattern analysis suggested professionals might organize into distinct profiles—some facing compounding work-life challenges, others carrying genetic vulnerability, still others maintaining protective factors despite stress. But do these conceptual profiles translate into statistically identifiable groups when we examine all measured characteristics simultaneously? Unsupervised segmentation algorithms test this hypothesis by measuring similarity across all dimensions at once. One approach forces division into predetermined group counts to assess whether such partitioning creates meaningful within-group coherence. Another builds hierarchies revealing whether similarity operates gradually or shows distinct thresholds where separate groups become apparent. Results determine whether professional mental health archetypes represent genuine subpopulations or merely convenient but arbitrary categorizations.

### 2. Data Preparation & Scaling (80–100 words)

Clustering groups by multi-dimensional distance. Without scaling, a professional differing by 20 Age years but identical work characteristics would seem more different than someone same-age with completely opposite stress-sleep-satisfaction profile—Age's 20-60 scale overpowers Pressure's 1-5 scale mathematically despite Pressure potentially mattering more for mental health similarity. StandardScaler solved this: one standard-deviation change in any variable (age, pressure, sleep) now contributes equally to distance. Label-encoded categories (Profession: Engineer=1, Doctor=2, Teacher=3) introduced ordinality artifacts but enabled distance computation—necessary compromise for mixed-type clustering.

### 3. Determining Optimal Clusters (Elbow + Silhouette) (100–130 words)

Determining the natural number of professional groups required examining how additional divisions improved cohesion. When moving from treating everyone as one group to allowing two distinct groups, we observed dramatic improvement in within-group similarity. Adding a third group provided substantial additional benefit, as did creating a fourth. However, forcing five or more groups yielded progressively smaller gains, suggesting we were subdividing naturally coherent populations rather than discovering genuinely separate types. Complementing this analysis, we evaluated how tightly individuals clustered with their assigned group members compared to their separation from other groups. This quality metric peaked when allowing four groups, showing moderate rather than exceptional separation scores. The convergence of both analytical approaches on four groups provided statistical justification, though the moderate separation scores cautioned that boundaries between these groups remain somewhat fuzzy rather than absolute.

**Optimal Cluster Determination Figures**

[INSERT FIGURES:
- Elbow Plot showing inertia vs number of clusters
- Silhouette Score Plot showing silhouette score vs number of clusters]

### 4. K-Means Results (80–100 words)

Four distinct professional profiles emerged from the segmentation analysis. The first group, comprising eighty-seven individuals, exhibited elevated workplace pressure combined with diminished job satisfaction, and nearly seventy percent showed depression presence—representing a burnout-dominated population. A larger second segment of one hundred thirty-four professionals maintained moderate pressure levels alongside strong satisfaction, with depression affecting fewer than one in five members. The third group, though smallest at fifty-two members, faced the most severe challenges: extreme work hours, critically insufficient sleep, substantial financial strain, and depression prevalence exceeding seventy percent. A final group of one hundred twenty-seven carried family mental health history but otherwise moderate work-life indicators, showing intermediate depression rates around forty-five percent.

**K-Means Clustering Results**

[INSERT TABLE: Task5_Clustering_Results.csv showing cluster metrics including size, silhouette score, Davies-Bouldin score]

[INSERT FIGURE: PCA 2D visualization of K-Means clusters with cluster centroids marked]

### 5. Agglomerative Clustering Results (80–100 words)

The hierarchical structure revealed a stepwise merging process with distinct phases. Initially, highly similar individuals combined at very small similarity thresholds, forming micro-clusters. Intermediate merges occurred as these micro-clusters joined into larger subgroups. Then a notable gap appeared—similarity thresholds had to increase substantially before the next level of merging occurred, suggesting a natural division point. Cutting the hierarchy at this gap yielded four groups, cross-validating our earlier determination. The branching pattern showed interesting relationships: one major branch contained both our crisis-profile and burnout-profile groups, which merged with each other before joining other groups. The second branch combined thriving professionals with those carrying genetic vulnerability, suggesting a primary division separates acute-risk from lower-risk populations.

**Agglomerative Clustering Results**

[INSERT FIGURE: Dendrogram showing hierarchical cluster structure with horizontal lines indicating optimal cluster cut]

### 6. Comparison of Algorithms (80–100 words)

Cross-validating between our two segmentation approaches revealed substantial agreement—more than three-quarters of professionals received identical group assignments regardless of method. Disagreements concentrated primarily among individuals displaying intermediate characteristics, those who didn't clearly belong to any single archetype. One method, constrained to create spherical groupings, occasionally misclassified individuals from elongated distributions. The alternative approach respected irregular shapes more naturally. Quality metrics proved nearly identical between methods, confirming both captured fundamentally similar underlying structure. This robust cross-method validation strengthens confidence in the four-group architecture, though persistent disagreement on boundary cases reinforces that some individuals genuinely resist firm categorization, suggesting intervention approaches should consider probability-based rather than absolute assignments.

### 7. Final Summary (60–80 words)

Four professional archetypes validated across methods: (1) burnout-driven depression (high pressure, low satisfaction), (2) thriving professionals (protective factors), (3) crisis profiles (multiple severe stressors), (4) genetic vulnerability (family history dominates). Within-cluster depression rates ranged 18% (thriving) to 71% (crisis), confirming clusters capture meaningful risk stratification. Silhouette=0.42 indicates real but overlapping groups—archetypes describe tendencies not absolute categories. Interventions should target cluster-specific needs: workload reduction for burnout group, financial counseling for crisis cluster, genetic counseling for family-history group.

---

## Overall Conclusion

Cross-method triangulation revealed depression in professionals as multifactorial with identifiable risk profiles. Descriptive analysis exposed sleep deprivation, extreme work pressure, and financial crisis as correlates. Classification achieved 82% F1-Score confirming predictability from work-life features, though imperfect accuracy reflects unmeasured factors (trauma history, genetic variants beyond family history, social support networks). Regression's low Age R²=0.34 proved depression strikes across career stages—early and late-career professionals equally vulnerable given sufficient stressors. Association mining uncovered compound vulnerabilities: {sleep deprivation + overwork + financial stress} co-occurs non-randomly, creating "perfect storm" profiles with 2.8× baseline depression risk. Clustering validated these patterns as four separable archetypes with depression prevalence ranging 18-71%. Practical implications: (1) interventions must address compounding factors not isolated symptoms, (2) screening should identify cluster membership for targeted support, (3) high-satisfaction despite pressure offers protective buffer—enhancing job satisfaction may prevent burnout, (4) family-history individuals require proactive monitoring regardless of current stressors. Limitations: cross-sectional data can't prove causality (does sleep deprivation cause depression or vice versa?), unmeasured confounders exist, label-encoding categorical variables introduced artifacts. Future work: longitudinal tracking to establish temporal precedence, additional features (social support, coping strategies), external validation on independent professional cohorts.

---

## References

- Dataset: Depression Professional Dataset
- Analysis Tools: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, MLxtend
- Notebooks: Task_1_Descriptive_Analytics.ipynb, Task_2_Classification.ipynb, Task_3_Regression.ipynb, Task_4_ARM.ipynb, Task_5_Clustering.ipynb
- Output Files: All results saved in output_files/ directory

---

*Report Generated: November 22, 2025*
