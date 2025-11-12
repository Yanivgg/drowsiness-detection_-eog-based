# הוראות הרצה - Phase 2: Feature Engineering & ML

## סקירה כללית

**Phase 2** משתמש ב-**feature engineering** מסורתי + מודלי ML (Random Forest, SVM) לזיהוי drowsiness.

- **Train**: נבדקים 01-06, 08-10 (16 קבצים)
- **Test**: נבדק 07 (זהה ל-Phase 1 CNN!)
- **Window**: 16 שניות (כמו CNN_16s הטוב ביותר)
- **Stride**: 8 שניות (50% overlap)
- **Features**: 63 features (time, frequency, non-linear, EOG-specific)

---

## שלב 1: חילוץ Features (מקומי - Windows)

### 1.1 וידוא שקבצי ה-MAT קיימים

```bash
dir data\processed\files\*.mat
```

אמור להיות **20 קבצים** (נבדקים 01-10, כל אחד 2 trials).

### 1.2 הרצת חילוץ Features

```bash
cd phase_2_project
python feature_extraction.py
```

**זמן משוער**: ~10 דקות

**Output**:
- `features/train_features_16s.csv` (~17 MB, 16,280 windows)
- `features/test_features_16s.csv` (~2 MB, 1,818 windows)

### 1.3 בדיקת תקינות

```bash
python -c "import pandas as pd; train=pd.read_csv('features/train_features_16s.csv'); test=pd.read_csv('features/test_features_16s.csv'); print(f'Train: {len(train)} rows, {train.Label.sum()} drowsy'); print(f'Test: {len(test)} rows, {test.Label.sum()} drowsy')"
```

**תוצאה מצופה**:
```
Train: 16280 rows, 1012 drowsy (6.22%)
Test: 1818 rows, 445 drowsy (24.48%)
```

---

## שלב 2: הכנת חבילה ל-Colab

### 2.1 הרצת סקריפט Packaging

```bash
python prepare_phase2_colab.py
```

**Output**: `phase2_colab_package.zip` (~19 MB)

**תוכן החבילה**:
- `train_features_16s.csv` - נתוני train
- `test_features_16s.csv` - נתוני test
- `train_ml_colab.ipynb` - מחברת אימון
- `train_ml_colab.py` - גיבוי (קוד Python)
- `phase2_README.md` - תיעוד
- `feature_engineering/*.py` - מודולי features (לעיון)

---

## שלב 3: העלאה ל-Colab והרצה

### 3.1 פתיחת Colab חדש

1. גש ל-[Google Colab](https://colab.research.google.com/)
2. לחץ על **File** → **Upload notebook**
3. העלה את `train_ml_colab.ipynb`

### 3.2 העלאת קבצי נתונים

**אופציה A: העלאה ישירה** (מומלץ)
```python
# בתא הראשון של Colab:
from google.colab import files
uploaded = files.upload()  # בחר את phase2_colab_package.zip
!unzip -q phase2_colab_package.zip
!ls -lh *.csv
```

**אופציה B: דרך Google Drive**
1. העלה את `phase2_colab_package.zip` ל-Drive
2. רץ בColab:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/phase2_colab_package.zip .
!unzip -q phase2_colab_package.zip
```

### 3.3 הרצת המחברת

1. **Mount Drive** (תא 2): לשמירת מודלים ותוצאות
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Load Data** (תא 3): טעינת features
   - בודק שהקבצים עם העמודות הנכונות
   - מפריד metadata מ-features

3. **Train Random Forest** (תא 4): אימון RF
   - זמן: ~2-3 דקות
   - `class_weight='balanced'` למטפל ב-imbalance

4. **Feature Importance** (תא 5): ניתוח חשיבות
   - Top 20 features
   - גרף + CSV

5. **Train SVM** (תא 6): אימון SVM
   - זמן: ~5-7 דקות
   - RBF kernel

6. **Comparison** (תא 7): השוואת מודלים
   - RF vs SVM
   - גרפים + טבלה

7. **Save Results** (תא 8): שמירה ל-Drive
   - Models (`.joblib`)
   - Results (`.mat`, `.csv`)
   - Plots (`.png`)

### 3.4 תוצאות מצופות

**Random Forest**:
- Kappa: ~0.25-0.35
- Recall: ~20-30%
- Precision: ~40-60%

**SVM**:
- דומה ל-RF או מעט נמוך יותר

**זמן כולל**: ~10-15 דקות (CPU בלבד, לא צריך GPU!)

---

## שלב 4: הורדת תוצאות

### 4.1 קבצים שנשמרו ב-Drive

כל הקבצים נמצאים ב:
```
/content/drive/MyDrive/drowsiness_ml_results/
```

**קבצים**:
- `random_forest_model.joblib` - מודל RF מאומן
- `svm_model.joblib` - מודל SVM מאומן
- `feature_importance.csv` - חשיבות features
- `model_comparison.csv` - השוואת ביצועים
- `predictions_subject07.csv` - חיזויים על test set
- `ml_test_results.mat` - תוצאות (פורמט MATLAB)
- `training_summary.txt` - סיכום טקסטואלי
- `*.png` - גרפים

### 4.2 הורדה

**מתוך Colab**:
```python
from google.colab import files
files.download('/content/drive/MyDrive/drowsiness_ml_results/training_summary.txt')
```

**או פשוט גש ל-Google Drive** ותוריד את התיקייה `drowsiness_ml_results/`.

---

## שלב 5: השוואה עם Phase 1 (CNN)

### 5.1 טעינת תוצאות CNN

מתוך Phase 1, אתה צריך את התוצאות של **CNN_16s** על נבדק 07:

```
/content/drive/MyDrive/microsleep_results/CNN_16s_test_results.mat
```

### 5.2 השוואה ידנית

| Model | Kappa | Precision | Recall | F1-Score |
|-------|-------|-----------|--------|----------|
| **CNN_16s** | 0.394 | 54.9% | 32.7% | 0.410 |
| **Random Forest** | ~0.30 | ~50% | ~25% | ~0.33 |
| **SVM** | ~0.28 | ~48% | ~23% | ~0.31 |

**מסקנות**:
- ✅ CNN טוב יותר (learned features vs. engineered)
- ✅ RF קרוב יחסית, אבל יש פער
- ✅ **ML interpretable** - יודעים איזה features חשובים!
- ✅ ML מהיר יותר לinference (ללא GPU)

### 5.3 Feature Importance Insights

בדוק את `feature_importance.csv` - אילו features הכי חשובים?

**צפוי**:
- Frequency features (delta, theta) - חשובים לdrowsiness
- EOG blink features - מעיד על ירידה בקשב
- Non-linear features - מורכבות של האות

---

## Troubleshooting

### בעיה: "FileNotFoundError: train_features_16s.csv"

**פתרון**: וודא שהעלית את הzip ופתחת אותו:
```python
!unzip -q phase2_colab_package.zip
!ls -lh *.csv
```

### בעיה: "KeyError: 'Label'"

**פתרון**: הקבצים נוצרו לא נכון. רוץ מחדש:
```bash
# מקומי:
python phase_2_project/feature_extraction.py
```

### בעיה: "No module named 'sklearn'"

**פתרון**: Colab אמור לכלול sklearn. אם לא:
```python
!pip install -q scikit-learn joblib
```

### בעיה: "Memory error" בSVM

**פתרון**: הקטן את train set:
```python
# בתא Load Data, הוסף:
from sklearn.model_selection import train_test_split
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train, y_train, train_size=0.5, random_state=42, stratify=y_train
)
```

---

## קבצים חשובים

```
phase_2_project/
├── feature_extraction.py          # (מקומי) חילוץ features
├── train_ml_colab.ipynb          # (Colab) אימון ML
├── train_ml_colab.py             # גיבוי Python
├── prepare_phase2_colab.py       # packaging
├── INSTRUCTIONS.md               # הקובץ הזה!
├── README.md                     # תיעוד כללי
├── requirements_phase2.txt       # dependencies
├── features/
│   ├── train_features_16s.csv    # 16,280 windows
│   └── test_features_16s.csv     # 1,818 windows
└── feature_engineering/          # מודולי features
    ├── time_domain.py
    ├── frequency_domain.py
    ├── nonlinear.py
    └── eog_specific.py
```

---

## סיכום תהליך

1. ✅ **מקומי**: חילוץ features (`feature_extraction.py`)
2. ✅ **מקומי**: packaging (`prepare_phase2_colab.py`)
3. ✅ **Colab**: העלאה + unzip
4. ✅ **Colab**: הרצת `train_ml_colab.ipynb`
5. ✅ **Drive**: הורדת תוצאות
6. ✅ **השוואה**: CNN vs ML

---

## שאלות נפוצות

**Q: למה Phase 2 משתמש בנבדקים 08-10 לtrain ולא לvalidation?**

A: במודלי ML אין צורך בvalidation set נפרד כי:
- אימון מהיר (דקות, לא שעות)
- אפשר לעשות cross-validation
- נבדק 07 הוא test מצוין (הרבה drowsy events)

**Q: איך אני יכול לשנות את ה-threshold לlabel assignment?**

A: ב-`feature_extraction.py`, שנה את הפרמטר:
```python
def assign_window_label(labels_in_window, threshold=0.1):  # ← כאן
```
- `0.1` = 10% drowsy → label=1
- `0.05` = 5% drowsy → יותר דוגמאות drowsy
- `0.2` = 20% drowsy → פחות דוגמאות drowsy

**Q: למה stride=8s ולא 1s כמו ב-CNN?**

A: חילוץ features איטי (0.03s/window). עם stride=1s:
- 180+ שעות לכל הקבצים
- עם stride=8s (50% overlap):
- ~10 דקות לכל הקבצים
- עדיין מספיק augmentation!

**Q: אפשר להוסיף features נוספים?**

A: כן! ערוך את המודולים ב-`feature_engineering/`:
- `time_domain.py` - time features
- `frequency_domain.py` - spectral features
- `nonlinear.py` - entropy, fractals
- `eog_specific.py` - blinks, saccades

רוץ מחדש את `feature_extraction.py`.

---

## תמיכה

אם נתקעת:
1. בדוק את ה-error message
2. חפש ב-Troubleshooting למעלה
3. וודא שכל הקבצים קיימים
4. נסה להריץ מחדש את התאים

בהצלחה! 🚀
