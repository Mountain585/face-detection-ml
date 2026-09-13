import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Forces matplotlib to run headlessly in Ubuntu terminal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
REAL_DIR = 'real_subset' 
FAKE_DIR = 'fake_subset'
IMG_SIZE = (256, 256)
N_SAMPLES = 2000
RANDOM_STATE = 42

def load_images(directory, n_samples):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in valid_exts]
    files = files[:n_samples]
    images = []
    print(f"Loading {len(files)} images from {directory}...")
    for f in files:
        img_path = os.path.join(directory, f)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
    return np.array(images)

print("--- Step 1: Loading Data ---")
real_images = load_images(REAL_DIR, N_SAMPLES)
fake_images = load_images(FAKE_DIR, N_SAMPLES)

y = np.concatenate([np.zeros(len(real_images)), np.ones(len(fake_images))])
X_images = np.concatenate([real_images, fake_images])
print(f"Loaded {len(real_images)} real and {len(fake_images)} fake images.\n")

# ==========================================
# 2. COMPRESSION SHORTCUT CHECK
# ==========================================
print("--- Step 2: Compression Shortcut Check ---")
def check_compression_shortcut(images_real, images_fake, sample_size=200):
    lap_var_real = np.mean([cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for img in images_real[:sample_size]])
    lap_var_fake = np.mean([cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for img in images_fake[:sample_size]])
    print(f"Average Laplacian Variance (Real): {lap_var_real:.2f}")
    print(f"Average Laplacian Variance (Fake): {lap_var_fake:.2f}")
    if abs(lap_var_real - lap_var_fake) > 50: 
        print("⚠️ WARNING: Large sharpness disparity detected! Compression shortcut likely.")
        return True
    else:
        print("✅ Sharpness levels are similar.")
        return False

shortcut_detected = check_compression_shortcut(real_images, fake_images)

print("Applying uniform JPEG compression (Q=85) to neutralize compression artifacts...")
def uniform_compress(images, quality=85):
    compressed = []
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for img in images:
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        compressed.append(decimg)
    return np.array(compressed)

X_images = uniform_compress(X_images)
print("Uniform compression applied.\n")

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
print("--- Step 3: Extracting Features ---")

def extract_baseline_features(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / (gray.shape[0] * gray.shape[1]) 
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        features.append(np.concatenate([hist, [mean_val, std_val]]))
    return np.array(features)

def extract_enhanced_features(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        dct = cv2.dct(gray)
        ll = dct[:h//2, :w//2]; lh = dct[:h//2, w//2:]
        hl = dct[h//2:, :w//2]; hh = dct[h//2:, w//2:]
        e_ll = np.sum(ll**2); e_lh = np.sum(lh**2)
        e_hl = np.sum(hl**2); e_hh = np.sum(hh**2)
        total_e = e_ll + e_lh + e_hl + e_hh + 1e-10
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blur
        n_var = np.var(noise)
        n_mean_abs = np.mean(np.abs(noise))
        features.append([e_lh/total_e, e_hl/total_e, e_hh/total_e, n_var, n_mean_abs])
    return np.array(features)

X_baseline = extract_baseline_features(X_images)
X_enhanced = extract_enhanced_features(X_images)
print(f"Baseline features shape: {X_baseline.shape}")
print(f"Enhanced features shape: {X_enhanced.shape}\n")

# ==========================================
# 4. MODEL TRAINING & COMPARISON
# ==========================================
print("--- Step 4: Training and Comparing Models ---")

X_train_b, X_test_b, X_train_e, X_test_e, y_train, y_test, _, img_test = train_test_split(
    X_baseline, X_enhanced, y, X_images, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler_b = StandardScaler().fit(X_train_b)
scaler_e = StandardScaler().fit(X_train_e)

X_train_b_sc = scaler_b.transform(X_train_b)
X_test_b_sc = scaler_b.transform(X_test_b)
X_train_e_sc = scaler_e.transform(X_train_e)
X_test_e_sc = scaler_e.transform(X_test_e)

classifiers = {
    "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
}

results = []
best_score = -1
best_y_pred = None
best_feat_name = ""
best_model_name = ""

for name, clf in classifiers.items():
    for feat_name, X_tr, X_te in [("Baseline", X_train_b_sc, X_test_b_sc), ("Enhanced", X_train_e_sc, X_test_e_sc)]:
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        results.append({"Model": name, "Features": feat_name, "Accuracy": acc, "F1-Score": f1})
        print(f"{name:15} | {feat_name:10} | Acc: {acc:.4f} | F1: {f1:.4f}")
        if f1 > best_score:
            best_score = f1
            best_y_pred = y_pred
            best_feat_name = feat_name
            best_model_name = name

print(f"\n🏆 Best Model: {best_model_name} with {best_feat_name} features (F1: {best_score:.4f})\n")

# ==========================================
# 5. FINAL EVALUATION & SAVING VISUALS
# ==========================================
print("--- Step 5: Saving Final Visuals ---")

# 1. Confusion Matrix
y_pred_final = best_y_pred
cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title(f'Confusion Matrix - Best Model ({best_feat_name} Features)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("✅ Saved: confusion_matrix.png")

# 2. Misclassified Samples
misclassified_indices = np.where(y_pred_final != y_test)[0]
fp_indices = misclassified_indices[y_test[misclassified_indices] == 0]
fn_indices = misclassified_indices[y_test[misclassified_indices] == 1]

fp_samples = img_test[fp_indices[:4]]
fn_samples = img_test[fn_indices[:4]]

fig, axes = plt.subplots(2, max(len(fp_samples), len(fn_samples), 1), figsize=(12, 6))
fig.suptitle('Misclassified Samples (Top: Real->Fake, Bottom: Fake->Real)', fontsize=14)

for i, img in enumerate(fp_samples):
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"True: Real\nPred: Fake", color='red')
    axes[0, i].axis('off')

for i, img in enumerate(fn_samples):
    axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"True: Fake\nPred: Real", color='blue')
    axes[1, i].axis('off')

for j in range(len(fp_samples), axes.shape[1]):
    axes[0, j].axis('off')
for j in range(len(fn_samples), axes.shape[1]):
    axes[1, j].axis('off')

plt.tight_layout()
plt.savefig('misclassified_samples.png', dpi=300)
plt.close()
print("✅ Saved: misclassified_samples.png")
print("\n🎉 Project Complete! Check your folder for the PNG images.")
