import numpy as np
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- Медианная фильтрация изображений ---
def denoise_with_median(X, size=2):
    X_denoised = np.zeros_like(X)
    for i in range(X.shape[0]):
        img = X[i].reshape(28, 28)
        filtered = median_filter(img, size=size)
        X_denoised[i] = filtered.flatten()
    return X_denoised

# --- Экстрактор признаков: медианная фильтрация + PCA ---
class PCADenoizedFeatureExtractor:
    def __init__(self, pca_components: int):
        self.pca_components = pca_components
        self.pca = PCA(n_components=self.pca_components)

    def fit(self, X):
        self.pca.fit(X / 255.0)

    def transform(self, X):
        denoized = denoise_with_median(X, size=2)
        return self.pca.transform(denoized / 255.0)

# --- Основной пайплайн решения ---
def denoize_pca_solution(X_train, X_test, y_train):
    feature_extractor = PCADenoizedFeatureExtractor(pca_components=20)
    feature_extractor.fit(X_train)
    train_features = feature_extractor.transform(X_train)
    test_features = feature_extractor.transform(X_test)

    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=5000)
    model.fit(train_features, y_train)
    y_pred = model.predict(test_features)
    return y_pred

def main(X, y):
    """Для краткости пропустим часть с загрузкой данных"""

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Предсказание и метрика
    y_pred = denoize_pca_solution(X_train, X_test, y_train)
    acc = accuracy_score(y_test, y_pred)
    print(f"Denoize PCA solution accuracy: {acc:.4f}")
