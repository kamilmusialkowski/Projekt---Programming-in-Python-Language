import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SKLearnKMeans
import time
from kmeans.custom_kmeans import CustomKMeans

# konfiguracja parametrów testu
N_KLASTROW_GEN = 4
N_PROBEK = 600
RANDOM_STATE = 42


def generate_data(n_samples, n_clusters, random_state):
    """generowanie syntetycznych danych"""
    X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters,
                           cluster_std=0.60, random_state=random_state)
    print(f"Wygenerowano {X.shape[0]} próbek danych z {X.shape[1]} cechami.")
    return X


def plot_elbow_method(X):
    """
    wizualizacja Metody Łokcia (Elbow Method) dla określenia optymalnego K;
    używamy implementacji SKLearn dla szybkości obliczeń wielu wariantów
    """
    print("\ngenerowanie metody łokcia")
    inertias = []
    k_range = range(1, 10)

    for k in k_range:
        # używamy SKLearn dla szybkości przy wielokrotnym uruchamianiu
        kmeans = SKLearnKMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-', markersize=8, linewidth=2)
    plt.xlabel('Liczba klastrów (K)')
    plt.ylabel('Inercja (Suma kwadratów odległości)')
    plt.title('Metoda Łokcia do wyboru optymalnego K')
    plt.grid(True)



def visualize_iterations(X, custom_kmeans):
    """
    wizualizuje proces przesuwania się centroidów w kolejnych iteracjach
    """
    history = custom_kmeans.history
    n_iterations = len(history)

    # wybieramy maksymalnie 4 kroki do wyświetlenia, aby wykres był czytelny
    # zawsze pokazujemy: start, środek 1, środek 2, koniec
    if n_iterations < 2:
        return

    steps_to_show = [0]  # zawsze pierwsza iteracja
    if n_iterations > 2:
        steps_to_show.append(n_iterations // 3)
    if n_iterations > 5:
        steps_to_show.append((n_iterations * 2) // 3)
    steps_to_show.append(n_iterations - 1)  # zawsze ostatnia iteracja

    steps_to_show = sorted(list(set(steps_to_show)))  # usuń duplikaty i posortuj

    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(15, 4))
    fig.suptitle(f"Ewolucja Centroidów (Własna Implementacja) - {n_iterations} iteracji łącznie")

    if len(steps_to_show) == 1:
        axes = [axes]

    final_labels = custom_kmeans.labels

    for idx, step_idx in enumerate(steps_to_show):
        ax = axes[idx]
        centroids = history[step_idx]

        # rysujemy punkty (używamy finalnych etykiet dla czytelności,
        # choć w trakcie iteracji one się zmieniają)
        ax.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', s=20, alpha=0.3)

        # rysujemy centroidy w danym kroku
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   marker='X', s=200, c='red', edgecolor='black', label='Centroidy')

        ax.set_title(f"Iteracja {step_idx}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])



def compare_implementations(X, n_clusters, random_state):
    """główne porównanie z wykresami wynikowymi"""

    print(f"\n--- PORÓWNANIE K-MEANS (K={n_clusters}) ---")

    # 1. własna implementacja
    start_time_custom = time.time()
    custom_kmeans = CustomKMeans(n_clusters=n_clusters, random_state=random_state)
    custom_labels = custom_kmeans.fit_predict(X)
    end_time_custom = time.time()
    custom_time = end_time_custom - start_time_custom
    print(f"1. Własna K-Means: Czas wykonania: {custom_time:.4f}s")

    # 2. SKLearn
    start_time_sklearn = time.time()
    sklearn_kmeans = SKLearnKMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    sklearn_labels = sklearn_kmeans.fit_predict(X)
    end_time_sklearn = time.time()
    sklearn_time = end_time_sklearn - start_time_sklearn
    print(f"2. SKLearn K-Means: Czas wykonania: {sklearn_time:.4f}s")

    # wizualizacja porównania (oryginalna)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Porównanie K-Means (K={n_clusters}) - Wynik Końcowy")

    # własna
    axes[0].scatter(X[:, 0], X[:, 1], c=custom_labels, cmap='viridis', s=50, alpha=0.8)
    axes[0].scatter(custom_kmeans.centroids[:, 0], custom_kmeans.centroids[:, 1],
                    marker='X', s=250, color='red', label='Centroidy')
    axes[0].set_title(f"Własna Implementacja ({custom_time:.4f}s)")
    axes[0].legend()

    # SKLearn
    axes[1].scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis', s=50, alpha=0.8)
    axes[1].scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1],
                    marker='X', s=250, color='red', label='Centroidy')
    axes[1].set_title(f"SKLearn Implementacja ({sklearn_time:.4f}s)")
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    # zwracamy obiekt custom_kmeans, aby użyć go do wizualizacji historii
    return custom_kmeans


if __name__ == "__main__":
    # 1. generowanie danych
    X = generate_data(n_samples=N_PROBEK, n_clusters=N_KLASTROW_GEN, random_state=RANDOM_STATE)

    # 2. wizualizacja Metody Łokcia (przed właściwym klastrowaniem)
    # pozwala zobaczyć, czy K=4 jest faktycznie optymalne
    plot_elbow_method(X)

    # 3. wykonanie porównania i pobranie modelu
    custom_model = compare_implementations(X, N_KLASTROW_GEN, RANDOM_STATE)

    # 4. wizualizacja historii iteracji (jak centroidy wędrowały)
    visualize_iterations(X, custom_model)

    # 5. wyświetl wszystkie okna naraz na samym końcu
    plt.show()
