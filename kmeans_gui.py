import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.optimize import linear_sum_assignment
import io
import warnings

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk

# ===== DATOS DE ENTRADA =====
X_raw = np.array([
    [2.32, 3.14], [2.33, 3.11], [1.97, 2.03], [2.37, 3.17],
    [2.58, 2.60], [1.98, 2.05], [2.01, 1.95], [2.27, 3.14],
    [2.64, 2.57], [2.59, 1.59], [1.93, 1.99], [2.64, 2.58]
])

true_labels = np.array([3, 3, 1, 3, 2, 1, 1, 3, 2, 2, 1, 2])
original_indices = np.array([11, 10, 1, 9, 6, 3, 2, 12, 5, 8, 4, 7])
n = X_raw.shape[0]
K_TRUE = len(np.unique(true_labels))

# ===== FUNCIONES =====

def detect_outliers_simple(X, method='iqr'):
    """Detección simple y robusta para datasets pequeños."""
    n = X.shape[0]
    
    if method == 'iqr':
        outlier_mask = np.zeros(n, dtype=bool)
        for dim in range(X.shape[1]):
            q1 = np.percentile(X[:, dim], 25)
            q3 = np.percentile(X[:, dim], 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask |= (X[:, dim] < lower_bound) | (X[:, dim] > upper_bound)
        
        scores = np.zeros(n)
        for i in range(n):
            if outlier_mask[i]:
                for dim in range(X.shape[1]):
                    q1 = np.percentile(X[:, dim], 25)
                    q3 = np.percentile(X[:, dim], 75)
                    iqr = q3 - q1
                    if iqr > 0:
                        if X[i, dim] < q1:
                            scores[i] += abs(X[i, dim] - q1) / iqr
                        elif X[i, dim] > q3:
                            scores[i] += abs(X[i, dim] - q3) / iqr
            else:
                scores[i] = 0
        return outlier_mask, scores
    
    elif method == 'distance':
        centroid = X.mean(axis=0)
        distances = np.array([np.linalg.norm(x - centroid) for x in X])
        q1_dist = np.percentile(distances, 25)
        q3_dist = np.percentile(distances, 75)
        iqr_dist = q3_dist - q1_dist
        threshold = q3_dist + 1.5 * iqr_dist
        return distances > threshold, distances
    else:
        raise ValueError(f"Método '{method}' no soportado")

def farthest_point_initialization(X, K, random_state=42):
    """Inicialización por puntos más lejanos."""
    if K > len(X):
        raise ValueError(f"K={K} no puede ser mayor que n={len(X)}")
    
    centroids = [X.mean(axis=0)]
    for _ in range(K - 1):
        distances = np.array([
            min([np.linalg.norm(x - c) for c in centroids])
            for x in X
        ])
        farthest_idx = np.argmax(distances)
        centroids.append(X[farthest_idx])
    
    centroids = np.array(centroids)
    unique_centroids = np.unique(centroids, axis=0)
    if len(unique_centroids) < K:
        rng = np.random.RandomState(random_state)
        centroids += rng.normal(0, 0.01, centroids.shape)
    return centroids

def match_labels_safe(predicted_labels, true_labels):
    """Mapeo seguro con validación estricta de cardinalidad."""
    unique_pred = np.unique(predicted_labels)
    unique_true = np.unique(true_labels)
    
    if len(unique_pred) != len(unique_true):
        return None, None
    
    confusion = np.zeros((len(unique_pred), len(unique_true)))
    for i, pred_label in enumerate(unique_pred):
        for j, true_label in enumerate(unique_true):
            confusion[i, j] = np.sum((predicted_labels == pred_label) & 
                                     (true_labels == true_label))
    
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {unique_pred[i]: unique_true[j] for i, j in zip(row_ind, col_ind)}
    mapped_labels = np.array([mapping[label] for label in predicted_labels])
    return mapped_labels, mapping

def find_optimal_k(X, k_range=range(2, 8), random_state=42):
    """Búsqueda de K óptimo."""
    if max(k_range) >= len(X):
        k_range = range(2, len(X))
    
    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(X)
        
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
        else:
            silhouette = 0
            calinski = 0
        
        results.append({
            'k': k,
            'silhouette': silhouette,
            'calinski': calinski,
            'inertia': kmeans.inertia_
        })
    
    # Find optimal K based on silhouette score (highest)
    optimal_idx = np.argmax([r['silhouette'] for r in results])
    optimal_k = results[optimal_idx]['k']
    
    return optimal_k, results

def validate_clustering_simple(X, labels, centroids):
    """Validación simple del clustering."""
    issues = []
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < len(centroids):
        issues.append(f"{len(centroids) - len(unique_labels)} cluster(s) vacío(s)")
    
    if len(unique_labels) > 1:
        ch_score = calinski_harabasz_score(X, labels)
        if ch_score < 10:
            issues.append(f"Baja separación entre clusters (CH={ch_score:.2f})")
    
    return issues

def run_kmeans_analysis(config, output_text):
    """Ejecuta el análisis de K-means y retorna resultados."""
    results = {}
    log_messages = []
    
    def log(msg, level='INFO'):
        """Registra un mensaje de log con formato mejorado."""
        # Formatear según el tipo de mensaje
        if level == 'SECTION':
            # Secciones principales
            log_messages.append(f"\n{'─' * 70}\n{msg}\n{'─' * 70}")
        elif level == 'SUCCESS':
            # Mensajes de éxito
            log_messages.append(f"  ✓ {msg}")
        elif level == 'WARNING':
            # Advertencias
            log_messages.append(f"  ⚠ {msg}")
        elif level == 'INFO':
            # Información general
            log_messages.append(f"  • {msg}")
        else:
            log_messages.append(f"  {msg}")
        
        if output_text:
            output_text.append(log_messages[-1] + "\n")
    
    log("=" * 80)
    log("K-MEANS CLUSTERING - ANÁLISIS")
    log("=" * 80)
    log(f"Dataset: {n} observaciones, {X_raw.shape[1]} dimensiones", 'INFO')
    
    # Determinar K
    if config['auto_k']:
        log("Buscando K óptimo...", 'INFO')
        scaler_temp = StandardScaler()
        X_temp = scaler_temp.fit_transform(X_raw) if config['scale_data'] else X_raw.copy()
        optimal_k, k_results = find_optimal_k(X_temp, range(2, min(n, 8)), config['random_state'])
        config['K'] = optimal_k
        log(f"K óptimo encontrado: {optimal_k} (basado en Silhouette Score)", 'SUCCESS')
    else:
        config['K'] = int(config['K'])
        log(f"K manual: {config['K']}", 'INFO')
    
    log(f"Configuración: K={config['K']}, init='{config['init_method']}', "
        f"scale={config['scale_data']}, random_state={config['random_state']}", 'INFO')
    
    # 1. ESCALADO
    if config['scale_data']:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        log("Datos escalados correctamente", 'SUCCESS')
    else:
        X = X_raw.copy()
        scaler = None
        log("Datos sin escalar", 'INFO')
    
    # 2. DETECCIÓN DE OUTLIERS
    n_outliers = 0
    outlier_mask = None
    outlier_scores = None
    if config['detect_outliers']:
        log("DETECCIÓN DE OUTLIERS", 'SECTION')
        
        actual_method = config['outlier_method']
        if n < 30 and actual_method not in ['iqr', 'distance']:
            actual_method = 'iqr'
        
        outlier_mask, outlier_scores = detect_outliers_simple(X, method=actual_method)
        n_outliers = np.sum(outlier_mask)
        log(f"Método: {actual_method.upper()}", 'INFO')
        log(f"Outliers detectados: {n_outliers}/{n}", 'INFO')
    
    # 3. INICIALIZACIÓN
    log("INICIALIZACIÓN DE CENTROIDES", 'SECTION')
    
    if config['init_method'] == 'farthest':
        initial_centroids = farthest_point_initialization(X, config['K'], random_state=config['random_state'])
        n_init = 1
        log("Método: Farthest Point - Selecciona puntos más lejanos entre sí", 'INFO')
    elif config['init_method'] == 'kmeans++':
        initial_centroids = 'k-means++'
        n_init = 10
        log("Método: K-means++ - Centroides bien distribuidos (sklearn)", 'INFO')
        log("Reinicios: 10 para mayor estabilidad", 'INFO')
    else:
        initial_centroids = X[:config['K']].copy()
        n_init = 1
        log(f"Método: Primeros {config['K']} puntos del dataset", 'INFO')
    
    # 4. EJECUTAR K-MEANS
    log("EJECUTANDO ALGORITMO K-MEANS", 'SECTION')
    
    kmeans = KMeans(
        n_clusters=config['K'],
        init=initial_centroids,
        n_init=n_init,
        max_iter=300,
        tol=1e-6,
        random_state=config['random_state']
    )
    
    kmeans.fit(X)
    log(f"✓ Convergencia alcanzada en {kmeans.n_iter_} iteraciones", 'SUCCESS')
    log(f"Inercia (distancia total a centroides): {kmeans.inertia_:.6f}", 'INFO')
    
    # 5. VALIDACIÓN
    log("VALIDACIÓN DEL CLUSTERING", 'SECTION')
    
    issues = validate_clustering_simple(X, kmeans.labels_, kmeans.cluster_centers_)
    if issues:
        for issue in issues:
            log(f"{issue}", 'WARNING')
    else:
        log("No se detectaron problemas en el clustering", 'SUCCESS')
    
    # 6. MÉTRICAS
    log("MÉTRICAS DE CALIDAD DEL CLUSTERING", 'SECTION')
    
    unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    if len(unique_labels) > 1:
        silhouette = silhouette_score(X, kmeans.labels_)
        davies_bouldin = davies_bouldin_score(X, kmeans.labels_)
        calinski = calinski_harabasz_score(X, kmeans.labels_)
        
        log(f"Silhouette Score: {silhouette:.4f} (mejor → 1, peor → -1)", 'INFO')
        log(f"Davies-Bouldin Index: {davies_bouldin:.4f} (mejor → 0)", 'INFO')
        log(f"Calinski-Harabasz Index: {calinski:.4f} (mayor es mejor)", 'INFO')
    else:
        silhouette = davies_bouldin = calinski = 0
        log("Solo un cluster activo - No se pueden calcular métricas", 'WARNING')
    
    log("Distribución de puntos por cluster:", 'INFO')
    for label, count in zip(unique_labels, counts):
        log(f"  Cluster {label + 1}: {count} puntos ({count/n*100:.1f}%)", 'INFO')
    
    # 7. COMPARACIÓN CON GROUND TRUTH
    log("COMPARACIÓN CON CLASES VERDADERAS", 'SECTION')
    
    mapped_labels, mapping = match_labels_safe(kmeans.labels_, true_labels)
    accuracy = None
    
    if mapped_labels is not None:
        log(f"Mapeo exitoso entre clusters predichos y clases verdaderas", 'SUCCESS')
        accuracy = np.sum(mapped_labels == true_labels) / n * 100
        log(f"Exactitud: {np.sum(mapped_labels == true_labels)}/{n} puntos = {accuracy:.2f}%", 'INFO')
        
        comparison_df = pd.DataFrame({
            'Índice Original': original_indices,
            'Clase Verdadera': true_labels,
            'Clase Asignada': mapped_labels,
            'Correcto': ['✓' if m == t else '✗' for m, t in zip(mapped_labels, true_labels)]
        })
        log("\nTabla de Comparación:", 'INFO')
        log(comparison_df.to_string(index=False), 'INFO')
    else:
        log("No se pudo calcular accuracy - Número de clusters predichos ≠ clases verdaderas", 'WARNING')
        log("Matriz de confusión:", 'INFO')
        for pred_label in np.unique(kmeans.labels_):
            log(f"\nCluster {pred_label + 1}:")
            for true_label in np.unique(true_labels):
                count = np.sum((kmeans.labels_ == pred_label) & (true_labels == true_label))
                if count > 0:
                    log(f"  Clase {true_label}: {count} puntos", 'INFO')
    
    # 8. PREPARAR DATOS PARA VISUALIZACIÓN
    if config['scale_data']:
        centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    else:
        centroids_original = kmeans.cluster_centers_
    
    # Calcular métricas para gráfico del codo
    k_range = list(range(2, min(n, 8)))
    inertia_values = []
    silhouette_values = []
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=config['random_state'])
        labels_temp = kmeans_temp.fit_predict(X)
        inertia_values.append(kmeans_temp.inertia_)
        if len(np.unique(labels_temp)) > 1:
            silhouette_values.append(silhouette_score(X, labels_temp))
        else:
            silhouette_values.append(0)
    
    results = {
        'kmeans': kmeans,
        'X': X,
        'X_raw': X_raw,
        'labels': kmeans.labels_,
        'centroids_original': centroids_original,
        'scaler': scaler,
        'config': config,
        'metrics': {
            'silhouette': silhouette if len(unique_labels) > 1 else 0,
            'davies_bouldin': davies_bouldin if len(unique_labels) > 1 else 0,
            'calinski': calinski if len(unique_labels) > 1 else 0,
            'inertia': kmeans.inertia_,
            'accuracy': accuracy
        },
        'k_range': k_range,
        'inertia_values': inertia_values,
        'silhouette_values': silhouette_values,
        'outlier_mask': outlier_mask,
        'outlier_scores': outlier_scores,
        'n_outliers': n_outliers,
        'log_messages': log_messages
    }
    
    log("ANÁLISIS COMPLETADO EXITOSAMENTE", 'SECTION')
    
    # Resumen final
    log("\n📊 RESUMEN DE RESULTADOS:", 'SUCCESS')
    log(f"  K clusters identificados: {config['K']}", 'INFO')
    log(f"  Total de puntos: {n}", 'INFO')
    if silhouette > 0:
        log(f"  Silhouette Score: {silhouette:.4f} (excelente si > 0.5, bueno si > 0.3)", 'INFO')
    if accuracy is not None:
        log(f"  Exactitud vs clases verdaderas: {accuracy:.2f}%", 'INFO')
    if n_outliers > 0:
        log(f"  Outliers detectados: {n_outliers} puntos", 'INFO')
    
    return results

def create_plots(results):
    """Crea los gráficos separados y los convierte a imágenes para la GUI."""
    cmap = plt.colormaps['tab10']
    colors_map = [cmap(i) for i in range(results['config']['K'])]
    
    plots = {}
    
    # Gráfica 1: Clustering
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    for k in range(results['config']['K']):
        mask = results['labels'] == k
        if np.any(mask):
            ax1.scatter(results['X_raw'][mask, 0], results['X_raw'][mask, 1], 
                      c=[colors_map[k]], label=f'Cluster {k + 1}', s=120, alpha=0.7, 
                      edgecolors='black', linewidths=1.5)
    
    for k in range(results['config']['K']):
        ax1.scatter(results['centroids_original'][k, 0], results['centroids_original'][k, 1],
                   c=[colors_map[k]], marker='X', s=500, edgecolors='black', linewidths=2.5)
    
    ax1.set_xlabel('X₁ (Original)', fontsize=11)
    ax1.set_ylabel('X₂ (Original)', fontsize=11)
    ax1.set_title('K-Means Clustering', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
    buf1.seek(0)
    plots['clustering'] = buf1.read()
    plt.close(fig1)
    
    # Gráfica 2: Elbow Method
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(results['k_range'], results['inertia_values'], 'bo-', linewidth=2, markersize=8)
    ax2.axvline(results['config']['K'], color='red', linestyle='--', linewidth=2,
               label=f'K actual={results["config"]["K"]}')
    ax2.set_xlabel('Número de Clusters (K)', fontsize=11)
    ax2.set_ylabel('Inercia', fontsize=11)
    ax2.set_title('Método del Codo', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
    buf2.seek(0)
    plots['elbow'] = buf2.read()
    plt.close(fig2)
    
    # Gráfica 3: Silhouette o Outliers
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    if results['config']['detect_outliers'] and results['n_outliers'] > 0:
        scatter = ax3.scatter(results['X_raw'][:, 0], results['X_raw'][:, 1], 
                            c=results['outlier_scores'], cmap='YlOrRd', s=150, 
                            edgecolors='black', linewidths=1.5)
        plt.colorbar(scatter, ax=ax3, label='Outlier Score')
        ax3.scatter(results['X_raw'][results['outlier_mask'], 0], 
                   results['X_raw'][results['outlier_mask'], 1],
                   marker='o', s=400, facecolors='none', edgecolors='red', linewidths=3)
        ax3.set_xlabel('X₁ (original)', fontsize=11)
        ax3.set_ylabel('X₂ (original)', fontsize=11)
        ax3.set_title('Detección de Outliers', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.plot(results['k_range'], results['silhouette_values'], 'go-', linewidth=2, markersize=8)
        ax3.axvline(results['config']['K'], color='red', linestyle='--', linewidth=2,
                   label=f'K actual={results["config"]["K"]}')
        ax3.set_xlabel('Número de Clusters (K)', fontsize=11)
        ax3.set_ylabel('Silhouette Score', fontsize=11)
        ax3.set_title('Silhouette vs K', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
    buf3.seek(0)
    plots['silhouette'] = buf3.read()
    plt.close(fig3)
    
    return plots

# ===== GUI =====

class ToolTip:
    """Clase para crear tooltips (información al pasar el mouse)."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        widget.bind('<Enter>', self.enter)
        widget.bind('<Leave>', self.leave)
        widget.bind('<ButtonPress>', self.leave)
    
    def enter(self, event=None):
        self.schedule()
    
    def leave(self, event=None):
        self.unschedule()
        self.hidetip()
    
    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)
    
    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)
    
    def showtip(self):
        x, y, cx, cy = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("tahoma", "8", "normal"), wraplength=300)
        label.pack(ipadx=1)
    
    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def show_info_window():
    """Muestra una ventana con información descriptiva de los métodos."""
    info_window = tk.Toplevel()
    info_window.title('Información de Métodos y Opciones')
    info_window.geometry('800x700')
    info_window.minsize(600, 400)  # Tamaño mínimo
    
    # Frame con scrollbar
    canvas = tk.Canvas(info_window, bg='white')
    scrollbar = ttk.Scrollbar(info_window, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Contenido - usar grid en lugar de pack para mejor responsive
    content = ttk.Frame(scrollable_frame)
    content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Título
    title = ttk.Label(content, text='Información de Métodos y Opciones', 
                     font=('Helvetica', 14, 'bold'))
    title.pack(pady=(0, 20), fill=tk.X)
    
    # Diccionario con todas las secciones para crear dinámicamente
    sections_data = {
        'Selección de K': [
            ('• Automático (basado en Silhouette):', 
             'El algoritmo busca automáticamente el número óptimo de clusters (K) evaluando '
             'diferentes valores de K y seleccionando el que maximiza el Silhouette Score. '
             'Este método es recomendado cuando no se conoce a priori el número de clusters.'),
            ('• Manual:', 
             'Permite especificar manualmente el número de clusters. Útil cuando se conoce '
             'el número esperado de grupos en los datos.')
        ],
        'Opciones de Procesamiento': [
            ('• Escalar datos:', 
             'Normaliza los datos usando StandardScaler (media=0, desviación=1). Es recomendado '
             'cuando las características tienen diferentes escalas o unidades. Mejora la '
             'convergencia del algoritmo K-means.'),
            ('• Detectar outliers:', 
             'Identifica puntos anómalos que pueden afectar el clustering. Los outliers se '
             'muestran en la visualización con un círculo rojo.'),
            ('• Método de detección de outliers:', 
             '- IQR (Interquartile Range): Usa el rango intercuartílico para identificar valores '
             'fuera del rango [Q1-1.5*IQR, Q3+1.5*IQR]. Robusto para datasets pequeños.\n'
             '- Distance: Calcula la distancia de cada punto al centroide y detecta outliers '
             'basándose en la distribución de distancias.')
        ],
        'Métodos de Inicialización': [
            ('• kmeans++:', 
             'Método inteligente que selecciona los centroides iniciales de forma que estén '
             'bien distribuidos. Reduce la probabilidad de convergencia a mínimos locales. '
             'Recomendado para la mayoría de casos.'),
            ('• farthest:', 
             'Selecciona los centroides iniciales como los puntos más alejados entre sí. '
             'Útil cuando se quiere garantizar una buena separación inicial.'),
            ('• first_k:', 
             'Usa los primeros K puntos del dataset como centroides iniciales. Método simple '
             'pero puede llevar a resultados subóptimos.'),
            ('• Random State:', 
             'Semilla para la generación de números aleatorios. Usar el mismo valor garantiza '
             'resultados reproducibles.')
        ],
        'Métricas de Calidad': [
            ('• Silhouette Score:', 
             'Mide qué tan bien separados están los clusters. Rango: [-1, 1]. Valores más '
             'altos indican mejor separación. Ideal: cercano a 1.'),
            ('• Davies-Bouldin Index:', 
             'Mide la relación entre la dispersión dentro de los clusters y la separación '
             'entre clusters. Valores más bajos indican mejor clustering. Ideal: cercano a 0.'),
            ('• Calinski-Harabasz Index:', 
             'Ratio de la suma de dispersiones entre clusters y dentro de clusters. Valores '
             'más altos indican mejor clustering.'),
            ('• Inertia:', 
             'Suma de las distancias al cuadrado de cada punto a su centroide más cercano. '
             'Se usa en el método del codo para encontrar el K óptimo.')
        ]
    }
    
    # Crear secciones dinámicamente
    for section_title, items in sections_data.items():
        section = ttk.LabelFrame(content, text=section_title, padding="10")
        section.pack(fill=tk.X, pady=5, padx=0)
        
        for item_title, item_text in items:
            # Título del item
            item_label = ttk.Label(section, text=item_title, 
                                   font=('Helvetica', 10, 'bold'))
            item_label.pack(anchor=tk.W, pady=2, padx=0)
            
            # Texto descriptivo con wrap dinámico
            desc_label = ttk.Label(section, text=item_text, 
                                   justify=tk.LEFT, wraplength=650)
            desc_label.pack(anchor=tk.W, padx=20, pady=2, fill=tk.X)
    
    # Empaquetar canvas y scrollbar con fill y expand
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Botón cerrar en la ventana principal (no en el scrollable)
    button_frame = ttk.Frame(info_window)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    ttk.Button(button_frame, text="Cerrar", command=info_window.destroy).pack()
    
    # Habilitar mousewheel en el canvas
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"
    
    # Binding simple en el canvas (sin recursión)
    canvas.bind("<MouseWheel>", on_mousewheel)
    scrollable_frame.bind("<MouseWheel>", on_mousewheel)
    info_window.bind("<MouseWheel>", on_mousewheel)
    
    # Optimización: cachear todos los labels una sola vez
    all_labels = []
    
    def collect_labels(widget):
        """Recorre una sola vez y colecta todos los labels."""
        if isinstance(widget, ttk.Label):
            all_labels.append(widget)
        for child in widget.winfo_children():
            collect_labels(child)
    
    collect_labels(content)
    
    # Control para evitar múltiples actualizaciones simultáneas
    last_width = [None]
    
    # Hacer que el wraplength se ajuste dinámicamente al redimensionar
    def on_window_resize(event=None):
        # Obtener el ancho actual del canvas
        canvas_width = canvas.winfo_width()
        
        # Solo actualizar si el ancho cambió significativamente (más de 10 px)
        if last_width[0] is not None and abs(canvas_width - last_width[0]) < 10:
            return
        
        last_width[0] = canvas_width
        
        # Ajustar wraplength basado en el ancho disponible
        new_wraplength = max(300, canvas_width - 80)
        
        # Actualizar solo los labels que fueron cacheados (optimizado)
        for label in all_labels:
            try:
                label.configure(wraplength=new_wraplength)
            except tk.TclError:
                pass
    
    # Bind solo a la ventana (mejor que a cada evento Configure)
    info_window.bind("<Configure>", on_window_resize, add=True)
    
    # Llamar una vez al inicio
    info_window.after(150, on_window_resize)

def create_gui():
    """Crea y muestra la interfaz gráfica usando tkinter."""
    root = tk.Tk()
    root.title('K-Means Clustering GUI')
    root.geometry('1200x800')
    
    # Variables
    k_selection_var = tk.StringVar(value='auto')
    scale_var = tk.BooleanVar(value=True)
    detect_outliers_var = tk.BooleanVar(value=True)
    outlier_method_var = tk.StringVar(value='iqr')
    init_method_var = tk.StringVar(value='kmeans++')
    k_value_var = tk.StringVar(value='3')
    random_state_var = tk.StringVar(value='42')
    
    # Main container
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    
    # Title with info button
    title_frame = ttk.Frame(main_frame)
    title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
    title_label = ttk.Label(title_frame, text='K-Means Clustering - Interfaz Gráfica', 
                           font=('Helvetica', 16, 'bold'))
    title_label.pack(side=tk.LEFT)
    info_button = ttk.Button(title_frame, text='ℹ️ Información', command=show_info_window, width=15)
    info_button.pack(side=tk.RIGHT, padx=10)
    ToolTip(info_button, 'Abre una ventana con información detallada sobre los métodos y opciones disponibles')
    
    # Separator
    ttk.Separator(main_frame, orient='horizontal').grid(row=1, column=0, columnspan=2, 
                                                         sticky=(tk.W, tk.E), pady=5)
    
    # Configuration section
    config_label = ttk.Label(main_frame, text='Configuración', font=('Helvetica', 12, 'bold'))
    config_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
    
    # K Selection frame
    k_frame = ttk.LabelFrame(main_frame, text='Selección de K', padding="10")
    k_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)
    
    auto_k_radio = ttk.Radiobutton(k_frame, text='Automático (basado en Silhouette)', 
                    variable=k_selection_var, value='auto',
                    command=lambda: k_value_entry.config(state='disabled'))
    auto_k_radio.grid(row=0, column=0, sticky=tk.W)
    ToolTip(auto_k_radio, 'Busca automáticamente el K óptimo evaluando diferentes valores y seleccionando el que maximiza el Silhouette Score')
    
    manual_k_radio = ttk.Radiobutton(k_frame, text='Manual', variable=k_selection_var, value='manual',
                    command=lambda: k_value_entry.config(state='normal'))
    manual_k_radio.grid(row=1, column=0, sticky=tk.W)
    ToolTip(manual_k_radio, 'Especifica manualmente el número de clusters')
    
    ttk.Label(k_frame, text='K:').grid(row=2, column=0, sticky=tk.W)
    k_value_entry = ttk.Entry(k_frame, textvariable=k_value_var, width=10, state='disabled')
    k_value_entry.grid(row=2, column=1, sticky=tk.W, padx=5)
    ToolTip(k_value_entry, 'Número de clusters a usar (solo cuando se selecciona modo Manual)')
    
    # Options frame
    options_frame = ttk.LabelFrame(main_frame, text='Opciones', padding="10")
    options_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)
    
    scale_check = ttk.Checkbutton(options_frame, text='Escalar datos', variable=scale_var)
    scale_check.grid(row=0, column=0, sticky=tk.W)
    ToolTip(scale_check, 'Normaliza los datos usando StandardScaler (media=0, desviación=1). Recomendado cuando las características tienen diferentes escalas')
    
    outlier_check = ttk.Checkbutton(options_frame, text='Detectar outliers', variable=detect_outliers_var)
    outlier_check.grid(row=1, column=0, sticky=tk.W)
    ToolTip(outlier_check, 'Identifica puntos anómalos que pueden afectar el clustering')
    
    ttk.Label(options_frame, text='Método outliers:').grid(row=2, column=0, sticky=tk.W)
    outlier_combo = ttk.Combobox(options_frame, textvariable=outlier_method_var, 
                                 values=['iqr', 'distance'], width=10, state='readonly')
    outlier_combo.grid(row=2, column=1, sticky=tk.W, padx=5)
    ToolTip(outlier_combo, 'IQR: Usa rango intercuartílico. Distance: Basado en distancia al centroide')
    
    # Initialization frame
    init_frame = ttk.LabelFrame(main_frame, text='Inicialización', padding="10")
    init_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
    
    ttk.Label(init_frame, text='Método:').grid(row=0, column=0, sticky=tk.W)
    init_combo = ttk.Combobox(init_frame, textvariable=init_method_var, 
                             values=['kmeans++', 'farthest', 'first_k'], width=12, state='readonly')
    init_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
    ToolTip(init_combo, 'kmeans++: Centroides bien distribuidos (recomendado). farthest: Puntos más alejados. first_k: Primeros K puntos')
    
    ttk.Label(init_frame, text='Random State:').grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
    random_state_entry = ttk.Entry(init_frame, textvariable=random_state_var, width=10)
    random_state_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=(5, 0))
    ToolTip(random_state_entry, 'Semilla para números aleatorios. Mismo valor = resultados reproducibles')
    
    # Run button
    run_button = ttk.Button(main_frame, text='Ejecutar Análisis', 
                           command=lambda: run_analysis())
    run_button.grid(row=5, column=0, columnspan=2, pady=10)
    
    # Separator
    ttk.Separator(main_frame, orient='horizontal').grid(row=6, column=0, columnspan=2, 
                                                         sticky=(tk.W, tk.E), pady=10)
    
    # Results section
    results_label = ttk.Label(main_frame, text='Resultados', font=('Helvetica', 12, 'bold'))
    results_label.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
    
    # Left column - Metrics and Log
    left_frame = ttk.Frame(main_frame)
    left_frame.grid(row=8, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
    
    metrics_label = ttk.Label(left_frame, text='Métricas:', font=('Helvetica', 11, 'bold'))
    metrics_label.grid(row=0, column=0, sticky=tk.W)
    metrics_text = scrolledtext.ScrolledText(left_frame, width=40, height=8, state='disabled')
    metrics_text.grid(row=1, column=0, pady=(5, 10))
    
    log_label = ttk.Label(left_frame, text='Log del Análisis:', font=('Helvetica', 11, 'bold'))
    log_label.grid(row=2, column=0, sticky=tk.W)
    log_text = scrolledtext.ScrolledText(left_frame, width=40, height=15, state='disabled', 
                                        font=('Courier', 9), wrap=tk.WORD, bg='#f5f5f5')
    log_text.grid(row=3, column=0, pady=(5, 0))
    
    # Configurar estilos de colores para el log
    log_text.tag_config('section', foreground='#1e3a8a', font=('Courier', 9, 'bold'), background='#dbeafe')
    log_text.tag_config('success', foreground='#15803d', font=('Courier', 9, 'bold'))
    log_text.tag_config('warning', foreground='#b45309', font=('Courier', 9, 'bold'))
    log_text.tag_config('info', foreground='#424242')
    
    # Right column - Plots (separated)
    right_frame = ttk.Frame(main_frame)
    right_frame.grid(row=8, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
    
    # Create scrollable frame for plots
    plots_canvas = tk.Canvas(right_frame, bg='white')
    plots_scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=plots_canvas.yview)
    plots_scrollable_frame = ttk.Frame(plots_canvas)
    
    plots_scrollable_frame.bind(
        "<Configure>",
        lambda e: plots_canvas.configure(scrollregion=plots_canvas.bbox("all"))
    )
    
    plots_canvas.create_window((0, 0), window=plots_scrollable_frame, anchor="nw")
    plots_canvas.configure(yscrollcommand=plots_scrollbar.set)
    
    # Variables para almacenar las imágenes originales
    original_plots_data = {}
    
    # Function to create a plot frame
    def create_plot_frame(parent, title, plot_key, main_scroll_canvas):
        """Crea un frame para una gráfica individual."""
        plot_frame = ttk.LabelFrame(parent, text=title, padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        # Header with zoom button
        header = ttk.Frame(plot_frame)
        header.pack(fill=tk.X, pady=(0, 5))
        
        zoom_btn = ttk.Button(header, text='🔍 Ampliar', width=12)
        zoom_btn.pack(side=tk.RIGHT)
        ToolTip(zoom_btn, f'Abre "{title}" en una ventana ampliada')
        
        # Canvas for plot
        canvas = tk.Canvas(plot_frame, width=600, height=400, bg='white', cursor='hand2')
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind click to zoom - capture title and plot_key in closure
        def make_zoom_handler(plot_key, title):
            def show_zoom():
                nonlocal original_plots_data
                if plot_key in original_plots_data:
                    show_zoomed_plot(original_plots_data[plot_key], title)
                else:
                    messagebox.showinfo('Información', 'Primero debe ejecutar un análisis para ver la gráfica ampliada.')
            return show_zoom
        
        zoom_handler = make_zoom_handler(plot_key, title)
        zoom_btn.config(command=zoom_handler)
        canvas.bind('<Button-1>', lambda e: zoom_handler())
        
        # Bind mouse wheel to propagate scroll to main canvas
        def on_canvas_mousewheel(event):
            # Propagate scroll event to main scrollable canvas
            main_scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"  # Prevent event from propagating further
        
        canvas.bind("<MouseWheel>", on_canvas_mousewheel)
        
        # Also bind to the frame and header to ensure scroll works everywhere
        def on_frame_mousewheel(event):
            main_scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"
        
        plot_frame.bind("<MouseWheel>", on_frame_mousewheel)
        header.bind("<MouseWheel>", on_frame_mousewheel)
        
        return plot_frame, canvas
    
    # Create three plot frames
    clustering_frame, clustering_canvas = create_plot_frame(plots_scrollable_frame, 'K-Means Clustering', 'clustering', plots_canvas)
    elbow_frame, elbow_canvas = create_plot_frame(plots_scrollable_frame, 'Método del Codo', 'elbow', plots_canvas)
    silhouette_frame, silhouette_canvas = create_plot_frame(plots_scrollable_frame, 'Silhouette / Outliers', 'silhouette', plots_canvas)
    
    plots_canvas.pack(side="left", fill="both", expand=True)
    plots_scrollbar.pack(side="right", fill="y")
    
    # Enable mouse wheel scrolling on main canvas and scrollable frame
    def on_mousewheel_plots(event):
        plots_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    plots_canvas.bind("<MouseWheel>", on_mousewheel_plots)
    plots_scrollable_frame.bind("<MouseWheel>", on_mousewheel_plots)
    
    # Also bind to right_frame to catch events in empty areas
    right_frame.bind("<MouseWheel>", on_mousewheel_plots)
    
    # Configure grid weights
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=2)
    main_frame.rowconfigure(8, weight=1)
    left_frame.rowconfigure(3, weight=1)
    
    # Function to show zoomed plot
    def show_zoomed_plot(img_data, title):
        """Muestra la gráfica en una ventana ampliada."""
        zoom_window = tk.Toplevel(root)
        zoom_window.title(f'Gráfica Ampliada - {title}')
        zoom_window.geometry('1400x700')
        
        # Create scrollable canvas
        canvas_frame = ttk.Frame(zoom_window, padding="10")
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Load and display full-size image
        img = Image.open(io.BytesIO(img_data))
        photo = ImageTk.PhotoImage(img)
        
        zoom_canvas = tk.Canvas(canvas_frame, width=img.width, height=img.height)
        zoom_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=zoom_canvas.yview)
        h_scrollbar = ttk.Scrollbar(zoom_window, orient=tk.HORIZONTAL, command=zoom_canvas.xview)
        
        zoom_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        zoom_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        zoom_canvas.image = photo  # Keep reference
        zoom_canvas.configure(scrollregion=zoom_canvas.bbox("all"))
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            zoom_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def on_shift_mousewheel(event):
            zoom_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        
        zoom_canvas.bind("<MouseWheel>", on_mousewheel)
        zoom_canvas.bind("<Shift-MouseWheel>", on_shift_mousewheel)
        
        # Close button
        ttk.Button(zoom_window, text="Cerrar", command=zoom_window.destroy).pack(pady=5)
    
    # Store references for the run_analysis function
    def run_analysis():
        nonlocal original_plots_data
        try:
            # Prepare configuration
            config = {
                'auto_k': k_selection_var.get() == 'auto',
                'K': int(k_value_var.get()) if k_selection_var.get() == 'manual' else 3,
                'init_method': init_method_var.get(),
                'scale_data': scale_var.get(),
                'detect_outliers': detect_outliers_var.get(),
                'outlier_method': outlier_method_var.get(),
                'random_state': int(random_state_var.get())
            }
            
            # Update UI
            log_text.config(state='normal')
            log_text.delete(1.0, tk.END)
            log_text.insert(tk.END, 'Ejecutando análisis...\n')
            log_text.config(state='disabled')
            
            metrics_text.config(state='normal')
            metrics_text.delete(1.0, tk.END)
            metrics_text.insert(tk.END, 'Calculando...\n')
            metrics_text.config(state='disabled')
            
            root.update()
            
            # Run analysis
            output_text = []
            results = run_kmeans_analysis(config, output_text)
            
            # Update log
            log_text.config(state='normal')
            log_text.delete(1.0, tk.END)
            
            # Insertar mensajes con colores apropiados
            for msg in results['log_messages']:
                if '─' in msg:  # Sección
                    log_text.insert(tk.END, msg + "\n", 'section')
                elif '✓' in msg:  # Success
                    log_text.insert(tk.END, msg + "\n", 'success')
                elif '⚠' in msg:  # Warning
                    log_text.insert(tk.END, msg + "\n", 'warning')
                elif msg.startswith('  '):  # Info
                    log_text.insert(tk.END, msg + "\n", 'info')
                else:
                    log_text.insert(tk.END, msg + "\n")
            
            log_text.config(state='disabled')
            log_text.see(tk.END)  # Auto-scroll al final
            
            # Update metrics
            metrics_text.config(state='normal')
            metrics_text.delete(1.0, tk.END)
            metrics_text_str = f"Silhouette Score: {results['metrics']['silhouette']:.4f}\n"
            metrics_text_str += f"Davies-Bouldin Index: {results['metrics']['davies_bouldin']:.4f}\n"
            metrics_text_str += f"Calinski-Harabasz Index: {results['metrics']['calinski']:.4f}\n"
            metrics_text_str += f"Inertia: {results['metrics']['inertia']:.6f}\n"
            if results['metrics']['accuracy'] is not None:
                metrics_text_str += f"Accuracy: {results['metrics']['accuracy']:.2f}%\n"
            else:
                metrics_text_str += "Accuracy: N/A (K mismatch)\n"
            metrics_text.insert(tk.END, metrics_text_str)
            metrics_text.config(state='disabled')
            
            # Create and display plots
            plots_dict = create_plots(results)
            
            # Save original images for zoom
            original_plots_data = plots_dict.copy()
            
            # Display each plot in its canvas
            plot_canvases = {
                'clustering': clustering_canvas,
                'elbow': elbow_canvas,
                'silhouette': silhouette_canvas
            }
            
            plot_titles = {
                'clustering': 'K-Means Clustering',
                'elbow': 'Método del Codo',
                'silhouette': 'Silhouette / Outliers'
            }
            
            for plot_key, canvas in plot_canvases.items():
                if plot_key in plots_dict:
                    img_data = plots_dict[plot_key]
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Resize if needed to fit canvas
                    img_width, img_height = img.size
                    canvas_width, canvas_height = 600, 400
                    if img_width > canvas_width or img_height > canvas_height:
                        ratio = min(canvas_width / img_width, canvas_height / img_height)
                        new_width = int(img_width * ratio)
                        new_height = int(img_height * ratio)
                        # Use LANCZOS for compatibility with older PIL versions
                        try:
                            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        except AttributeError:
                            img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(img)
                    canvas.delete("all")
                    canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
                    canvas.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror('Error', f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
    
    root.mainloop()

if __name__ == '__main__':
    create_gui()

