K-MEANS CLUSTERING - GUI VERSION
================================

ARCHIVO: kmeans_gui.py

CARACTERÍSTICAS:
---------------
✓ Interfaz gráfica moderna con PySimpleGUI
✓ Opción para selección automática de K óptimo (basado en Silhouette Score)
✓ Opción para selección manual de K
✓ Configuración de parámetros desde la interfaz:
  - Método de inicialización (kmeans++, farthest, first_k)
  - Escalado de datos
  - Detección de outliers
  - Random state
✓ Visualización de resultados en tiempo real:
  - Gráfico de clustering con centroides
  - Método del codo (Elbow Method)
  - Gráfico de Silhouette o detección de outliers
✓ Métricas mostradas:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
  - Inertia
  - Accuracy (si K coincide con clases verdaderas)
✓ Log detallado del proceso

CÓMO USAR:
----------
1. Ejecutar el script:
   python kmeans_gui.py

2. En la ventana:
   - Seleccionar "Automático" para que el sistema encuentre el K óptimo
     O seleccionar "Manual" e ingresar un valor de K
   
   - Configurar opciones:
     * Escalar datos (recomendado)
     * Detectar outliers
     * Método de inicialización
     * Random state
   
   - Hacer clic en "Ejecutar Análisis"
   
   - Ver resultados:
     * Métricas en el panel izquierdo
     * Log del proceso
     * Gráficos en el panel derecho

SELECCIÓN AUTOMÁTICA DE K:
--------------------------
El sistema prueba K desde 2 hasta n-1 y selecciona el K con el mayor
Silhouette Score. Esto generalmente da buenos resultados, pero puede
no coincidir exactamente con el número de clases verdaderas.

SELECCIÓN MANUAL:
-----------------
Si conoces el número de clases verdaderas (en este caso 3), puedes
seleccionar "Manual" e ingresar ese valor para obtener mejor accuracy.

REQUISITOS:
-----------
- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- PySimpleGUI (se instala automáticamente si no está presente)

NOTAS:
------
- Los gráficos se muestran directamente en la ventana
- El log muestra todo el proceso paso a paso
- Los resultados se actualizan cada vez que ejecutas el análisis
- Puedes cambiar la configuración y volver a ejecutar

