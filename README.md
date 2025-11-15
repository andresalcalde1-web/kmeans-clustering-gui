# K-Means Clustering GUI

Interfaz gráfica interactiva para análisis de clustering K-Means con visualización de resultados, detección de outliers y múltiples métodos de inicialización.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Descripción

Esta aplicación proporciona una interfaz gráfica completa para realizar análisis de clustering K-Means con las siguientes características:

- **Selección inteligente de K**: Automática (basada en Silhouette Score) o manual
- **Preprocesamiento**: Escalado de datos opcional con StandardScaler
- **Detección de outliers**: Métodos IQR y Distance
- **Múltiples inicializaciones**: kmeans++, farthest point, first K points
- **Visualización interactiva**: Gráficas separadas con opción de ampliación
- **Métricas de calidad**: Silhouette Score, Davies-Bouldin, Calinski-Harabasz, Inertia
- **Información contextual**: Tooltips y ventana de ayuda con descripciones detalladas

## 🚀 Requisitos

- Python 3.7 o superior
- Las bibliotecas listadas en `requirements.txt`

## 📦 Instalación

### Método 1: Usando requirements.txt (Recomendado)

```bash
pip install -r requirements.txt
```

### Método 2: Instalación manual

```bash
pip install numpy pandas matplotlib scikit-learn scipy Pillow
```

### Nota sobre tkinter

`tkinter` viene incluido con Python en la mayoría de las distribuciones. Si no está disponible:

- **Windows/Mac:** Generalmente viene preinstalado
- **Linux (Ubuntu/Debian):** `sudo apt-get install python3-tk`
- **Linux (CentOS/RHEL):** `sudo yum install python3-tk`

## 💻 Uso

Ejecutar el script:

```bash
python kmeans_gui.py
```

### Características de la interfaz:

1. **Configuración**: Selecciona el número de clusters, opciones de escalado y detección de outliers
2. **Ejecutar análisis**: Haz clic en "Ejecutar Análisis" para procesar los datos
3. **Visualizar resultados**: 
   - Revisa las métricas en el panel izquierdo
   - Explora las gráficas en el panel derecho
   - Haz clic en cualquier gráfica o usa el botón "🔍 Ampliar" para verla en tamaño completo
4. **Información**: Usa el botón "ℹ️ Información" para ver descripciones detalladas de los métodos

## 🎯 Características principales

### Selección de K
- **Automático**: Busca el K óptimo evaluando diferentes valores y seleccionando el que maximiza el Silhouette Score
- **Manual**: Permite especificar el número de clusters deseado

### Métodos de inicialización
- **kmeans++**: Selección inteligente de centroides (recomendado)
- **farthest**: Puntos más alejados entre sí
- **first_k**: Primeros K puntos del dataset

### Detección de outliers
- **IQR**: Basado en el rango intercuartílico
- **Distance**: Basado en la distancia al centroide

### Visualización
- **Gráficas separadas**: Cada visualización en su propio espacio
- **Scroll intuitivo**: Navegación fluida entre gráficas
- **Ampliación**: Vista detallada de cada gráfica con scrollbars

## 📊 Métricas incluidas

- **Silhouette Score**: Mide la separación entre clusters (mejor: → 1)
- **Davies-Bouldin Index**: Relación entre dispersión y separación (mejor: → 0)
- **Calinski-Harabasz Index**: Ratio de dispersiones (mejor: ↑)
- **Inertia**: Suma de distancias al cuadrado (usado en método del codo)

## 🔧 Estructura del proyecto

```
.
├── kmeans_gui.py          # Script principal
├── requirements.txt       # Dependencias
├── README.md             # Este archivo
├── .gitignore           # Archivos a ignorar en Git
└── GITHUB_SETUP.md      # Instrucciones para GitHub
```

## ✅ Portabilidad

El script es completamente portable:
- ✅ No requiere archivos externos
- ✅ No usa rutas absolutas
- ✅ Funciona en Windows, Mac y Linux
- ✅ Solo necesita las bibliotecas instaladas

## 📚 Dependencias

- `numpy` - Cálculos numéricos
- `pandas` - Manipulación de datos
- `matplotlib` - Visualización
- `scikit-learn` - Algoritmos de machine learning
- `scipy` - Optimización y algoritmos científicos
- `Pillow` - Procesamiento de imágenes
- `tkinter` - Interfaz gráfica (incluido en Python)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## 👤 Autor

Creado como herramienta educativa para análisis de clustering K-Means.

---

⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!

