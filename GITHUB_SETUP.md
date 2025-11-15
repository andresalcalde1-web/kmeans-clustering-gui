# Instrucciones para subir a GitHub

## Opción 1: Usando GitHub Desktop (Recomendado para principiantes)

1. Descarga e instala [GitHub Desktop](https://desktop.github.com/)
2. Inicia sesión con tu cuenta de GitHub
3. Ve a File > Add Local Repository
4. Selecciona la carpeta donde está `kmeans_gui.py`
5. Haz clic en "Publish repository"
6. Elige un nombre para el repositorio (ej: `kmeans-clustering-gui`)
7. Selecciona la organización o tu cuenta personal
8. Haz clic en "Publish Repository"

## Opción 2: Usando Git desde la línea de comandos

### Si no tienes Git instalado:
1. Descarga Git desde: https://git-scm.com/downloads
2. Instálalo con las opciones por defecto

### Pasos para subir el código:

1. **Abre una terminal en la carpeta del proyecto** (donde está `kmeans_gui.py`)

2. **Inicializa el repositorio Git:**
```bash
git init
```

3. **Añade todos los archivos:**
```bash
git add .
```

4. **Haz el primer commit:**
```bash
git commit -m "Initial commit: K-Means Clustering GUI"
```

5. **Crea un repositorio en GitHub:**
   - Ve a https://github.com/new
   - Nombre del repositorio: `kmeans-clustering-gui` (o el que prefieras)
   - Descripción: "Interfaz gráfica para análisis de clustering K-Means con visualización interactiva"
   - Elige si será público o privado
   - **NO marques** "Initialize with README" (ya tenemos uno)
   - Haz clic en "Create repository"

6. **Conecta tu repositorio local con GitHub:**
```bash
git remote add origin https://github.com/TU_USUARIO/kmeans-clustering-gui.git
```
(Reemplaza `TU_USUARIO` con tu nombre de usuario de GitHub)

7. **Sube el código:**
```bash
git branch -M main
git push -u origin main
```

## Opción 3: Usando GitHub CLI (gh)

Si tienes GitHub CLI instalado:

```bash
gh repo create kmeans-clustering-gui --public --source=. --remote=origin --push
```

## Estructura de archivos que se subirán:

- `kmeans_gui.py` - Script principal
- `requirements.txt` - Dependencias
- `README.md` - Documentación
- `.gitignore` - Archivos a ignorar

## Sugerencias para el repositorio:

**Nombre sugerido:** `kmeans-clustering-gui`

**Descripción sugerida:**
```
Interfaz gráfica interactiva para análisis de clustering K-Means con visualización de resultados, detección de outliers y múltiples métodos de inicialización.
```

**Topics/Tags sugeridos:**
- `machine-learning`
- `kmeans`
- `clustering`
- `python`
- `tkinter`
- `scikit-learn`
- `data-visualization`
- `gui`

