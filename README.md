# Introducción a la Ciencia de Datos - Tarea 2

Repositorio del grupo 18 del curso IntroCD - Introducción a la Ciencia de Datos.

## Descripción del Proyecto

Este proyecto implementa análisis de datos y modelos de machine learning para clasificar discursos políticos de las elecciones presidenciales de Estados Unidos 2020. El proyecto incluye:

- Análisis de Componentes Principales (PCA) para visualización de datos
- Implementación de modelos de clasificación (Naive Bayes, Decision Trees)
- Experimentos con diferentes configuraciones de datos y parámetros
- Visualizaciones y métricas de evaluación

## Prerrequisito

Es necesario tener instalada la fuente *Latin Modern Roman* en el sistema. De no ser así, la misma puede descargarse desde: https://mirror.ctan.org/fonts/lm.zip.
Para instalar la fuente:

1. Descomprima el ZIP
2. Ir a `fonts > opentype > public > lm`
3. Seleccionar todos los archivos, hacer click derecho e instalar.

## Informes del Proyecto

Los informes detallados de las tareas se encuentran en el directorio `docs/`:

- **`docs/IntroCD_Tarea1.pdf`** - Informe completo de la Tarea 1
- **`docs/IntroCD_Tarea2.pdf`** - Informe completo de la Tarea 2

## Estructura del Proyecto

```
introcd-tarea2/
├── data/                           # Datos de entrada
│   └── us_2020_election_speeches.csv
├── docs/                          # Documentación
│   ├── IntroCD_Tarea1.pdf         # Informe de la Tarea 1
│   └── Tarea 2 - Introducción a la Ciencia de Datos.pdf
├── experiments/                   # Configuraciones de experimentos
├── research/                      # Scripts de investigación
├── results/                       # Resultados de experimentos
├── utils/                         # Utilidades y módulos
├── main.py                        # CLI principal
└── requirements.txt               # Dependencias
```

## Setup del Repositorio

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/delpodiego/introcd-tarea2.git
   cd introcd-tarea2
   ```

2. **Crear un entorno virtual (recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## Uso del CLI

El proyecto utiliza Typer para crear una interfaz de línea de comandos. El archivo `main.py` proporciona dos comandos principales:

### 1. Análisis PCA

Ejecuta el análisis de componentes principales y genera visualizaciones:

```bash
python main.py pca
```

Este comando:
- Carga y procesa los datos de discursos
- Genera gráficos de distribución de datos (pie chart)
- Aplica Bag of Words y TF-IDF
- Realiza PCA y genera visualizaciones
- Guarda los resultados en el directorio actual

### 2. Ejecutar Experimentos

Ejecuta experimentos de machine learning basados en archivos de configuración YAML:

```bash
python main.py run-experiment <path-to-experiment.yaml>
```

**Ejemplos de uso:**
```bash
# Experimento con Naive Bayes en discursos completos
python main.py run-experiment experiments/naive-bayes-speeches.yaml

# Experimento con Decision Tree en oraciones
python main.py run-experiment experiments/decision-tree-sentences.yaml

# Experimento con Naive Bayes con datos desbalanceados
python main.py run-experiment experiments/naive-bayes-desbalance.yaml
```

### Estructura de Archivos de Experimento

Los archivos YAML en `experiments/` definen la configuración de cada experimento:

```yaml
experiment_name: "naive-bayes-speeches"
data: "speeches"  # original, speeches, sentences
speakers: ["Donald Trump", "Joe Biden", "Mike Pence"]
model: "naive_bayes"
scoring: "f1_macro"
refit: "f1_macro"
grid_search:
  alpha: [0.1, 1.0, 10.0]
```


## Resultados

Los resultados de los experimentos se guardan automáticamente en el directorio `results/` con la siguiente estructura:

```
results/
└── <experiment-name>/
    ├── best_params.yaml          # Mejores parámetros encontrados
    ├── metrics.yaml              # Métricas de evaluación
    ├── conf_matrix_train.png     # Matriz de confusión (train)
    ├── conf_matrix_test.png      # Matriz de confusión (test)
    ├── violin_plot.png           # Gráfico de violín cross-validation
    ├── strip_plot.png            # Gráfico de strip cross-validation
    └── bar_plot.png              # Gráfico de barras cross-validation
```

