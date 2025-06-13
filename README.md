# De Prompt a Máscara: Segmentación Inteligente con Transformers

## Seminario UV - Visión Artificial Avanzada

Este seminario práctico explora cómo **convertir descripciones en texto natural en máscaras de segmentación precisas** utilizando la librería **Transformers de Hugging Face**.

---

## Objetivos del Seminario

- **Modelos multimodales**: Comprensión de la combinación entre visión y lenguaje natural
- **Transformers en práctica**: Uso profesional de la librería líder en IA
- **Segmentación zero-shot**: Detección de objetos sin necesidad de entrenar modelos
- **Optimización**: Gestión de modelos en local y procesamiento en lotes
- **Pipelines de IA**: Integración de múltiples modelos para tareas complejas

---

## Tecnologías Utilizadas

### Librerías Principales
- **Transformers**: Framework principal para modelos preentrenados
- **PyTorch**: Backend de deep learning
- **PIL/OpenCV**: Procesamiento de imágenes
- **Matplotlib**: Visualización de resultados

### Modelos de IA
- **Grounding DINO**: Detección de objetos guiada por texto
- **SAM (Segment Anything Model)**: Segmentación precisa a nivel píxel  
- **CLIP**: Clasificación imagen-texto zero-shot
- **BLIP**: Generación automática de descripciones
- **Qwen2.5-VL**: Modelo conversacional multimodal

---

## Contenido del Seminario

### Notebook 1: `GD_SAM.ipynb` - Enfoque Clásico
**Pipeline de múltiples modelos especializados**

```
Texto → Grounding DINO → Bounding Boxes → SAM → Máscaras Precisas
         ↓
    CLIP + BLIP (análisis adicional)
```

**Contenido:**
- Configuración y descarga de modelos en local
- CLIP para clasificación imagen-texto
- BLIP para generación de descripciones  
- Grounding DINO para detección guiada por texto
- SAM para segmentación pixel-perfect
- Integración completa del pipeline

### Notebook 2: `QWEN_SAM.ipynb` - Enfoque Moderno
**Pipeline conversacional simplificado**

```
Conversación Natural → Qwen2.5-VL → JSON Estructurado → SAM → Máscaras
```

**Ventajas del enfoque moderno:**
- **Menor complejidad**: 2 modelos vs 4 del enfoque clásico
- **Mayor inteligencia**: Comprensión de instrucciones complejas
- **Interfaz conversacional**: Interacción natural en lenguaje humano
- **Salida estructurada**: Formato JSON automático

---

## Requisitos e Instalación

### Requisitos del Sistema
- **GPU**: Compatible con CUDA (recomendado)
- **RAM**: Mínimo 16GB
- **Almacenamiento**: Aproximadamente 15GB para modelos

### Instalación
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate opencv-python matplotlib huggingface_hub[hf_xet] qwen-vl-utils ipykernel ipywidgets
```

### Estructura del Proyecto
```
prompt_to_mask/
├── GD_SAM.ipynb          # Enfoque clásico (múltiples modelos)
├── QWEN_SAM.ipynb        # Enfoque moderno (modelo unificado)
├── assets/               # Imágenes de ejemplo
├── README.md             # Documentación
└── how_to_install_local.txt
```

---

## Fundamentos Teóricos
- **Zero-shot Learning**: Capacidad de generalización sin entrenamiento específico
- **IA Multimodal**: Integración de modalidades visuales y textuales  
- **Ingeniería de Pipelines**: Conexión eficiente de modelos especializados
- **Optimización Práctica**: Gestión local de modelos y técnicas de aceleración

