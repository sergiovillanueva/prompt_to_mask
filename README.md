# De Prompt a Máscara: Segmentación Inteligente con la librería Transformers

<div style="display: flex; align-items: flex-start; gap: 20px;">
  <img src="assets/logo_etse.jpg" alt="ETSE-UV Logo" width="200"/>
  <div style="flex: 1;">
    <h2>ETSE-UV</h2>
    <p><strong>Autor:</strong> Sergio Villanueva López<br>
    <strong>Contacto:</strong> s.villanuevalopez@gmail.com</p>
  </div>
</div>

---

## Descripción del Proyecto

Este proyecto implementa dos enfoques distintos para **segmentación semántica guiada por texto** utilizando modelos de inteligencia artificial preentrenados. El objetivo es convertir descripciones en lenguaje natural en máscaras de segmentación precisas a nivel de píxel.

## Metodologías Implementadas

### Enfoque Clásico: Pipeline Multi-Modelo
**Archivo:** `1_GD_SAM.ipynb`

```
Texto → Grounding DINO → Bounding Boxes → SAM → Máscaras
```

**Modelos utilizados:**
- **CLIP**: Clasificación imagen-texto zero-shot
- **BLIP**: Generación automática de descripciones
- **Grounding DINO**: Detección de objetos guiada por texto  
- **SAM**: Segmentación precisa a nivel píxel

**Características:**
- Pipeline modular con modelos especializados
- Mayor precisión en tareas específicas
- Control granular sobre cada etapa del proceso

### Enfoque Moderno: IA Conversacional
**Archivo:** `2_QWEN_SAM.ipynb`

```
Conversación Natural → Qwen2.5-VL → JSON Estructurado → SAM → Máscaras
```

**Modelos utilizados:**
- **Qwen2.5-VL-3B**: Modelo multimodal conversacional
- **SAM**: Segmentación precisa a nivel píxel

**Ventajas:**
- Comprensión de instrucciones complejas
- Interfaz conversacional natural
- Salida estructurada en formato JSON

## Casos de Uso Prácticos

- **Control de calidad industrial**: Detección de anomalías
- **Análisis urbano**: Medición de superficies en imágenes satelitales
- **Inventario automatizado**: Conteo y clasificación de objetos
- **Análisis geoespacial**: Cálculo de perímetros y áreas

## Requisitos Técnicos

### Especificaciones del Sistema
- **Plataforma**: Google Colab compatible
- **GPU**: Compatible con CUDA (recomendado)
- **RAM**: Mínimo 16GB
- **Almacenamiento**: ~15GB para modelos preentrenados (opcional)

### Instalación

```bash
# PyTorch con soporte CUDA
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Dependencias adicionales
pip install transformers accelerate opencv-python matplotlib huggingface_hub[hf_xet] qwen-vl-utils ipykernel ipywidgets
```

## Estructura del Proyecto

```
prompt_to_mask/
├── 1_GD_SAM.ipynb          # Pipeline clásico multi-modelo
├── 2_QWEN_SAM.ipynb        # Enfoque conversacional moderno
├── assets/                 # Conjunto de imágenes de prueba
│   ├── cars.jpg
│   ├── fruits.jpg
│   ├── park.jpg
│   └── ...
├── README.md
```

## Fundamentos Teóricos

- **Zero-shot Learning**: Capacidad de generalización sin entrenamiento específico
- **IA Multimodal**: Integración de modalidades visuales y textuales
- **Transformers**: Arquitectura neuronal basada en mecanismos de atención
- **Segmentación Semántica**: Clasificación de píxeles por categorías semánticas

---

*Seminario de Visión Artificial - Universidad de Valencia*

