# Documentación de Modelos Entrenados

Este archivo tiene como objetivo recopilar, en un único lugar, los detalles de cada uno de los modelos de redes neuronales que hemos ido entrenando en el proyecto. Para evitar confusiones y pérdidas de información, cada modelo quedará registrado con la fecha de creación y la configuración empleada (arquitectura, conjunto de datos, hiperparámetros, etc.), así como notas relevantes sobre su rendimiento y usos posteriores.

---

## Índice

- [models_2024-12-13](#models_2024-12-13)
- [models_2024-12-14](#models_2024-12-14)
- [models_2025-02-12](#models_2025-02-12)
- [models_2025-02-14](#models_2025-02-14)
- [models_2025-03-05](#models_2025-03-05)
- [models_2025-03-12](#models_2025-03-12)
- [models_2025-03-16](#models_2025-03-16)
- [models_2025-03-17](#models_2025-03-17)

> **Nota**: En caso de que más adelante se generen nuevos modelos, añadirlos como secciones adicionales en este mismo archivo o en un nuevo bloque, siguiendo la misma estructura.

---

## models_2024-12-13

**Descripción general**  
- **Fecha de creación**: 13 de diciembre de 2024  
- **Tipo de modelo**: (ej. PyTorch, Scikit-Learn, TabNet...)  
- **Objetivo**: (ej. Clasificación del movimiento de Pac-Man a partir de dataset inicial)  
- **Conjunto de datos**:  
  - Nombre del dataset: (ej. `pacman_dataset_v1.csv`)  
  - Tamaño del dataset: (número de filas / características)  
  - Observaciones: (ej. primer dataset con pocos ejemplos, sin filtrado de estados)  

**Arquitectura / Hiperparámetros (si aplica)**  
- **Capas**: (ej. 3 capas densas, 128-64-32 neuronas)  
- **Función de activación**: (ej. ReLU)  
- **Optimizador**: (ej. Adam con LR=0.001)  
- **Épocas**: (ej. 100)  
- **Batch size**: (ej. 64)  

**Métricas de Rendimiento**  
- **Accuracy / F1-Score / etc.**: (completar)  
- **Pérdida (loss)**: (completar)  
- **Observaciones**: (ej. sobreajuste, subajuste, tiempo de entrenamiento...)  

**Notas Adicionales**  
- (ej. “Este modelo sirvió como base inicial, pero con un dataset limitado”; “no se integró con socket aún”, etc.)

---

## models_2024-12-14

**Descripción general**  
- **Fecha de creación**: 14 de diciembre de 2024  
- **Tipo de modelo**:  
- **Objetivo**:  
- **Conjunto de datos**:  

**Arquitectura / Hiperparámetros**  
- **Capas**:  
- **Función de activación**:  
- **Optimizador**:  
- **Épocas**:  
- **Batch size**:  

**Métricas de Rendimiento**  
- **Accuracy / F1-Score / etc.**:  
- **Pérdida (loss)**:  
- **Observaciones**:  

**Notas Adicionales**  
- (ej. “Se mejoró el dataset incluyendo variables X e Y”; “se corrigió la codificación One-Hot en las columnas de movimientos”)

---

## models_2025-02-12

**Descripción general**  
- **Fecha de creación**: 12 de febrero de 2025  
- **Tipo de modelo**:  
- **Objetivo**:  
- **Conjunto de datos**:  

**Arquitectura / Hiperparámetros**  
- (Completar según corresponda)  

**Métricas de Rendimiento**  
- (Indicar métricas relevantes)  

**Notas Adicionales**  
- (Incluir si se usaron técnicas de explicabilidad: SHAP, LIME, Feature Importance, etc.)

---

## models_2025-02-14

**Descripción general**  
- **Fecha de creación**: 14 de febrero de 2025  
- **Tipo de modelo**:  
- **Objetivo**:  
- **Conjunto de datos**:  

**Arquitectura / Hiperparámetros**  
- (Completar)  

**Métricas de Rendimiento**  
- (Completar)  

**Notas Adicionales**  
- (Señalar mejoras o cambios relevantes)

---

## models_2025-03-05

**Descripción general**  
- **Fecha de creación**: 5 de marzo de 2025  
- **Tipo de modelo**:  
- **Objetivo**:  
- **Conjunto de datos**:  

**Arquitectura / Hiperparámetros**  
- (Completar)  

**Métricas de Rendimiento**  
- (Completar)  

**Notas Adicionales**  
- (Ej. “Se integró con socket Java-Python y se probó en partidas reales”)

---

## models_2025-03-12

**Descripción general**  
- **Fecha de creación**: 12 de marzo de 2025  
- **Tipo de modelo**:  
- **Objetivo**:  
- **Conjunto de datos**:  

**Arquitectura / Hiperparámetros**  
- (Completar)  

**Métricas de Rendimiento**  
- (Completar)  

**Notas Adicionales**  
- (Poner si se usaron o no mapas de calor de explicabilidad, etc.)

---

## models_2025-03-16

**Descripción general**  
- **Fecha de creación**: 16 de marzo de 2025  
- **Tipo de modelo**:  
- **Objetivo**:  
- **Conjunto de datos**:  

**Arquitectura / Hiperparámetros**  
- (Completar)  

**Métricas de Rendimiento**  
- (Completar)  

**Notas Adicionales**  
- (Evolución respecto a versiones anteriores, problemas detectados, etc.)

---

## models_2025-03-17

**Descripción general**  
- **Fecha de creación**: 17 de marzo de 2025  
- **Tipo de modelo**:  
- **Objetivo**:  
- **Conjunto de datos**:  

**Arquitectura / Hiperparámetros**  
- (Completar)  

**Métricas de Rendimiento**  
- (Completar)  

**Notas Adicionales**  
- (Comentarios finales)

---

## Recomendaciones para Mantener la Documentación

1. **Actualizar al entrenar un nuevo modelo**  
   - Cada vez que se entrene una red neuronal diferente o se modifique un modelo existente, añadir una nueva sección en este archivo con la fecha y detalles relevantes.

2. **Incluir referencias cruzadas**  
   - Si un modelo sirve de base para otro, especificar la relación. Por ejemplo: “El modelo `models_2025-03-12` se basa en la versión del `models_2025-02-14` con cambios en el optimizador...”.

3. **Registrar observaciones de pruebas**  
   - Si se integra el modelo en el juego, anotar su comportamiento en partidas simuladas: puntuación media, porcentaje de victorias, etc.

4. **Conservar métricas y configuraciones**  
   - Si se cambia la arquitectura (por ejemplo, se añade una nueva capa, se cambian hiperparámetros, o se varía el dataset), dejar constancia para conocer el “por qué” de la mejora o deterioro en el rendimiento.

5. **Versión del código**  
   - Es recomendable anotar la rama de Git o el commit usado para entrenar cada modelo, de modo que sea más fácil replicar experimentos.

---

> **Fin de la documentación**  
>  
> Este archivo .md quedará como referencia central para consultar la información de cada modelo entrenado. Se recomienda mantenerlo siempre actualizado y versionado en el repositorio Git del proyecto.
