# Modernización de `detectorBayas`: cambios implementados

## 1. Objetivo

El worker original `workers/detectorBayas` estaba fuertemente acoplado a una
pila de software antigua:

- Ubuntu 18.04;
- CUDA 9.2;
- Python 3.6;
- PyTorch 0.4.1;
- extensiones CUDA y Cython compiladas al iniciar;
- una copia amplia de CircleNet con entrenamiento, varias arquitecturas y
  utilidades que no participan de la inferencia del pipeline.

Esa combinación fue construida antes de las GPU NVIDIA Ampere y no es una base
adecuada para una RTX 3050. Además, hacía difícil distinguir qué código era
realmente necesario para detectar bayas.

La solución implementada fue crear un worker nuevo e independiente:

```text
workers/detectorBayasModern/
```

El worker viejo **no fue eliminado ni modificado**. Permanece disponible como
referencia y rollback. Cuando este documento habla de elementos "eliminados",
significa que esos elementos no se copiaron ni forman parte del runtime del
worker moderno.

## 2. Resumen antes y después

| Aspecto | Worker anterior | Worker moderno |
| --- | --- | --- |
| Base GPU | CUDA 9.2 | CUDA 11.8 |
| Framework | PyTorch 0.4.1 | PyTorch 2.6.0 |
| cuDNN | Deshabilitado y parcheado | cuDNN 9 habilitado |
| Python | 3.6 | Python 3.11 de la imagen oficial |
| Arquitecturas incluidas | Hourglass, DLA, ResNet y DCNv2 | Sólo Hourglass de dos stacks |
| Compilación al arrancar | DCNv2 y NMS Cython | Ninguna |
| Carga del modelo | Una vez por request | Una vez al iniciar FastAPI |
| Dependencias directas `requirements.txt` | 16, además del entorno Conda | 5 |
| Checkpoint incompatible | Se completaban pesos silenciosamente | El servicio falla con un error explícito |
| Escritura JSON | Directa | Temporal y reemplazo atómico |
| Estado del servicio | Logs impresos manualmente | Endpoint `/health` |
| Pruebas | Sin suite automatizada | Unitarias, API, video y paridad de arquitectura |

La imagen CUDA moderna sigue siendo grande porque incluye el runtime oficial
de CUDA y cuDNN. La mejora principal no es únicamente el tamaño de la imagen,
sino la eliminación de compilaciones frágiles, dependencias muertas y código
que no pertenece al flujo de inferencia.

## 3. Qué se mantuvo exactamente

La modernización se diseñó como un reemplazo transparente para el resto del
pipeline.

### 3.1 Contrato HTTP

Se conserva:

```http
POST /detector_task
```

Request:

```json
{
  "input_folder": "/shared/prueba",
  "output_folder": "/shared/prueba",
  "video_name": "prueba"
}
```

Response:

```json
{
  "message": "Detección completada",
  "video_name": "prueba",
  "output_folder": "/shared/prueba"
}
```

No fue necesario modificar las tareas HTTP del orquestador ni el tracker.

### 3.2 JSON de detecciones

Se conserva la ruta:

```text
<output_folder>/<video_name>.json
```

Y la estructura:

```json
{
  "0": {
    "0": [123.4, 456.7, 12.8]
  }
}
```

Se mantienen las coordenadas `[x, y, radio]`, los IDs reiniciados por frame y
el filtro heredado que descarta centros con `x <= 0` o `y <= 0`. No se agregan
scores, clases ni datos de oclusión porque eso cambiaría la entrada de
`trackerBayas`.

### 3.3 `detector_frames/`

Se mantiene la exportación solicitada:

```text
<output_folder>/detector_frames/00000.jpg
<output_folder>/detector_frames/00001.jpg
...
```

Cada imagen se guarda antes de ejecutar la inferencia, con cinco dígitos y
numeración desde cero, como en el worker anterior.

### 3.4 Checkpoint

Se conserva el mismo archivo:

```text
2022.11.30_grapes_mix_iou.pth
```

No se convirtió, reentrenó ni alteró. La jerarquía interna de módulos de la red
nueva conserva las claves y shapes del `state_dict` original.

## 4. Nueva red mantenida por el proyecto

La auditoría determinó que el endpoint anterior seleccionaba siempre:

```text
arch = hourglass
num_stacks = 2
```

Aunque el repositorio incluía DLA, ResNet y DCNv2, esas arquitecturas no eran
usadas por el endpoint. Por eso se creó `BerryHourglassNet` con únicamente:

- preprocesamiento convolucional;
- bloques residuales;
- módulos Hourglass recursivos;
- conexiones entre los dos stacks;
- cabezas `hm`, `cl`, `reg` y `occ`.

La configuración compatible es:

```text
input_height = 1024
input_width = 576
down_ratio = 4
num_stacks = 2
confidence_threshold = 0.4
max_detections = 1000
```

Cabezas:

```python
{
    "hm": 1,
    "cl": 1,
    "reg": 2,
    "occ": 1,
}
```

Normalización BGR heredada:

```text
mean = [0.408, 0.447, 0.470]
std  = [0.289, 0.274, 0.278]
```

### Por qué mejora

- La arquitectura efectiva queda explícita y fácil de localizar.
- No existe una factory genérica con alternativas que producción nunca usa.
- El código puede evolucionar dentro del proyecto sin depender de la
  estructura completa de CircleNet.
- La compatibilidad con los pesos existentes evita reentrenar.
- Las licencias y atribuciones de CircleNet/CornerNet se conservan en
  `LICENSE` y `THIRD_PARTY_NOTICES.md`.

## 5. Elementos excluidos del worker moderno

### 5.1 DCNv2 y fuentes CUDA locales

No se incluyeron:

- `models/networks/DCNv2`;
- fuentes `.cu`, `.c` y `.h`;
- scripts `build.py`, `make.sh` y variantes double;
- compilación de `_cdpooling.so`.

### Motivo

La red seleccionada es Hourglass y no utiliza deformable convolutions. Sin
embargo, el worker anterior intentaba compilar DCNv2 igualmente al arrancar.
Esa extensión fue escrita para APIs antiguas de PyTorch/CUDA y era una fuente
principal de incompatibilidad.

### Mejora

- El contenedor no necesita `nvcc`, GCC o G++ para iniciar.
- Desaparecen errores por ABI, compute capability o headers de PyTorch.
- El arranque es determinista y más rápido.
- No se generan binarios distintos según la GPU donde inicia el contenedor.

### 5.2 NMS Cython

No se incluyeron `external/nms.pyx`, su `Makefile` ni `setup.py`.

### Motivo

El endpoint no activa la opción de Soft-NMS. Importar y compilar esa extensión
era trabajo sin efecto sobre las detecciones producidas.

### Mejora

- Se elimina Cython del runtime.
- Se evita otra extensión nativa dependiente de NumPy y del compilador.

### 5.3 Arquitecturas no utilizadas

No se incluyeron:

- DLA y DLA-DCN;
- ResNet y ResNet-DCN;
- MSRA ResNet;
- factories genéricas de modelos.

### Motivo

El endpoint fijaba `hourglass`; no existía configuración del request o del
Compose que eligiera esas alternativas.

### Mejora

- Menor superficie de mantenimiento.
- Menos imports y dependencias transitivas.
- La arquitectura usada no queda oculta detrás de opciones heredadas.

### 5.4 Código de entrenamiento y evaluación

No se incluyeron:

- losses;
- optimizadores y resume de training;
- DataParallel y scatter/gather;
- integración COCOAPI;
- logger de experimentos;
- debugger y visualizaciones;
- utilidades 3D;
- opciones CLI de datasets, épocas y augmentations.

### Motivo

`detectorBayasModern` es un servicio de inferencia. No existe un endpoint ni
un flujo del pipeline que entrene el modelo.

### Mejora

- Se evita mezclar serving y experimentación.
- Las dependencias de producción reflejan lo que realmente se ejecuta.
- Es más sencillo auditar qué código accede a GPU, video y filesystem.

### 5.5 Entorno Conda e instaladores heredados

No se incluyeron:

- `environment.yml` con Python 3.6;
- `install_pytorch041.sh`;
- instalación manual de CUDA 9.2;
- descarga del wheel PyTorch 0.4.1;
- `compiler.sh` y el entrypoint basado en Conda;
- actualización completa de Conda durante el build.

### Motivo

El worker ahora parte de una imagen oficial de PyTorch que ya contiene una
combinación coherente de framework, CUDA y cuDNN.

### Mejora

- Build reproducible con versiones fijadas.
- Menos descargas desde URLs históricas o repositorios retirados.
- El host no necesita instalar CUDA: sólo driver NVIDIA, Docker y NVIDIA
  Container Toolkit.

### 5.6 Parches internos de PyTorch y cuDNN

Se eliminó conceptualmente del worker moderno el `sed` que modificaba archivos
de `torch.nn.functional` y la ejecución de:

```python
torch.backends.cudnn.enabled = False
```

### Motivo

Modificar una librería instalada hace que el entorno sea difícil de reproducir
y ocultaba problemas de compatibilidad en lugar de resolverlos.

### Mejora

- cuDNN 9 permanece habilitado.
- Se utilizan kernels mantenidos y optimizados para GPUs modernas.
- La instalación de PyTorch queda intacta y verificable.

## 6. Actualización de PyTorch y CUDA

La imagen nueva usa:

```text
pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime
```

CUDA 11.8 cubre las generaciones declaradas para este proyecto:

- GTX serie 10;
- RTX serie 20;
- RTX serie 30, incluida RTX 3050;
- RTX serie 40.

El driver mínimo documentado para CUDA 11.8 en Linux es 520.61.05. Se
recomienda utilizar un driver NVIDIA reciente soportado por la distribución.

### Por qué no se instaló simplemente una CUDA nueva en el Dockerfile viejo

PyTorch 0.4.1, Python 3.6, cuDNN, NumPy y las extensiones nativas formaban una
unidad de compatibilidad. Cambiar sólo CUDA habría dejado extensiones y wheels
construidos contra APIs y ABIs antiguas. Por eso se reemplazó la pila completa
por una combinación oficial.

## 7. Carga estricta del checkpoint

El loader nuevo:

1. Verifica que el archivo exista.
2. Carga inicialmente en CPU.
3. Usa `weights_only=True`.
4. Acepta checkpoints con prefijo `module.` de DataParallel.
5. Compara claves faltantes y sobrantes.
6. Compara la shape de cada tensor.
7. Usa `load_state_dict(..., strict=True)`.
8. Mueve el modelo al dispositivo sólo después de validar.

El loader anterior sustituía silenciosamente parámetros incompatibles por los
valores recién inicializados del modelo. Eso podía dejar el servicio activo
con una parte de la red usando pesos aleatorios.

### Mejora

- Un checkpoint incorrecto impide que el servicio quede healthy.
- El error identifica claves o shapes problemáticas.
- Se evita producir detecciones aparentemente válidas con pesos parciales.
- La carga restringida reduce el riesgo de deserializar objetos arbitrarios.

## 8. Modelo cargado una sola vez

Antes, `/detector_task` construía la red y cargaba aproximadamente 483 MB de
checkpoint en cada request.

Ahora FastAPI utiliza su lifespan para:

1. Leer configuración.
2. Construir `BerryHourglassNet`.
3. Cargar y validar el checkpoint.
4. Mover la red a GPU.
5. Mantener la instancia mientras viva el proceso.

### Mejora

- Menor latencia al comenzar cada video.
- Menor cantidad de lecturas de disco.
- Se evita reservar y liberar repetidamente VRAM.
- Los errores de modelo aparecen al iniciar, no después de aceptar un trabajo.

Se usa un único worker Uvicorn y un lock de procesamiento. Esto evita cargar
varias copias del modelo en la misma GPU o mezclar dos requests que escriban
artefactos simultáneamente.

## 9. Preprocesado y decoder

El preprocesado conserva:

- frames BGR de OpenCV;
- transformación afín;
- resolución `1024 × 576`;
- normalización original;
- tensor NCHW con batch uno;
- retorno a coordenadas del frame original.

El decoder conserva el algoritmo de CircleNet CDIou usado por el worker:

- sigmoid de `hm`;
- top-k;
- umbral inicial;
- selección por oclusión;
- offsets `reg`;
- radio `cl`;
- salida `[x, y, radio]`.

La selección inicial fue vectorizada. El código anterior ejecutaba `.item()`
por cada candidato, forzando hasta 1000 sincronizaciones CPU/GPU por frame. La
versión nueva copia los candidatos directamente con operaciones tensoriales.

La comparación del umbral promueve el tensor a `float64` antes de evaluar. Ese
detalle reproduce la semántica anterior de `float32.item() > float_de_Python`,
incluido el caso límite de un score representado alrededor de `0.4`.

### Mejora

- Menos transferencias y esperas entre CPU y GPU.
- Se conserva el comportamiento observado del umbral.
- Se reemplaza `np.float`, eliminado en NumPy moderno.

## 10. Procesamiento de video y persistencia

El módulo de video ahora:

- verifica que el MP4 exista;
- valida que OpenCV pueda abrirlo;
- libera siempre `VideoCapture` mediante `finally`;
- comprueba el resultado de cada `cv2.imwrite`;
- mantiene `detector_frames/`;
- transforma valores NumPy a floats serializables;
- escribe el JSON mediante un archivo temporal;
- llama `fsync` y luego `os.replace`.

### Por qué se usa escritura atómica

El tracker consume el JSON después del detector. Si el proceso era interrumpido
durante `json.dump`, podía quedar un archivo truncado que existía pero no era
JSON válido. Ahora el path final aparece únicamente cuando la escritura se
completó.

Los JPEG se siguen publicando frame por frame para conservar el comportamiento
solicitado. Si una ejecución reutiliza un directorio con frames viejos, el
worker no elimina automáticamente archivos sobrantes; esto evita una operación
destructiva y mantiene el comportamiento legado.

## 11. API y healthcheck

Se agregó:

```http
GET /health
```

Informa:

- estado ready;
- modelo cargado;
- dispositivo;
- nombre de GPU;
- versión de PyTorch;
- versión CUDA del runtime.

También se agregaron errores HTTP diferenciados:

- 404 para video inexistente;
- 422 para video ilegible o sin frames;
- 500 para fallos de escritura o inferencia.

### Mejora

- Docker Compose sabe cuándo el modelo terminó de cargar.
- El orquestador no inicia basándose solamente en que existe un proceso.
- Los fallos de entrada/salida son diagnosticables sin revisar un traceback
  genérico.

## 12. Cambios en Docker Compose

`detector-service` conserva nombre, puerto y endpoint, pero ahora construye:

```yaml
build: ./workers/detectorBayasModern
```

Cambios adicionales:

- `gpus: all` solicita una GPU mediante Compose.
- Se mantiene `${SHARED_VOL}:/shared`.
- El modelo se monta en `/models:ro`.
- Se eliminó el bind mount del código anterior sobre `/app`.
- Se fijaron `MODEL_PATH`, `DEVICE`, `CONFIDENCE_THRESHOLD` y
  `MAX_DETECTIONS`.
- Se agregó healthcheck.
- `orquestador-fastapi` espera `service_healthy`.

### Por qué se eliminó el bind mount del código

Montar el directorio fuente sobre `/app` ocultaba los archivos copiados durante
el build. La imagen construida podía ser correcta, pero al ejecutarla usaba una
versión diferente del código presente en el host.

### Mejora

- La imagen probada es la misma que se ejecuta.
- El checkpoint queda separado del código y en sólo lectura.
- El rollback consiste en restaurar el `build` anterior.

## 13. Dependencias de producción

El nuevo `requirements.txt` contiene únicamente:

- FastAPI;
- NumPy;
- OpenCV headless;
- Pydantic;
- Uvicorn.

PyTorch, CUDA y cuDNN provienen de la imagen base oficial.

Dependencias de tests como `pytest` y `httpx` están separadas en
`requirements-dev.txt` y en un target Docker llamado `test`. No se instalan en
la imagen final de producción.

## 14. Dockerfile multi-stage

El Dockerfile define:

- `runtime`: aplicación y dependencias productivas;
- `test`: agrega pytest y copia la suite;
- `production`: imagen final basada en runtime.

### Mejora

- La misma base se usa para pruebas y producción.
- Las herramientas de test no aumentan el entorno productivo.
- No hay compilación al iniciar el contenedor.

## 15. Pruebas agregadas

Se agregaron pruebas para:

- carga de checkpoint con prefijo DataParallel;
- rechazo de checkpoints incompletos;
- decoder y layout `[x, y, radio, oclusión, score, clase]`;
- equivalencia de la selección vectorizada con el loop legado;
- transformaciones afines;
- contrato del JSON;
- nombres `00000.jpg`, `00001.jpg`, etc.;
- respuesta exacta de `/detector_task`;
- `/health`;
- HTTP 404 para videos ausentes;
- claves y shapes del modelo moderno contra `large_hourglass.py`;
- logits exactos de ambas implementaciones con pesos iniciales equivalentes.

Resultados obtenidos durante la implementación:

```text
10 passed, 1 skipped
```

La prueba pesada de arquitectura se ejecutó por separado:

```text
1 passed
```

También se validaron:

- `python -m compileall`;
- `docker compose config`;
- `git diff --check`;
- build de la imagen de producción;
- forward pass completo de ambos stacks.

## 16. Archivos principales

| Archivo | Responsabilidad |
| --- | --- |
| `berry_detector/architecture.py` | `BerryHourglassNet` mínima |
| `berry_detector/checkpoint.py` | Carga estricta de pesos |
| `berry_detector/config.py` | Configuración y variables de entorno |
| `berry_detector/geometry.py` | Transformaciones afines |
| `berry_detector/decoding.py` | Top-k y decodificación de círculos |
| `berry_detector/detector.py` | Preprocesado, red y postprocesado |
| `berry_detector/video.py` | MP4, JPEG y JSON atómico |
| `api/main.py` | Lifespan, endpoint y healthcheck |
| `Dockerfile` | Runtime CUDA y targets test/production |
| `tests/` | Suite automática y paridad legada |

## 17. Qué no se cambió intencionalmente

- No se reentrenó la red.
- No se alteró el checkpoint.
- No se cambió el umbral por defecto.
- No se modificó el formato JSON.
- No se agregaron scores al tracker.
- No se cambió la resolución de entrada.
- No se eliminaron `detector_frames`.
- No se borró `workers/detectorBayas`.
- No se agregó soporte declarado para RTX 50.
- No se incluyó entrenamiento en el worker moderno.

Estas restricciones reducen el riesgo de mezclar una migración de plataforma
con un cambio de calidad del modelo.

## 18. Limitaciones y validaciones pendientes

Durante la implementación no había una GPU NVIDIA accesible mediante
`nvidia-smi`, por lo que queda pendiente ejecutar un video real en RTX 3050 y
registrar:

- nombre de GPU y driver;
- VRAM máxima;
- tiempo total y tiempo por frame;
- cantidad de detecciones por frame;
- comparación de JSON contra el worker anterior;
- ejecución posterior de `trackerBayas`.

La estructura y el forward de la red moderna sí se compararon contra la
Hourglass legada. La prueba end-to-end con checkpoint y video reales es el paso
que confirma también la equivalencia de preprocesado, decoder y versiones de
PyTorch sobre hardware NVIDIA.

La inferencia con `DEVICE=cpu` está permitida para diagnóstico, pero la red es
grande y puede resultar demasiado lenta para videos completos.

## 19. Cómo probar

### Tests automáticos

```bash
docker build \
  --target test \
  -t detector-bayas-modern:test \
  workers/detectorBayasModern

docker run --rm detector-bayas-modern:test
```

### Inferencia real

Colocar el checkpoint en:

```text
workers/detectorBayasModern/models/2022.11.30_grapes_mix_iou.pth
```

Preparar el video:

```bash
mkdir -p shared/prueba
cp /ruta/al/video.mp4 shared/prueba/prueba.mp4
```

Levantar el worker:

```bash
export SHARED_VOL="$PWD/shared"
docker compose up --build detector-service
```

Ejecutar:

```bash
curl -X POST http://localhost:8001/detector_task \
  -H 'Content-Type: application/json' \
  -d '{
    "input_folder": "/shared/prueba",
    "output_folder": "/shared/prueba",
    "video_name": "prueba"
  }'
```

Verificar:

```text
shared/prueba/prueba.json
shared/prueba/detector_frames/00000.jpg
shared/prueba/detector_frames/00001.jpg
...
```

## 20. Rollback

El worker anterior permanece intacto. Para volver a utilizarlo, restaurar en
`docker-compose.yml`:

```yaml
detector-service:
  build: ./workers/detectorBayas
```

También se debe restaurar el montaje de código y cualquier configuración que
necesite el Dockerfile legado. El formato del request y de las salidas no
cambia, por lo que el orquestador y el tracker no requieren una migración de
datos para hacer rollback.
