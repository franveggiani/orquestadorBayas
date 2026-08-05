# detectorBayasModern

Worker de inferencia compatible con `detectorBayas`, reducido a la Hourglass
de dos stacks que utiliza el checkpoint de producción. No contiene código de
entrenamiento, DCNv2, COCOAPI ni extensiones CUDA compiladas localmente.

La explicación completa de la migración, las exclusiones y sus beneficios está
en [CAMBIOS_IMPLEMENTADOS.md](CAMBIOS_IMPLEMENTADOS.md).

## Compatibilidad GPU

La imagen usa PyTorch 2.6.0, CUDA 11.8 y cuDNN 9. El host necesita:

- una GPU NVIDIA GTX serie 10, RTX serie 20, 30 o 40;
- driver NVIDIA 520.61.05 o superior;
- Docker con NVIDIA Container Toolkit configurado.

CUDA y PyTorch no se instalan en el host: vienen dentro del contenedor.

Verifique el runtime antes de levantar el pipeline:

```bash
nvidia-smi
docker run --rm --gpus all pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Checkpoint

El checkpoint no se versiona. Cópielo en el directorio configurado:

```text
workers/detectorBayasModern/models/2022.11.30_grapes_mix_iou.pth
```

Para usar otra ubicación en el host:

```bash
export DETECTOR_MODELS_DIR=/ruta/absoluta/a/modelos
```

Dentro del contenedor se carga desde:

```text
/models/2022.11.30_grapes_mix_iou.pth
```

El servicio falla al arrancar si el archivo falta o su `state_dict` no es
exactamente compatible.

## Ejecución

Desde la raíz del repositorio:

```bash
export SHARED_VOL="$PWD/shared"
docker compose up --build detector-service
```

Estado del modelo:

```bash
curl http://localhost:8001/health
```

Petición compatible con el worker anterior:

```bash
curl -X POST http://localhost:8001/detector_task \
  -H 'Content-Type: application/json' \
  -d '{
    "input_folder": "/shared/VID_20230322_173233",
    "output_folder": "/shared/VID_20230322_173233",
    "video_name": "VID_20230322_173233"
  }'
```

El video debe existir como
`<input_folder>/<video_name>.mp4`. El worker genera:

```text
<output_folder>/<video_name>.json
<output_folder>/detector_frames/00000.jpg
<output_folder>/detector_frames/00001.jpg
...
```

## Configuración

| Variable | Valor predeterminado |
| --- | --- |
| `MODEL_PATH` | `/models/2022.11.30_grapes_mix_iou.pth` |
| `DEVICE` | `cuda` |
| `CONFIDENCE_THRESHOLD` | `0.4` |
| `MAX_DETECTIONS` | `1000` |

`DEVICE=cpu` existe para pruebas y diagnóstico; producción falla de forma
intencional si solicita CUDA y la GPU no está disponible.

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest -q
```

La prueba pesada de compatibilidad estructural con el modelo legado es
optativa:

```bash
RUN_MODEL_TESTS=1 pytest -q -m slow
```

Para rollback, cambie temporalmente el `build` de `detector-service` en el
Compose principal de vuelta a `./workers/detectorBayas`.
