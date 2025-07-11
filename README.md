# Sistema de Código Abierto para la generación de modelos 3D de racimos de uvas a partir de un video MP4
## Descripción y funcionamiento
Este es un Sistema innovador basado en Arquitectura de Microservicios para la generación automatizada de nubes densas de puntos 3D a partir de un video MP4. El video es procesado por una Red Neuronal Convolucional para detectar cada una de las uvas del racimo, posteriormente se realiza un Seguimiento de Objetos Múltiples (MOT) para reconstruir tridimensionalmente mediante Structure from Motrion (SfM). 

Su funcionamiento consiste en enviar una lista de videos con las siguientes características. 

![VID_20230321_142835 (1)](https://github.com/user-attachments/assets/8bf1e53b-22c9-4cb4-8b75-a0464ce96335)

Con los siguientes parámetros para su procesamiento 
```
    num_processes: int
    generar_video_qr: bool
    factor_lentitud: float
    baya_thresh: float
    qr_thresh: float
    cant_nubes: int
    calib_file: str 
    qr_dist: float
    dists_list: list[int]
    num_points: int
    umbral_triangulacion: float
    max_workers_triangulacion: int
```
Un ejemplo de estos parámetros es
```
{
  "num_processes": 4,
  "generar_video_qr": true,
  "factor_lentitud": 0.5,
  "baya_thresh": 105.0,
  "qr_thresh": 120,
  "cant_nubes": 3,
  "calib_file": "MotorolaG200_Javo_Horizontal.yaml",
  "qr_dist": 2.1,
  "dists_list": [10, 21, 5],
  "num_points": 100,
  "umbral_triangulacion": 0.08,
  "max_workers_triangulacion": 4
}
```
La petición HTTP tendría la siguiente forma, siendo `@body` el json con parámetros
```
curl -X POST \
     -F "videos=@VID_20230321_142835.mp4" \
     -F "videos=@VID_20230322_173233.mp4" \
     -F "videos=@VID_20230322_173621.mp4" \
     -F "params=@body.json;type=application/json" \
     http://127.0.0.1:8000/upload_videos
```
Antes de hacer la petición es necesario levantar el sistema de contenedores, esto se hace estando situado en la carpeta raíz y escribiendo el siguiente comando
```
docker-compose down && docker-compose up --build
```
Una vez terminado el procesamiento, los resultados son parecidos a los siguientes (depende del video MP4)

![image](https://github.com/user-attachments/assets/4a8412bc-6f11-4033-af98-a03c9037a45d)

## Hacer antes de utilizar

Este sistema funciona con CUDA, únicamente para los modelos NVIDIA 10XX por las versiones de Pytorch y CUDA que utiliza el Detector de Bayas.

Es fundamental descargar los pesos del Detector de Bayas en el siguiente link: https://drive.google.com/file/d/1XsvtpexSGl8kwA1xgeCUBpyOIuSvXy8V/view?usp=drive_link. Se debe poner este archivo en la carpeta `workers/detectorBayas`. Es recomendable utilizar el siguiente comando en consola 

```
gdown https://drive.google.com/file/d/1XsvtpexSGl8kwA1xgeCUBpyOIuSvXy8V/view?usp=drive_link
```

Además, se debe dar permisos de ejecución a los scripts .sh 

```
cd workers/detectorBayas
chmod +x entrypoint.sh install_pytorch041.sh compiler.sh
```

```
cd workers/nubesBayas
chmod +x entrypoint.sh
```

El sistema debe tener instalado `nvidia-smi` y NVIDIA Container Toolkit para que los contenedores Docker puedan utilizar la GPU del host: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Otras salidas de utilidad que brinda el sistema
Las otras salidas que el sistema brinda para mejorar la precisión del mismo son
* Detección de códigos QR frame a frame
* Detección de bayas frame a frame
* Triangulación de bayas frame a frame (por cada triangulación realizada) 
* Video generado con detección de códigos QR a lo largo del video
* Gráfico de distribución de detección de códigos QR y gráfico temporal de detección de códigos QR


