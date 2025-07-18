from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from celery import chain, group
from .tasks import detector_http_task, qr_detector_http_task, tracker_http_task, nubes_http_task
from typing import List, Dict, Any
import os
import shutil
import json

app = FastAPI()
SHARED_PATH = "/shared"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_videos")
async def upload_videos(
    videos: List[UploadFile] = File(...),
    num_processes: int = Form(...),
    factor_lentitud: float = Form(...),
    bayas_thresh: float = Form(...),
    qr_thresh: float = Form(...),
    cant_nubes: int = Form(...),
    calib_file: str = Form(...),
    qr_dist: float = Form(...),
    dists_list: str = Form(...),
    num_points: int = Form(...),
    umbral_triangulacion: float = Form(...),
    max_workers_triangulacion: int = Form(...),
    generar_video_qr: bool = Form(False),
):
    
    """Endpoint que recibe videos y lanza pipelines en paralelo."""

    all_task = []
    all_task_ids = [] 
    uploaded_videos = []
    
    # Convertir dists_list que vino como JSON string a lista Python
    dists_list = json.loads(dists_list)
    dists_list = [int(x) for x in dists_list]


    for video in videos:
        
        # Crear carpeta por cada video en SHARED_PATH
        video_name = str(video.filename).replace('.mp4', '')
        video_folder = os.path.join(SHARED_PATH, video_name)
        os.makedirs(video_folder, exist_ok=True)
        video_path = os.path.join(video_folder, video_name + '.mp4')
        
        try:
            with open(video_path, 'wb') as f:
                shutil.copyfileobj(video.file, f)
                print(f"Video guardado en: {video_path}")
        except Exception as e:
            return {"error": f"Error al guardar el video: {str(e)}"}
        
        uploaded_videos.append(video_folder)

        # Crear un grupo de pipelines (cada pipeline es una cadena de tareas)
        pipeline = chain(
            group (detector_http_task.s(video_folder, video_folder, video_name), 
                   qr_detector_http_task.s(video_folder, video_folder, video_name, num_processes, generar_video_qr, factor_lentitud)
                   ),
            tracker_http_task.s(video_folder, video_folder, video_name, radius=10, draw_circles=True, draw_tracking=True),
            nubes_http_task.s(video_folder, 
                              video_folder, 
                              video_name, 
                              bayas_thresh, 
                              qr_thresh,
                              cant_nubes, 
                              calib_file, 
                              qr_dist, 
                              dists_list, 
                              num_points, 
                              umbral_triangulacion, 
                              max_workers_triangulacion)
        )

        all_task.append(pipeline)
    
    # Ejecutar el grupo en paralelo
    pipeline = group(all_task)
    group_result = pipeline.apply_async()
    
    # Acá guardamos los ids por video y por tarea
    detailed_task_ids = []

    for res in group_result.results:
        video_task_ids = []

        # res es un Chain
        if hasattr(res, "children"):
            first_stage = res.children[0]  # El group (detector, qr_detector)
            second_stage = res.children[1]  # tracker
            third_stage = res.children[2]   # nubes
            
            # Los ids del group (detector, qr_detector)
            detector_qr = first_stage.results if hasattr(first_stage, 'results') else []
            detector_qr_ids = [r.id for r in detector_qr]

            # Los otros ids
            tracker_id = second_stage.id if hasattr(second_stage, "id") else None
            nubes_id = third_stage.id if hasattr(third_stage, "id") else None

            video_task_ids.append({
                "detector": detector_qr_ids[0] if len(detector_qr_ids) > 0 else None,
                "qr_detector": detector_qr_ids[1] if len(detector_qr_ids) > 1 else None,
                "tracker": tracker_id,
                "nubes": nubes_id
            })

        detailed_task_ids.append(video_task_ids)

    return {
        "task_ids": detailed_task_ids,
        "uploaded_videos": uploaded_videos
    }

@app.post("/prueba")
async def prueba(
    videos: List[UploadFile] = File(...),
    num_processes: int = Form(...),
    factor_lentitud: float = Form(...),
    bayas_thresh: float = Form(...),
    qr_thresh: float = Form(...),
    cant_nubes: int = Form(...),
    calib_file: str = Form(...),
    qr_dist: float = Form(...),
    dists_list: str = Form(...),
    num_points: int = Form(...),
    umbral_triangulacion: float = Form(...),
    max_workers_triangulacion: int = Form(...),
    generar_video_qr: bool = Form(False),
):
    # Convertir dists_list que vino como JSON string a lista Python
    dists_list_parsed = json.loads(dists_list)

    # Mostrar nombres de los archivos recibidos
    videos_info = [{"filename": video.filename, "content_type": video.content_type} for video in videos]
    
    print(videos, num_processes, factor_lentitud, bayas_thresh, qr_thresh, cant_nubes, calib_file, qr_dist, dists_list_parsed, num_points, umbral_triangulacion, max_workers_triangulacion, generar_video_qr)

    # Devolver TODO para verificar que lo recibís bien
    return JSONResponse(content={
        "videos": videos_info,
        "num_processes": num_processes,
        "factor_lentitud": factor_lentitud,
        "bayas_thresh": bayas_thresh,
        "qr_thresh": qr_thresh,
        "cant_nubes": cant_nubes,
        "calib_file": calib_file,
        "qr_dist": qr_dist,
        "dists_list": dists_list_parsed,
        "num_points": num_points,
        "umbral_triangulacion": umbral_triangulacion,
        "max_workers_triangulacion": max_workers_triangulacion,
        "generar_video_qr": generar_video_qr,
    })
