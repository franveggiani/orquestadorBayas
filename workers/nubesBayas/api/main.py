from fastapi import FastAPI, HTTPException, Path
import subprocess
from .schemas import nubesRequest
from .bundles import read_bind_rows, procesar_datos, generate_image_table, write_in_folder
from .functions import video_to_frame, get_best_triangulacion, csv_to_ply_with_sphere, get_best_triangulacion2
import sys
import os


print("Python ejecutado:", sys.executable)
print("Directorio actual:", os.getcwd())
print("Archivos en el directorio actual:", os.listdir("."))

app = FastAPI()

@app.post('/nubes_task')
async def nubes(request: nubesRequest):
    
    try:    
        input_folder = request.input_folder
        output_folder = request.output_folder
        video_name = request.video_name
        baya_threshold = request.baya_threshold
        qr_threshold = request.qr_threshold
        qr_dist = request.qr_dist
        dists_list = request.dists_list
        cantidad_nubes = request.cantidad_nubes
        calib_path = os.path.join('/app/calib', request.calib_file)
        bundles_path = os.path.join(output_folder, 'bundles.csv')
        umbral_triangulacion = request.umbral_triangulacion
        max_workers_triangulacion = request.max_workers_triangulacion
        
        print(calib_path)
        
        # Generamos bundles.csv
        tracker_detection_path = os.path.join(input_folder, 'tracker_detections.csv')
        qr_detection_path = os.path.join(input_folder, 'qr_detections.csv')
        
        path_list = [tracker_detection_path, qr_detection_path]

        df = read_bind_rows(path_list)
        df = procesar_datos(df, 'baya')
        df = generate_image_table(df, baya_thresh=baya_threshold, qr_thresh=qr_threshold)
        write_in_folder(df, folder=output_folder)
        
        # Generamos los frames a partir del video. 
        # RECORDAR: El ID del racimo está hardcode en la función video_to_frame
        
        video_path = os.path.join(input_folder, video_name+'.mp4')
        output_frames = os.path.join(output_folder, 'frames')

        # Elimino el ./ del inicio de la ruta del video
        video_path = video_path.replace('./', '') if video_path.startswith('./') else video_path
        video_to_frame(video_path, output_frames, video_name)

        # RECONSTRUCCIÓN DE BAYAS    
        # Elegimos cuál es la mejor reconstrucción: min_mer, min_dist y min_path están preestablecidos, pero se pueden cambiar
        path_minimo, mer_minimo, dist_minimo = get_best_triangulacion(output_path=output_folder, 
                                                                        dists_list=dists_list,
                                                                        calib_path=calib_path,
                                                                        bundles_path=bundles_path,
                                                                        frames_path=output_frames,
                                                                        qr_dist=qr_dist,
                                                                        umbral_triangulacion=umbral_triangulacion,
                                                                        max_workers=max_workers_triangulacion
                                                                    )
        
        print("El mejor path es:", path_minimo)
        print("El mejor mer es:", mer_minimo)
        print("La mejor distancia es:", dist_minimo)
        
        
        # Generación de nubes de puntos
        for i in range(cantidad_nubes):
            print(f"Generando nube de puntos {i+1}...")
            ply_file = os.path.join(output_folder, f'nube_{i}.ply')
            csv_to_ply_with_sphere(path_minimo, ply_file, num_points=100)   # Lee reproyecciones.csv y genera nube de puntos en ply
        
        print("Listo!")
        
        return {
            "success": True
            # "output": result.stdout,
            # "error": result.stderr
        }

    except subprocess.CalledProcessError as e:
        
        detalles = {
            "error": "Error con triangulacion de bayas",
            "command": e.args,
            "exit_code": e.returncode, 
            "stdout": e.stdout,
            "stderr": e.stderr
        }
        
        raise HTTPException(status_code=400, detail=detalles)
