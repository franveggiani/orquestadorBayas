from pydantic import BaseModel

class UserRequest(BaseModel): 
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