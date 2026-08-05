from pydantic import BaseModel


class DetectorRequest(BaseModel):
    input_folder: str
    output_folder: str
    video_name: str
