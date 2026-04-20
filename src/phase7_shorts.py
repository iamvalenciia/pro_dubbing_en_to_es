import os
import cv2
import ffmpeg
from tqdm import tqdm

def run_phase7_shorts(video_path: str, output_short_path: str):
    """
    Fase 7: Creación de Shorts (9:16) con Face-Tracking Básico.
    Utiliza OpenCV Haarcascades pre-entrenados para mantener la cara en el centro.
    """
    print(f"--- Fase 7: Creación de Shorts Iniciada [{video_path}] ---")
    
    # Cargar clasificador frontal (incluido pip opencv-python)
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calcular target 9:16 (Manteniendo el height intacto)
    target_width = int(height * 9 / 16)
    
    temp_video_only = os.path.join(os.path.dirname(output_short_path), "temp_vid.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_only, fourcc, fps, (target_width, height))
    
    last_center = width // 2
    
    for _ in tqdm(range(total_frames), desc="Generando Short (Tracking)..."):
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Rostro principal (mayormente)
            last_center = x + w // 2
            
        # Suavizado de cámara o centro directo
        left_x = last_center - target_width // 2
        right_x = last_center + target_width // 2
        
        # Limites para no salir del frame real
        if left_x < 0:
            left_x = 0
            right_x = target_width
        elif right_x > width:
            right_x = width
            left_x = width - target_width
            
        cropped = frame[0:height, left_x:right_x]
        # Validar crop perfecto por bugs de CV2 de bordes
        if cropped.shape[1] == target_width and cropped.shape[0] == height:
            out.write(cropped)
        else:
            # Relleno negro si falla el borde (raro si es 16:9 -> 9:16 real sin escalados extra)
            fixed = cv2.resize(cropped, (target_width, height))
            out.write(fixed)
            
    cap.release()
    out.release()
    
    print("Integrando Audio Master al Short...")
    # Combina el audio del master de 16:9 original al nuevo video de cv2
    v_stream = ffmpeg.input(temp_video_only)
    a_stream = ffmpeg.input(video_path).audio

    # NVENC por default en A100; env SHORTS_VCODEC=libx264 para fallback CPU.
    _shorts_vcodec = os.environ.get("SHORTS_VCODEC", "h264_nvenc")
    _shorts_preset = "p4" if _shorts_vcodec.endswith("_nvenc") else "medium"
    ffmpeg.output(
        v_stream, a_stream, output_short_path,
        vcodec=_shorts_vcodec, acodec="aac",
        preset=_shorts_preset, strict="experimental",
    ).run(overwrite_output=True)
    os.remove(temp_video_only)
    
    print(f"--- Fase 7: Finalizada. Short Listo: {output_short_path} ---")
    return output_short_path
