# RunPod Workflow Definitivo (Pod + Network Volume + Subidas desde PC)

Esta guia esta pensada para tu flujo real:
- Encender Pod
- Procesar doblaje
- Descargar resultados
- Apagar Pod
- Conservar insumos/modelos en Network Volume

Asi no necesitas reconstruir imagen cada vez ni perder archivos al terminar la instancia.

## 1) Idea clave

En RunPod hay dos zonas de almacenamiento:

1. Network Volume (persistente)
- Sobrevive cuando apagas o terminas Pods.
- Lo accedes desde tu PC por endpoint S3 compatible.
- Ideal para: user_input, output_final, modelos cacheados.

2. Disco local del Pod (efimero)
- Rapido, pero se pierde al terminar el Pod.
- Ideal para temporales de procesamiento.

Recomendacion practica:
- Subir insumos y guardar finales en Network Volume.
- Usar disco local solo para temporales.

## 2) Variables sugeridas en tu PC (PowerShell)

Ajusta una vez por sesion:

$env:RUNPOD_S3_ENDPOINT = "https://s3api-us-ks-2.runpod.io"
$env:RUNPOD_REGION = "us-ks-2"
$env:RUNPOD_BUCKET = "l9dt5rqorw"

Opcional (si quieres simplificar comandos):

$env:AWS_DEFAULT_REGION = $env:RUNPOD_REGION

## 3) Verificar contenido actual del Network Volume

aws s3 ls s3://$env:RUNPOD_BUCKET/ --recursive --human-readable --summarize --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

En tu caso, ya viste que existe:
- user_input/full_video_en.mp4

Eso significa que el video esta listo para ser referenciado en el flujo de doblaje.

## 4) Subir archivos desde tu computadora al Network Volume

### 4.1 Subir un archivo puntual de entrada

aws s3 cp "C:\ruta\a\tu\video.mp4" "s3://$env:RUNPOD_BUCKET/user_input/full_video_en.mp4" --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

### 4.2 Subir carpeta completa de entradas

aws s3 sync "C:\ruta\a\inputs" "s3://$env:RUNPOD_BUCKET/user_input/" --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

### 4.3 Subir modelos/cache predescargados

aws s3 sync "C:\ruta\a\models" "s3://$env:RUNPOD_BUCKET/models/" --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

## 5) Descargar resultados del Network Volume a tu PC

### 5.1 Descargar salida final completa

aws s3 sync "s3://$env:RUNPOD_BUCKET/output_final/" "C:\ruta\local\output_final" --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

### 5.2 Descargar un archivo especifico

aws s3 cp "s3://$env:RUNPOD_BUCKET/output_final/mi_video_doblado.mp4" "C:\ruta\local\mi_video_doblado.mp4" --region $env:RUNPOD_REGION --endpoint-url $env:RUNPOD_S3_ENDPOINT

## 6) Comandos base para manejar Pods (runpodctl)

### 6.1 Ver Pods

runpodctl pod list
runpodctl pod list --all

### 6.2 Ver detalle de un Pod

runpodctl pod get POD_ID

### 6.3 Crear Pod con tu imagen y Network Volume

runpodctl pod create --name "qdp-dubbing" --image "TU_USUARIO/TU_IMAGEN:latest" --gpu-id "NVIDIA GeForce RTX 4090" --container-disk-in-gb 50 --network-volume-id "TU_NETWORK_VOLUME_ID" --volume-mount-path "/runpod-volume" --ports "7860/http,22/tcp" --env '{"QDP_NETWORK_DIR":"/runpod-volume","QDP_LOCAL_DIR":"/workspace/qdp_data","HF_HOME":"/workspace/qdp_data/torch_cache","TORCH_HOME":"/workspace/qdp_data/torch_cache"}'

Nota:
- Si no quieres montar volumen ahora, puedes omitir network-volume-id y volume-mount-path.
- Pero para flujo definitivo, si conviene montarlo.

### 6.4 Encender, apagar, reiniciar, borrar

runpodctl pod start POD_ID
runpodctl pod stop POD_ID
runpodctl pod restart POD_ID
runpodctl pod delete POD_ID

## 7) Flujo diario recomendado (simple y estable)

1. Subir video a user_input desde tu PC.
2. Encender Pod existente (o crear uno si no existe).
3. Ejecutar doblaje en la app.
4. Verificar que salida quede en output_final.
5. Descargar resultados con aws s3 sync/cp.
6. Apagar Pod para no pagar compute innecesario.

Con esto ya no necesitas reconstruir imagen en cada corrida.

## 8) Buenas practicas para evitar reprocesos

- Mantener nombres estables:
  - user_input/full_video_en.mp4
  - output_final/...
- Guardar modelos en models/ dentro del bucket.
- Usar disco local del Pod solo para temporales.
- Antes de apagar, verificar que output_final ya este en Network Volume.

## 9) Troubleshooting rapido

1. Error de credenciales AWS CLI
- Ejecuta aws configure o define AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY para tu S3 de RunPod.

2. No aparece archivo al listar
- Revisa bucket, region y endpoint.
- Ejecuta de nuevo el ls con --summarize.

3. El Pod inicia pero no encuentras archivos
- Confirma que el Pod fue creado con network-volume-id y volume-mount-path correctos.
- Confirma variables QDP_NETWORK_DIR y QDP_LOCAL_DIR.

4. Quieres conservar trabajo entre sesiones
- Guarda insumos/modelos/salidas en Network Volume.
- No dependas del disco local del Pod para persistencia.

## 10) Comandos listos para tu caso actual

Listar todo:

aws s3 ls s3://l9dt5rqorw/ --recursive --human-readable --summarize --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io

Subir un video nuevo:

aws s3 cp "C:\ruta\local\full_video_en.mp4" "s3://l9dt5rqorw/user_input/full_video_en.mp4" --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io

Descargar todos los resultados:

aws s3 sync "s3://l9dt5rqorw/output_final/" "C:\ruta\local\output_final" --region us-ks-2 --endpoint-url https://s3api-us-ks-2.runpod.io
