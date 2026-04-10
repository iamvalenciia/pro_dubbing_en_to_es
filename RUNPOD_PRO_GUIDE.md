# 🚀 Guía PRO para RunPod: Flujo Ultra-Rápido y Costo-Eficiente

Dado que usarás GPUs de alto calibre (como las **H200 de $3.59/hr** o **B200 de $4.99/hr** que mostraste en tus capturas) el objetivo principal es que **el servidor esté encendido el mínimo tiempo posible**.

El peor error financiero que puedes cometer es arrancar una H200 y pasar 15 minutos esperando a que la app descargue los modelos de IA de Qwen y Whisper (literalmente quemarías un par de dólares solo viendo barras de descarga).

Para evitar esto y hacer que el proceso de *Turn-On / Doblaje / Turn-Off* tarde literalmente segundos, implementaremos la estrategia de **Network Volume Persistente**.

---

## 🛠️ Fase 1: Preparación (Solo se hace 1 vez)

Usaremos una GPU baratísima durante un par de minutos solo para descargar y preparar todo el entorno. Así, cuando actives tu GPU bestial para doblar, todo arrancará al instante sin descargas extra.

1. **Crear el Disco Permanente (Network Volume):**
   - Ve a **Storage** en tu cuenta de RunPod y dale a **+ New Volume**.
   - Ponle de nombre `dubbing-storage`, asígnale unos **30 GB** (es baratísimo, cuesta apenas centavos al mes) y fíjalo en el datacenter/región que piensas usar frecuentemente.

2. **Preparar el Entorno Barato:**
   - Ve a **Pods > Deploy** y escoge una GPU muy barata del **Community Cloud** (P.ej. una RTX 3090 o RTX A4000 a ~$0.30/hr).
   - OJO: En la configuración del despliegue, abre opciones avanzadas y **conecta tu Network Volume** seleccionando la ruta `/workspace`.
   - Lanza el Pod.

3. **Subir y Cachear la Aplicación:**
   - Una vez la instancia inicie, sube los archivos de este proyecto al disco de red montado (vía JupyterLab o SSH). 
   - Abre un terminal en el Pod e instala las dependencias (`pip install -r requirements.txt`).
   - Ejecuta la aplicación por primera vez (`python app.py`). Automáticamente esto empezará a descargar todos los modelos pesados de Whisper y el clonador de voz de Qwen-TTS desde HuggingFace (~10 GB). 
   - Apaga la aplicación con presionar `CTRL+C` en el terminal de RunPod en cuanto terminen las descargas.
   - Dirígete al panel principal de RunPod.
   - **Haz clic en el botón `STOP` de ese Pod barato y seguidamente haz clic en el icono de la papelera (`TERMINATE`)**.
   - *Nota: Al hacer Terminate a ese Pod barato NO se pierde nada. Todo tu proyecto, códigos y modelos de 10 GB se quedaron guardados permanentemente seguros en tu Disco de Red.*

---

## ⚡ Fase 2: Producción Diaria Rápida (Turn ON / Turn OFF)

Tu sistema ya fue preparado. Ahora tu Disco de Red tiene absolutamente todo pre-descargado para acción inmediata. A partir de ahora para doblar tus videos harás exclusivamente este ciclo:

### 🟢 1. TURN ON (Arrancar GPU Fuerte)
1. Vas a **Pods > Deploy**.
2. Te diriges a la sección **Secure Cloud** y seleccionas a tu "monstruo" (P.ej. la `H100 PCIe` o la `H200`).
3. Seleccionas el template "RunPod PyTorch".
4. **⚠️ OJO:** Asegúrate de abrir configuraciones adicionales para **montar el Network Volume (`dubbing-storage`)**.
5. Das en Deploy. 
6. Entras a su JupyterLab/Terminal. Dado que el volumen montado ya contiene los módulos pesados que instalamos la vez pasada, el comando base (`python app.py`) arrancará a la velocidad de la luz en 3 segundos logeándote la URL de Gradio al puerto 7860.

### 🎬 2. Procesar (Hacer el Doblaje)
1. Abres la interfaz gráfica the Gradio.
2. Subes de inmediato el video original.
3. El proceso durará apenas fracciones de tiempo gracias a los miles de VRAM de Tensor Cores que procesarán los audios simultáneamente. 
4. El ffmpeg empaquetará el video con subtítulos incluidos y los reajustes temporales perfectos del paso anterior que desarrollamos (-shortest).

### 💾 3. Recuperar Archivos Finales e Inteligencia
1. Te saldrá el enlace de descarga directa en el navegador ("Download"). Guarda tu `.mp4` y `.mp3` localmente.
2. Acuérdate de usar el botón `Archivar Proyecto` en la misma App para purgar temporales sucios para que al siguiente uso te arranque con el proyecto limpio.

### 🔴 4. TURN OFF (Detener GPU Costosa)
1. Tan pronto el proceso arroje tu MP4 devuelta.
2. Vas a tu Pestaña Principal the **RunPod**.
3. Buscas el servidor activo (Pod).
4. Le dices **`STOP`**.
5. Al darle STOP, RunPod detiene **completamente la facturación gorda** de los 5 dolares la hora.
   - La belleza es que, si mantienes tu pod `STOPPED`, todos tus archivos y estado de tu sistema seguirán tal cual los dejaste para mañana/pasado (Runpod te cobrará sólamente por el almacenamiento inerte de la computadora, pero es irrisoriamente bajo, unos ~$2 mensual).
6. Al día siguiente que tengas tu otro video: Ve a **Runpod -> Start Pod -> Abre App -> Empieza Descarga**. Simple.

> **💡 Regla de Oro:** Siempre vigila que diga "Stopped", y recuerda jamás pagar 4 dolares para "descargar paquetes de Python". Esa es la lógica clave para usar hardware astronómico inteligentemente.
