# 🎤 Voice Records Transcript Summarizer

**Transcripción y resumen automatizado de grabaciones de campo con Whisper + HuggingFace**

---

## 📘 Descripción

Este script procesa grabaciones de voz (`.m4a`, `.wav`, etc.), las convierte a texto mediante modelos de **Whisper**, y genera resúmenes automáticos con modelos de **transformers** de HuggingFace.
Está diseñado para trabajo en campo (entrevistas, visitas, observaciones) y produce un **informe consolidado** con texto limpio, timestamps opcionales y resúmenes por archivo.

---

## ⚙️ Requisitos del sistema

* **Python** ≥ 3.10
* **ffmpeg** instalado y disponible en PATH

  * macOS → `brew install ffmpeg`
  * Ubuntu/Debian → `sudo apt install ffmpeg`
* (Opcional) GPU NVIDIA para acelerar la transcripción con CUDA

---

## 📦 Instalación

1. Crear y activar un entorno virtual:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Uso

Ejecuta el script principal con tus archivos de audio:

```bash
python3 recordings_report.py \
  --input "/ruta/a/carpeta/de/audios" \
  --outdir "./salida_informes" \
  --asr_model "openai/whisper-base" \
  --sum_model "pszemraj/long-t5-tglobal-base-16384-book-summary" \
  --language "es" \
  --order name \
  --device auto \
  --clean_tmp
```

---

## 🧠 Parámetros disponibles

| Parámetro           | Descripción                                            | Ejemplo                                            |
| ------------------- | ------------------------------------------------------ | -------------------------------------------------- |
| `--input`           | Carpeta con archivos `.m4a`, `.wav`, etc.              | `/Users/javierbatres/Downloads/audios`             |
| `--outdir`          | Carpeta donde guardar las transcripciones y resúmenes. | `./salida`                                         |
| `--asr_model`       | Modelo de transcripción Whisper (HuggingFace o local). | `openai/whisper-base`                              |
| `--sum_model`       | Modelo de resumen de texto (fallback incluido).        | `pszemraj/long-t5-tglobal-base-16384-book-summary` |
| `--language`        | Idioma principal del audio (`es`, `en`, etc.).         | `es`                                               |
| `--device`          | `cpu`, `cuda`, o `auto` (detecta GPU si disponible).   | `auto`                                             |
| `--order`           | Orden de procesamiento (`name` o `size`).              | `name`                                             |
| `--keep_timestamps` | Mantiene timestamps en la transcripción.               | (flag)                                             |
| `--clean_tmp`       | Limpia archivos temporales `.wav` tras el proceso.     | (flag)                                             |

---

## ⚡ Consejos de rendimiento

* Para **CPU**: usa modelos más ligeros (`openai/whisper-tiny` o `small`).
* Para **GPU (Linux)**: Whisper grande (`medium` o `large-v3`) acelera hasta 10×.
* Si el script se cuelga en un archivo grande, prueba convertir manualmente a WAV:

  ```bash
  ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le output.wav
  ```

---

## 🧬 Modelos recomendados

| Tipo                      | Modelo                                             | Descripción                               |
| ------------------------- | -------------------------------------------------- | ----------------------------------------- |
| **ASR (transcripción)**   | `openai/whisper-small`                             | Buen balance entre velocidad y precisión. |
| **ASR (rápido)**          | `guillaumekln/faster-whisper-small`                | Versión optimizada para CPU/GPU.          |
| **Resumen (texto largo)** | `pszemraj/long-t5-tglobal-base-16384-book-summary` | Ideal para entrevistas largas.            |
| **Resumen (alternativo)** | `google/pegasus-xsum`                              | Más rápido, menos contexto.               |

---

## 🛠 Mantenimiento y soporte

* Asegúrate de tener la última versión de **transformers** y **torch**.
* Si HuggingFace pide autenticación, ejecuta:

  ```bash
  huggingface-cli login
  ```
* Para depuración:

  ```bash
  TRANSFORMERS_VERBOSITY=info python3 recordings_report.py ...
  ```

---

## 📄 Licencia

MIT License © 2025 — Javier Batres

---

## ☕ Autor

**Javier Batres**

Cloud & Software Architect

📧 [LinkedIn](https://www.linkedin.com/in/fjbatresv/)
