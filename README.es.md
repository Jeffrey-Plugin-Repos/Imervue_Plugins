# Complementos de Imervue

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-TW.md">繁體中文</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <strong>Español</strong> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.pt-BR.md">Português (BR)</a> ·
  <a href="README.ru.md">Русский</a>
</p>

Complementos oficiales para [Imervue](https://github.com/JeffreyChen-s-Utils/Imervue), el visor de imágenes acelerado por GPU.

## Instalación

En Imervue, abre **Plugins → Download Plugins**, elige un complemento y haz clic en **Download**, luego reinicia Imervue (o usa **Plugins → Reload Plugins**). El descargador lee la rama `main` de este repositorio e instala cada complemento en el directorio `plugins/` junto a Imervue (**Plugins → Open Plugin Folder**).

Para instalar a mano, copia un directorio de complemento como `plugins/npr_filters/` en ese directorio `plugins/`. El nivel de categoría (`plugins/`, `languages/`) se descarta al instalar.

Los complementos que necesitan un paquete pesado (onnxruntime, rembg, OpenCV, ...) ofrecen instalarlo la primera vez que usas la función que lo requiere. Los complementos basados en ONNX buscan los archivos de modelo en una carpeta `models/` dentro de su propio directorio de complemento; esos archivos no se descargan, así que coloca ahí el modelo que quieras usar. Un complemento escrito para una versión más nueva de Imervue puede fallar al cargarse en una más antigua; actualiza Imervue si eso ocurre.

## Complementos

| Complemento | Qué hace | Necesita |
| --- | --- | --- |
| `plugins/ai_background_remover` | Elimina fondos de imagen con rembg (U²-Net), una sola imagen o por lotes | rembg, onnxruntime |
| `plugins/ai_colorize` | Colorea fotos en blanco y negro con paletas predefinidas o un modelo ONNX | onnxruntime para la ruta ONNX |
| `plugins/ai_denoise` | Reducción de ruido con filtro bilateral o red neuronal ONNX | onnxruntime para la ruta ONNX |
| `plugins/ai_motion_deblur` | Deconvolución de Wiener o eliminación de desenfoque de movimiento con ONNX | onnxruntime para la ruta ONNX |
| `plugins/ai_object_remove` | Haz clic en un objeto para seleccionarlo y borrarlo por inpainting; indicador de punto SAM opcional | onnxruntime para SAM |
| `plugins/ai_outpaint` | Amplía el lienzo y rellena el nuevo borde | — |
| `plugins/ai_portrait_relight` | Reiluminación direccional de retratos, heurística u ONNX | onnxruntime para la ruta ONNX |
| `plugins/ai_smart_resize` | Redimensionado consciente del contenido (seam carving) | — |
| `plugins/ai_style_transfer` | Transferencia de estilo neuronal rápida con modelos de estilo ONNX | onnxruntime |
| `plugins/cloud_share` | Sube la imagen actual a WebDAV o Imgur (solo HTTPS) | — |
| `plugins/npr_filters` | Estilos de boceto a lápiz, pintura al óleo, acuarela y arte lineal | opencv-python |
| `plugins/object_splitter` | Elimina el fondo y guarda cada objeto como su propio PNG transparente | rembg, onnxruntime |
| `plugins/png_to_icon` | Convierte un PNG en iconos `.png` y `.ico` de varios tamaños | — |
| `plugins/portrait_mode` | Desenfoca el fondo detrás del sujeto detectado | rembg, onnxruntime |
| `plugins/safety_review` | Detecta y pixela regiones explícitas, con un editor manual y exportación de conjunto de datos | nudenet + onnxruntime (fotos), ultralytics + huggingface_hub (anime) |
| `plugins/video_source` | Desplázate por un vídeo y extrae fotogramas fijos al navegador | — |
| `languages/spanish_translation` | Añade el español (Español) al menú de idiomas | — |

«—» significa que el complemento funciona con las dependencias predeterminadas de Imervue.

## Escribir un complemento

La API de complementos (hooks, descubrimiento, dependencias, trabajo en segundo plano, i18n, reglas de distribución) está documentada en un solo lugar: [`PLUGIN_DEV_GUIDE.md`](https://github.com/JeffreyChen-s-Utils/Imervue/blob/main/PLUGIN_DEV_GUIDE.md) en el repositorio de Imervue.

Los complementos se desarrollan y prueban en el repositorio de Imervue bajo `plugins/<name>/` y se reflejan aquí; este repositorio no contiene código propio. Mantén todos los archivos que necesita un complemento directamente dentro de su directorio: el descargador no obtiene subdirectorios.

## Licencia

Consulta [LICENSE](LICENSE).
