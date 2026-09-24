# Imervue-Plugins

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-TW.md">繁體中文</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <strong>Deutsch</strong> ·
  <a href="README.pt-BR.md">Português (BR)</a> ·
  <a href="README.ru.md">Русский</a>
</p>

Offizielle Plugins für [Imervue](https://github.com/JeffreyChen-s-Utils/Imervue), den GPU-beschleunigten Bildbetrachter.

## Installation

Öffne in Imervue **Plugins → Download Plugins**, wähle ein Plugin und klicke auf **Download**, starte dann Imervue neu (oder verwende **Plugins → Reload Plugins**). Der Downloader liest den `main`-Branch dieses Repositorys und installiert jedes Plugin in das `plugins/`-Verzeichnis neben Imervue (**Plugins → Open Plugin Folder**).

Zum manuellen Installieren kopierst du ein Plugin-Verzeichnis wie `plugins/npr_filters/` in dieses `plugins/`-Verzeichnis. Die Kategorieebene (`plugins/`, `languages/`) wird bei der Installation entfernt.

Plugins, die ein umfangreiches Paket benötigen (onnxruntime, rembg, OpenCV, ...), bieten dessen Installation an, wenn du die zugehörige Funktion zum ersten Mal verwendest. Die ONNX-basierten Plugins suchen Modelldateien in einem `models/`-Ordner innerhalb ihres eigenen Plugin-Verzeichnisses; diese Dateien werden nicht heruntergeladen, lege also das gewünschte Modell dort ab. Ein Plugin, das für ein neueres Imervue geschrieben wurde, lässt sich auf einem älteren möglicherweise nicht laden; aktualisiere Imervue in diesem Fall.

## Plugins

| Plugin | Was es tut | Benötigt |
| --- | --- | --- |
| `plugins/ai_background_remover` | Entfernt Bildhintergründe mit rembg (U²-Net), Einzelbild oder Stapel | rembg, onnxruntime |
| `plugins/ai_colorize` | Koloriert Schwarz-Weiß-Fotos mit voreingestellten Paletten oder einem ONNX-Modell | onnxruntime für den ONNX-Pfad |
| `plugins/ai_denoise` | Rauschunterdrückung per Bilateralfilter oder ONNX-Neuronalnetz | onnxruntime für den ONNX-Pfad |
| `plugins/ai_motion_deblur` | Wiener-Entfaltung oder ONNX-Bewegungsunschärfeentfernung | onnxruntime für den ONNX-Pfad |
| `plugins/ai_object_remove` | Klicke ein Objekt an, um es auszuwählen und per Inpainting zu entfernen; optionaler SAM-Punkt-Prompt | onnxruntime für SAM |
| `plugins/ai_outpaint` | Erweitert die Leinwand und füllt den neuen Rand | — |
| `plugins/ai_portrait_relight` | Gerichtete Porträt-Neubeleuchtung, heuristisch oder ONNX | onnxruntime für den ONNX-Pfad |
| `plugins/ai_smart_resize` | Inhaltsbewusste Größenänderung (Seam Carving) | — |
| `plugins/ai_style_transfer` | Schneller neuronaler Stiltransfer mit ONNX-Stilmodellen | onnxruntime |
| `plugins/cloud_share` | Lädt das aktuelle Bild zu WebDAV oder Imgur hoch (nur HTTPS) | — |
| `plugins/npr_filters` | Stile: Bleistiftskizze, Ölmalerei, Aquarell und Strichzeichnung | opencv-python |
| `plugins/object_splitter` | Entfernt den Hintergrund und speichert jedes Objekt als eigenes transparentes PNG | rembg, onnxruntime |
| `plugins/png_to_icon` | Wandelt ein PNG in `.png`- und `.ico`-Symbole in mehreren Größen um | — |
| `plugins/portrait_mode` | Verschwimmt den Hintergrund hinter dem erkannten Motiv | rembg, onnxruntime |
| `plugins/safety_review` | Erkennt und verpixelt explizite Bereiche, mit manuellem Editor und Datensatz-Export | nudenet + onnxruntime (Fotos), ultralytics + huggingface_hub (Anime) |
| `plugins/video_source` | Durchsuche ein Video und extrahiere Standbilder in den Browser | — |
| `languages/spanish_translation` | Fügt dem Sprachmenü Spanisch (Español) hinzu | — |

„—" bedeutet, dass das Plugin mit den Standardabhängigkeiten von Imervue läuft.

## Ein Plugin schreiben

Die Plugin-API (Hooks, Erkennung, Abhängigkeiten, Hintergrundarbeit, i18n, Verteilungsregeln) ist an einer Stelle dokumentiert: [`PLUGIN_DEV_GUIDE.md`](https://github.com/JeffreyChen-s-Utils/Imervue/blob/main/PLUGIN_DEV_GUIDE.md) im Imervue-Repository.

Plugins werden im Imervue-Repository unter `plugins/<name>/` entwickelt und getestet und hierher gespiegelt; dieses Repository enthält keinen eigenen Code. Halte jede Datei, die ein Plugin benötigt, direkt in seinem Verzeichnis: Der Downloader ruft keine Unterverzeichnisse ab.

## Lizenz

Siehe [LICENSE](LICENSE).
