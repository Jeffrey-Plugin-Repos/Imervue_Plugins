# Плагины Imervue

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-TW.md">繁體中文</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.pt-BR.md">Português (BR)</a> ·
  <strong>Русский</strong>
</p>

Официальные плагины для [Imervue](https://github.com/JeffreyChen-s-Utils/Imervue), просмотрщика изображений с ускорением на GPU.

## Установка

В Imervue откройте **Plugins → Download Plugins**, выберите плагин и нажмите **Download**, затем перезапустите Imervue (или используйте **Plugins → Reload Plugins**). Загрузчик читает ветку `main` этого репозитория и устанавливает каждый плагин в каталог `plugins/` рядом с Imervue (**Plugins → Open Plugin Folder**).

Чтобы установить вручную, скопируйте каталог плагина, например `plugins/npr_filters/`, в этот каталог `plugins/`. Уровень категории (`plugins/`, `languages/`) при установке отбрасывается.

Плагины, которым нужен тяжёлый пакет (onnxruntime, rembg, OpenCV, ...), предложат установить его при первом использовании функции, которая его требует. Плагины на основе ONNX ищут файлы моделей в папке `models/` внутри собственного каталога плагина; эти файлы не загружаются, поэтому поместите туда нужную вам модель. Плагин, написанный для более новой версии Imervue, может не загрузиться в более старой; в этом случае обновите Imervue.

## Плагины

| Плагин | Что делает | Требует |
| --- | --- | --- |
| `plugins/ai_background_remover` | Удаляет фон изображения с помощью rembg (U²-Net), одиночное изображение или пакет | rembg, onnxruntime |
| `plugins/ai_colorize` | Раскрашивает чёрно-белые фотографии с помощью готовых палитр или модели ONNX | onnxruntime для пути ONNX |
| `plugins/ai_denoise` | Билатеральный фильтр или нейросетевое шумоподавление ONNX | onnxruntime для пути ONNX |
| `plugins/ai_motion_deblur` | Деконволюция Винера или устранение смаза движения через ONNX | onnxruntime для пути ONNX |
| `plugins/ai_object_remove` | Щёлкните по объекту, чтобы выделить его и убрать инпейнтингом; необязательная точечная подсказка SAM | onnxruntime для SAM |
| `plugins/ai_outpaint` | Расширяет холст и заполняет новую границу | — |
| `plugins/ai_portrait_relight` | Направленное переосвещение портрета, эвристика или ONNX | onnxruntime для пути ONNX |
| `plugins/ai_smart_resize` | Изменение размера с учётом содержимого (seam carving) | — |
| `plugins/ai_style_transfer` | Быстрый нейросетевой перенос стиля с моделями стиля ONNX | onnxruntime |
| `plugins/cloud_share` | Загружает текущее изображение в WebDAV или Imgur (только HTTPS) | — |
| `plugins/npr_filters` | Стили: карандашный набросок, масляная живопись, акварель и контурный рисунок | opencv-python |
| `plugins/object_splitter` | Удаляет фон и сохраняет каждый объект как отдельный прозрачный PNG | rembg, onnxruntime |
| `plugins/png_to_icon` | Преобразует PNG в значки `.png` и `.ico` нескольких размеров | — |
| `plugins/portrait_mode` | Размывает фон за обнаруженным объектом | rembg, onnxruntime |
| `plugins/safety_review` | Обнаруживает и заштриховывает мозаикой откровенные области, с ручным редактором и экспортом набора данных | nudenet + onnxruntime (фото), ultralytics + huggingface_hub (аниме) |
| `plugins/video_source` | Прокручивайте видео и извлекайте стоп-кадры в браузер | — |
| `languages/spanish_translation` | Добавляет испанский (Español) в меню языков | — |

«—» означает, что плагин работает на стандартных зависимостях Imervue.

## Написание плагина

API плагинов (хуки, обнаружение, зависимости, фоновая работа, i18n, правила распространения) документирован в одном месте: [`PLUGIN_DEV_GUIDE.md`](https://github.com/JeffreyChen-s-Utils/Imervue/blob/main/PLUGIN_DEV_GUIDE.md) в репозитории Imervue.

Плагины разрабатываются и тестируются в репозитории Imervue в каталоге `plugins/<name>/` и зеркалируются сюда; этот репозиторий не содержит собственного кода. Держите каждый файл, который нужен плагину, непосредственно внутри его каталога: загрузчик не забирает подкаталоги.

## Лицензия

См. [LICENSE](LICENSE).
