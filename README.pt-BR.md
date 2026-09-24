# Plugins do Imervue

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-TW.md">繁體中文</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <strong>Português (BR)</strong> ·
  <a href="README.ru.md">Русский</a>
</p>

Plugins oficiais para o [Imervue](https://github.com/JeffreyChen-s-Utils/Imervue), o visualizador de imagens acelerado por GPU.

## Instalação

No Imervue, abra **Plugins → Download Plugins**, escolha um plugin e clique em **Download**, depois reinicie o Imervue (ou use **Plugins → Reload Plugins**). O downloader lê o branch `main` deste repositório e instala cada plugin no diretório `plugins/` ao lado do Imervue (**Plugins → Open Plugin Folder**).

Para instalar manualmente, copie um diretório de plugin como `plugins/npr_filters/` para esse diretório `plugins/`. O nível de categoria (`plugins/`, `languages/`) é descartado na instalação.

Plugins que precisam de um pacote pesado (onnxruntime, rembg, OpenCV, ...) oferecem instalá-lo na primeira vez que você usa o recurso que o requer. Os plugins baseados em ONNX procuram arquivos de modelo em uma pasta `models/` dentro do próprio diretório do plugin; esses arquivos não são baixados, então coloque ali o modelo que quiser usar. Um plugin escrito para um Imervue mais novo pode falhar ao carregar em um mais antigo; atualize o Imervue se isso acontecer.

## Plugins

| Plugin | O que faz | Precisa de |
| --- | --- | --- |
| `plugins/ai_background_remover` | Remove fundos de imagem com rembg (U²-Net), imagem única ou em lote | rembg, onnxruntime |
| `plugins/ai_colorize` | Coloriza fotos em preto e branco com paletas predefinidas ou um modelo ONNX | onnxruntime para o caminho ONNX |
| `plugins/ai_denoise` | Redução de ruído por filtro bilateral ou rede neural ONNX | onnxruntime para o caminho ONNX |
| `plugins/ai_motion_deblur` | Deconvolução de Wiener ou remoção de desfoque de movimento por ONNX | onnxruntime para o caminho ONNX |
| `plugins/ai_object_remove` | Clique em um objeto para selecioná-lo e apagá-lo por inpainting; prompt de ponto SAM opcional | onnxruntime para o SAM |
| `plugins/ai_outpaint` | Estende a tela e preenche a nova borda | — |
| `plugins/ai_portrait_relight` | Reiluminação direcional de retrato, heurística ou ONNX | onnxruntime para o caminho ONNX |
| `plugins/ai_smart_resize` | Redimensionamento sensível ao conteúdo (seam carving) | — |
| `plugins/ai_style_transfer` | Transferência de estilo neural rápida com modelos de estilo ONNX | onnxruntime |
| `plugins/cloud_share` | Envia a imagem atual para WebDAV ou Imgur (somente HTTPS) | — |
| `plugins/npr_filters` | Estilos de esboço a lápis, pintura a óleo, aquarela e arte de linha | opencv-python |
| `plugins/object_splitter` | Remove o fundo e salva cada objeto como seu próprio PNG transparente | rembg, onnxruntime |
| `plugins/png_to_icon` | Converte um PNG em ícones `.png` e `.ico` de vários tamanhos | — |
| `plugins/portrait_mode` | Desfoca o fundo atrás do sujeito detectado | rembg, onnxruntime |
| `plugins/safety_review` | Detecta e aplica mosaico em regiões explícitas, com um editor manual e exportação de conjunto de dados | nudenet + onnxruntime (fotos), ultralytics + huggingface_hub (anime) |
| `plugins/video_source` | Percorra um vídeo e extraia quadros estáticos para o navegador | — |
| `languages/spanish_translation` | Adiciona o espanhol (Español) ao menu de idiomas | — |

"—" significa que o plugin funciona com as dependências padrão do Imervue.

## Escrevendo um plugin

A API de plugins (hooks, descoberta, dependências, trabalho em segundo plano, i18n, regras de distribuição) está documentada em um único lugar: [`PLUGIN_DEV_GUIDE.md`](https://github.com/JeffreyChen-s-Utils/Imervue/blob/main/PLUGIN_DEV_GUIDE.md) no repositório do Imervue.

Os plugins são desenvolvidos e testados no repositório do Imervue em `plugins/<name>/` e espelhados aqui; este repositório não contém código próprio. Mantenha todos os arquivos de que um plugin precisa diretamente dentro de seu diretório: o downloader não busca subdiretórios.

## Licença

Consulte [LICENSE](LICENSE).
