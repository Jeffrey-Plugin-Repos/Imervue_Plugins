# Extensions Imervue

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-TW.md">繁體中文</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.es.md">Español</a> ·
  <strong>Français</strong> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.pt-BR.md">Português (BR)</a> ·
  <a href="README.ru.md">Русский</a>
</p>

Extensions officielles pour [Imervue](https://github.com/JeffreyChen-s-Utils/Imervue), la visionneuse d'images accélérée par GPU.

## Installation

Dans Imervue, ouvrez **Plugins → Download Plugins**, choisissez une extension et cliquez sur **Download**, puis redémarrez Imervue (ou utilisez **Plugins → Reload Plugins**). Le téléchargeur lit la branche `main` de ce dépôt et installe chaque extension dans le répertoire `plugins/` situé à côté d'Imervue (**Plugins → Open Plugin Folder**).

Pour installer à la main, copiez un répertoire d'extension tel que `plugins/npr_filters/` dans ce répertoire `plugins/`. Le niveau de catégorie (`plugins/`, `languages/`) est supprimé à l'installation.

Les extensions qui nécessitent un paquet lourd (onnxruntime, rembg, OpenCV, ...) proposent de l'installer la première fois que vous utilisez la fonctionnalité qui en a besoin. Les extensions basées sur ONNX recherchent les fichiers de modèle dans un dossier `models/` à l'intérieur de leur propre répertoire d'extension ; ces fichiers ne sont pas téléchargés, alors placez-y le modèle que vous voulez utiliser. Une extension écrite pour une version plus récente d'Imervue peut ne pas se charger sur une version plus ancienne ; mettez Imervue à jour si cela se produit.

## Extensions

| Extension | Ce qu'elle fait | Nécessite |
| --- | --- | --- |
| `plugins/ai_background_remover` | Supprime les arrière-plans d'image avec rembg (U²-Net), image seule ou par lot | rembg, onnxruntime |
| `plugins/ai_colorize` | Colorise des photos en noir et blanc avec des palettes prédéfinies ou un modèle ONNX | onnxruntime pour la voie ONNX |
| `plugins/ai_denoise` | Débruitage par filtre bilatéral ou réseau neuronal ONNX | onnxruntime pour la voie ONNX |
| `plugins/ai_motion_deblur` | Déconvolution de Wiener ou suppression du flou de mouvement par ONNX | onnxruntime pour la voie ONNX |
| `plugins/ai_object_remove` | Cliquez sur un objet pour le sélectionner et l'effacer par inpainting ; invite de point SAM optionnelle | onnxruntime pour SAM |
| `plugins/ai_outpaint` | Étend le canevas et remplit la nouvelle bordure | — |
| `plugins/ai_portrait_relight` | Ré-éclairage directionnel de portrait, heuristique ou ONNX | onnxruntime pour la voie ONNX |
| `plugins/ai_smart_resize` | Redimensionnement sensible au contenu (seam carving) | — |
| `plugins/ai_style_transfer` | Transfert de style neuronal rapide avec des modèles de style ONNX | onnxruntime |
| `plugins/cloud_share` | Téléverse l'image actuelle vers WebDAV ou Imgur (HTTPS uniquement) | — |
| `plugins/npr_filters` | Styles croquis au crayon, peinture à l'huile, aquarelle et dessin au trait | opencv-python |
| `plugins/object_splitter` | Supprime l'arrière-plan et enregistre chaque objet dans son propre PNG transparent | rembg, onnxruntime |
| `plugins/png_to_icon` | Convertit un PNG en icônes `.png` et `.ico` multi-tailles | — |
| `plugins/portrait_mode` | Floute l'arrière-plan derrière le sujet détecté | rembg, onnxruntime |
| `plugins/safety_review` | Détecte et pixellise les régions explicites, avec un éditeur manuel et export de jeu de données | nudenet + onnxruntime (photos), ultralytics + huggingface_hub (anime) |
| `plugins/video_source` | Parcourez une vidéo et extrayez des images fixes vers le navigateur | — |
| `languages/spanish_translation` | Ajoute l'espagnol (Español) au menu des langues | — |

« — » signifie que l'extension fonctionne avec les dépendances par défaut d'Imervue.

## Écrire une extension

L'API d'extension (hooks, découverte, dépendances, travail en arrière-plan, i18n, règles de distribution) est documentée en un seul endroit : [`PLUGIN_DEV_GUIDE.md`](https://github.com/JeffreyChen-s-Utils/Imervue/blob/main/PLUGIN_DEV_GUIDE.md) dans le dépôt Imervue.

Les extensions sont développées et testées dans le dépôt Imervue sous `plugins/<name>/` puis reflétées ici ; ce dépôt ne contient aucun code propre. Gardez chaque fichier dont une extension a besoin directement dans son répertoire : le téléchargeur ne récupère pas les sous-répertoires.

## Licence

Voir [LICENSE](LICENSE).
