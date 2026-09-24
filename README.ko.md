# Imervue 플러그인

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-TW.md">繁體中文</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <strong>한국어</strong> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.pt-BR.md">Português (BR)</a> ·
  <a href="README.ru.md">Русский</a>
</p>

GPU 가속 이미지 뷰어 [Imervue](https://github.com/JeffreyChen-s-Utils/Imervue)의 공식 플러그인입니다.

## 설치

Imervue에서 **Plugins → Download Plugins**를 열고 플러그인을 선택한 뒤 **Download**를 클릭하고 Imervue를 다시 시작합니다(또는 **Plugins → Reload Plugins** 사용). 다운로더는 이 저장소의 `main` 브랜치를 읽어 각 플러그인을 Imervue 옆의 `plugins/` 디렉터리에 설치합니다(**Plugins → Open Plugin Folder**).

직접 설치하려면 `plugins/npr_filters/`와 같은 플러그인 디렉터리를 그 `plugins/` 디렉터리로 복사하세요. 설치 시 카테고리 계층(`plugins/`, `languages/`)은 제거됩니다.

무거운 패키지(onnxruntime, rembg, OpenCV 등)가 필요한 플러그인은 해당 기능을 처음 사용할 때 설치를 제안합니다. ONNX 기반 플러그인은 자체 플러그인 디렉터리 안의 `models/` 폴더에서 모델 파일을 찾습니다. 이 파일들은 다운로드되지 않으므로 사용하려는 모델을 그곳에 두세요. 더 새로운 Imervue를 대상으로 작성된 플러그인은 더 오래된 Imervue에서 로드되지 않을 수 있습니다. 그런 경우 Imervue를 업데이트하세요.

## 플러그인

| 플러그인 | 기능 | 필요 항목 |
| --- | --- | --- |
| `plugins/ai_background_remover` | rembg(U²-Net)로 이미지 배경 제거, 단일 이미지 또는 일괄 처리 | rembg, onnxruntime |
| `plugins/ai_colorize` | 프리셋 팔레트 또는 ONNX 모델로 흑백 사진에 색 입히기 | ONNX 경로에는 onnxruntime |
| `plugins/ai_denoise` | 양방향 필터 또는 ONNX 신경망 노이즈 제거 | ONNX 경로에는 onnxruntime |
| `plugins/ai_motion_deblur` | Wiener 디컨볼루션 또는 ONNX 모션 블러 제거 | ONNX 경로에는 onnxruntime |
| `plugins/ai_object_remove` | 객체를 클릭해 선택하고 인페인팅으로 지우기; 선택적 SAM 포인트 프롬프트 | SAM에는 onnxruntime |
| `plugins/ai_outpaint` | 캔버스를 확장하고 새 경계를 채우기 | — |
| `plugins/ai_portrait_relight` | 방향성 인물 리라이팅, 휴리스틱 또는 ONNX | ONNX 경로에는 onnxruntime |
| `plugins/ai_smart_resize` | 내용 인식 크기 조정(심 카빙) | — |
| `plugins/ai_style_transfer` | ONNX 스타일 모델을 사용한 빠른 신경망 스타일 전이 | onnxruntime |
| `plugins/cloud_share` | 현재 이미지를 WebDAV 또는 Imgur에 업로드(HTTPS만) | — |
| `plugins/npr_filters` | 연필 스케치, 유화, 수채화, 라인아트 스타일 | opencv-python |
| `plugins/object_splitter` | 배경을 제거하고 각 객체를 개별 투명 PNG로 저장 | rembg, onnxruntime |
| `plugins/png_to_icon` | PNG를 여러 크기의 `.png` 및 `.ico` 아이콘으로 변환 | — |
| `plugins/portrait_mode` | 감지된 피사체 뒤의 배경을 흐리게 처리 | rembg, onnxruntime |
| `plugins/safety_review` | 노골적인 영역을 감지해 모자이크 처리, 수동 편집기와 데이터셋 내보내기 포함 | 사진은 nudenet + onnxruntime, 애니메이션은 ultralytics + huggingface_hub |
| `plugins/video_source` | 동영상을 스크럽하며 정지 프레임을 브라우저로 추출 | — |
| `languages/spanish_translation` | 언어 메뉴에 스페인어(Español) 추가 | — |

"—"는 해당 플러그인이 Imervue의 기본 의존성만으로 동작함을 의미합니다.

## 플러그인 작성

플러그인 API(훅, 검색, 의존성, 백그라운드 작업, i18n, 배포 규칙)는 한곳에 정리되어 있습니다. Imervue 저장소의 [`PLUGIN_DEV_GUIDE.md`](https://github.com/JeffreyChen-s-Utils/Imervue/blob/main/PLUGIN_DEV_GUIDE.md)를 참조하세요.

플러그인은 Imervue 저장소의 `plugins/<name>/` 아래에서 개발 및 테스트되어 이곳으로 미러링됩니다. 이 저장소 자체에는 코드가 없습니다. 플러그인에 필요한 모든 파일을 그 디렉터리 바로 안에 두세요. 다운로더는 하위 디렉터리를 가져오지 않습니다.

## 라이선스

[LICENSE](LICENSE)를 참조하세요.
