# 程式審核：Imervue_Plugins

## 審核者

| 項目 | 內容 |
| --- | --- |
| 審核者 | 工作區程式審核 |
| 日期 | 2026-10-22 |
| 對象 | `Imervue_Plugins` |
| 範圍 | 工作樹現況 |
| 結論 | 需修正 |

## 審核規範

1. 設計：外掛維持扁平目錄、進入點清楚，重依賴留在執行期安裝，不把主程式責任塞進單一外掛。
2. 可讀性：圈複雜度不超過 15、函式不超過 80 行、巢狀不超過 4、參數不超過 7、單行不超過 120；專案若有更嚴規定，從嚴。
3. 錯誤處理：失敗要讓呼叫端分辨，不可把失敗報成成功；邊界錯誤要記錄，不可裸吞。
4. 死碼：不留未使用的進入路徑或只為日誌而存在、卻影響控制流的分支。
5. 型別與 docstring：公開進入點與純函式要有型別與說明，且說明須與實作一致。
6. 安全：不寫死祕密；不可信輸入不得進 `eval` / `exec` / `shell=True` / `pickle` / 不安全的 `yaml.load`；不得關閉 TLS 驗證。
7. 測試：行為變更應有測試。本倉庫依契約只做鏡像，測試留在 Imervue，不在此另起一套。
8. 效能：先有量測再改演算法；不得為未量測的熱點重寫實作。
9. 跨專案契約：外掛目錄結構就是下載契約。只允許 `plugins/` 與 `languages/` 兩類；根目錄不得再增加會被舊版下載器當成類別的資料夾（包含 `docs/`）。執行期需要的檔案必須扁平放在外掛目錄直下。
10. 文件同步：外掛集合、下載契約與使用者可見行為變更時，`README.md` 與九份語言版必須同一變更對齊；`architecture.md` 在契約變更時更新。完成紀錄寄放在 Imervue，不在本倉庫建 `docs/`。

嚴重度：缺陷（必須修，未關閉即不得通過）、建議（應修，但無立即損害）、備註（記錄即可）。通過＝無未關閉缺陷；有條件通過＝無立即損害；需修正＝有未關閉缺陷。

## 審核紀錄

### 方法

讀取 `architecture.md`、`CLAUDE.md`、`progress.md` 前 80 行（檔案在第 8 行結束），以及 19 個官方外掛的進入點：`plugins/` 下 18 個與 `languages/spanish_translation/`。對套件內 `*.py`（不含 `__pycache__`）搜尋 `eval(`、`exec(`、`shell=True`、`pickle`、`yaml.load`、`verify=False`、裸 `except:`、`subprocess`。圈複雜度、函式長度、巢狀與參數個數以語法樹統計後，凡列入發現者皆再對照原始碼行號。未改其他檔、未執行測試、未啟動程式、未新增 `docs/`。

### 發現

#### CR-01 · 缺陷 · 多張上傳全部失敗仍以成功結束

- 位置：`plugins/cloud_share/cloud_share_plugin.py:236`
- 說明：`_UploadWorker.run` 在多張路徑時呼叫 `upload_batch`。`plugins/cloud_share/uploaders.py:140` 會把單張的 `UploadError`、`OSError`、`ValueError` 記成 `None` 並繼續。工人接著無條件 `done.emit(True, f"{succeeded}/{len(results)}")`（`:236-239`）。`_on_done`（`:197-207`）只在 `ok` 為假時顯示失敗並留下對話框；為真則顯示「已上傳」並 `accept()`。因此兩張以上全部失敗會變成成功提示（例如 `Uploaded 0/2`）並關閉對話框。單張路徑的例外才會 `emit(False, ...)`（`:233-234`、`:240-246`）。部分成功同樣走成功路徑，失敗的檔名不會回到介面。
- 依據：規範 3。發布結果被報成成功，使用者無法分辨圖片是否已離開本機。
- 狀態：未關閉
- 建議：`succeeded == 0` 時發出失敗；`0 < succeeded < len(results)` 時不要走與全部成功相同的關閉路徑，並讓介面能看出哪些檔案沒有上傳。單張行為可維持現狀。

#### CR-02 · 建議 · 九份 README 的外掛表漏列 GPU Develop

- 位置：`README.md:43`
- 說明：`plugins/gpu_develop/` 是第 19 個官方外掛之一（`plugins/gpu_develop/gpu_develop_plugin.py:62`，`plugin.json` 要求 API 3，依賴 `wgpu`，見 `gpu_develop_plugin.py:33`）。`README.md:18` 與 `:28` 已描述其行為，但外掛表在 `cloud_share`（`:43`）之後直接跳到 `npr_filters`（`:44`），沒有 `plugins/gpu_develop` 列，也沒有寫明需要 `wgpu`。同一缺口出現在 `README.zh-TW.md:43`、`README.zh-CN.md:43`、`README.ja.md:43`、`README.ko.md:43`、`README.es.md:43`、`README.fr.md:43`、`README.de.md:43`、`README.pt-BR.md:43`、`README.ru.md:43`。下載器依目錄列舉，不讀這張表，因此目前不會擋下載。`architecture.md:74` 已有對應說明。
- 依據：規範 10。外掛集合是使用者可見變更，九份 README 必須與程式同一內容。
- 狀態：未關閉
- 建議：在九份語言的外掛表於 `cloud_share` 與 `npr_filters` 之間補上 GPU Develop，並註明選用 `wgpu`、需要外掛 API 3。不要只改英文版。

#### CR-03 · 建議 · Cloud Share 的說明未包含已實作的 S3

- 位置：`plugins/cloud_share/cloud_share_plugin.py:52`
- 說明：`plugin_description` 只寫 WebDAV 或 Imgur。對話框已提供 S3 預簽 URL（`:97`），上傳實作在 `plugins/cloud_share/uploaders.py:109`。`README.md:43` 與另外八份 README 的同一列也只寫 WebDAV 或 Imgur。介面本身看得到第三個目標，因此不是功能缺失。
- 依據：規範 5、規範 10。
- 狀態：未關閉
- 建議：類別說明與九份 README 的該列補上預簽 S3（仍為僅 HTTPS）。密碼與預簽 URL 目前不寫入設定，這點應維持。

#### CR-04 · 建議 · safety_review 以長參數串傳選項，超過 7 個

- 位置：`plugins/safety_review/_workers.py:399`
- 說明：`_SubprocessBatchWorker.__init__`（`:399-407`）除 `self` 外有 18 個參數（7 個位置參數、11 個僅關鍵字參數）。同外掛還有 `_SubprocessSingleWorker.__init__`（`:320`，除 `self` 外 12 個）、`_detection.py:345` 的 `_process_single_image`（13 個）、`_runner.py:171` 的 `_process_one`（14 個）。皆未超過 80 行，但參數表已無法一眼對上呼叫端與 `_runner.py` 的命令列順序。
- 依據：規範 2（參數上限 7）。
- 狀態：未關閉
- 建議：把審查選項收成一個凍結的設定物件，工人、處理函式與子程序命令共用它，而不是繼續加參數。本次不要求為了達標去拆已經正確的演算法。

#### CR-05 · 建議 · 批次子程序入口的圈複雜度超過 15

- 位置：`plugins/safety_review/_runner.py:288`
- 說明：`_run_batch`（`:288-337`）本體約 50 行，未超過 80 行，巢狀也停在 4 以內。圈複雜度來自開頭的長度檢查、11 個預設值條件運算式（`:296-306`）、迴圈、單一圖片的 `except`，以及 `failed_dir` 分支，合計 16，超過 15。邏輯是命令列解譯加逐張處理，讀得懂，但預設值鏈與處理迴圈捆在一起。
- 依據：規範 2（圈複雜度 15）。
- 狀態：未關閉
- 建議：把引數預設值拆成純函式，`_run_batch` 只保留載入、迴圈與收尾。不要在未量測前改偵測或打碼演算法（規範 8）。

#### CR-06 · 建議 · 連通元件後備實作巢狀超過 4 層

- 位置：`plugins/object_splitter/_components.py:42`
- 說明：`_bfs_components`（`:35-57`）依序是 `for y`、`for x`、`if` 未標記前景、`while` 佇列、`for` 四鄰、`if` 界內且未標記，巢狀 6 層。函式約 23 行，結果與 SciPy 路徑對齊的意圖寫在模組說明裡。這是沒有 SciPy 時的後備，不是目前的主路徑。
- 依據：規範 2（巢狀上限 4）。
- 狀態：未關閉
- 建議：把「界內且為未標記前景」抽成區域函式，或讓掃描與染色分成兩段，使主迴圈不超過 4 層。沒有剖面數據前不要改寫成另一種演算法（規範 8）。

沒有其他已用行號確認的缺陷。未把譯文字串超過 120 字元、或僅差 1 個參數的建構式，另開發現。

### 總評

19 個官方外掛的進入點都宣告了 `plugin_class`，目錄維持 `plugins/` 與 `languages/` 兩類，執行期檔案扁平，沒有根目錄 `docs/`。危險呼叫搜尋沒有命中可利用的 `eval`、`exec`、`shell=True`、`pickle`、不安全 YAML 或關閉 TLS。子程序以引數列表啟動。唯一會誤導使用者的行為是 Cloud Share 的多張上傳：失敗仍走成功路徑。其餘是文件未跟上外掛集合，以及 `safety_review`、`object_splitter` 的可讀性上限。這些建議沒有立即損害，但不抵銷 CR-01。

### 已核對

- 進入點：`ai_background_remover.py`、`ai_colorize_plugin.py`、`ai_denoise_plugin.py`、`ai_motion_deblur_plugin.py`、`ai_object_remove_plugin.py`、`ai_outpaint_plugin.py`、`ai_portrait_relight_plugin.py`、`ai_smart_resize_plugin.py`、`ai_style_transfer_plugin.py`、`cloud_share_plugin.py`、`gpu_develop_plugin.py`、`npr_filters_plugin.py`、`object_splitter.py`、`pet_integrations_plugin.py`、`icon_converter_plugin.py`、`portrait_mode.py`、`safety_review.py`、`video_source_plugin.py`、`spanish_translation_plugin.py`。譯文表未逐條審。
- 熱點：`uploaders.py`、`webhook_server.py`、`twitch_chat_hook.py`（TLS 1.2、不使用明碼埠）、`_workers.py`、`_runner.py`、`_components.py`、`video_frames.py`、`_detection.py` 的單張處理。
- 安全搜尋：無 `shell=True`、`pickle`、`yaml.load`、`verify=False`、`eval(`、裸 `except:`。`exec(` 的命中是 Qt 對話框的 `.exec()` 或日誌字串。`subprocess.Popen` 僅出現在 `ai_background_remover.py:140`、`:210`，`object_splitter.py:116`，`safety_review/_workers.py:374`、`:487`；命令是列表，並傳入共用的 `_subprocess_kwargs`（標準輸入丟棄、UTF-8、Windows 不開主控台）。
- Webhook 綁定 `127.0.0.1`（`webhook_server.py:52`、`:274`），有本機 Origin 檢查與可選 Bearer。Cloud Share 拒絕非 `https`（`uploaders.py:29-34`）。WebDAV 密碼不寫入設定（`cloud_share_plugin.py:182-187`）。原始碼未寫死權杖或密碼。
- 契約：`progress.md` 的 Open 為空。`.gitignore` 已忽略 `__pycache__/`。未發現需被下載的巢狀程式目錄。Wide `except Exception` 多半有日誌，且註明工人執行緒必須回報；沒有把它們當成裸 `except`。
- 單行：超過 120 的主要是譯文與 `safety_review/_runner.py:4` 的用法字串（該行約 240 字元）。未因此另開缺陷。

### 未涵蓋

- 未執行測試，未啟動 Imervue，未做與 Imervue 工作樹的逐檔鏡像比對。
- 未逐行審完打碼、接縫裁切、著色與 GPU shader 的數值正確性，也未重跑效能數字。
- 未呼叫外部程式分析服務，未核對 `architecture.md` 開頭「Last verified: 2026-10-07」是否仍代表最新下載器行為。該檔正文已描述 GPU Develop。
- 子程序文字模式依賴 Imervue 的 `_subprocess_kwargs` 已設 `encoding`；此項只核對本倉庫的呼叫方式。

### 結論

需修正。未關閉發現 6 則：缺陷 1（CR-01）、建議 5（CR-02 至 CR-06）。CR-01 未關閉前不得視為通過。CR-02 至 CR-06 無立即損害，可與 CR-01 分開排程，但文件列應與外掛集合一起補齊。
