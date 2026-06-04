# Student Photo Cropper

生徒の顔写真を、ローカルPC上で 3cm x 4cm のJPEGへ一括トリミングします。

既定値:

- 物理サイズ: 3cm x 4cm
- DPI: 300dpi
- ピクセル: 354 x 472px
- 容量: 100KB以下
- 顔検出: OpenCV YuNet

## セットアップ

```bash
cd /Users/yuki/student-photo-cropper
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## 使い方

1. `input/` に元写真を入れます。
2. 次を実行します。

```bash
cd /Users/yuki/student-photo-cropper
.venv/bin/python crop_student_photos.py
```

別のフォルダを指定する場合:

```bash
.venv/bin/python crop_student_photos.py --input /path/to/photos
```

サブフォルダもまとめて処理する場合:

```bash
.venv/bin/python crop_student_photos.py --input /path/to/photos --recursive
```

`--recursive` では出力先にもフォルダ構造を維持します。

例:

```text
photos/
  1A/1101.jpg
  1B/1201.jpg

output_ok/
  1A/1101.jpg
  1B/1201.jpg
```

出力:

- `output_ok/`: 自動トリミング成功画像
- `output_review/`: 顔検出失敗・複数顔・切り抜き範囲が危険な画像
- `preview_sheet/preview.jpg`: 成功分のBefore/After一覧
- `report.csv`: 処理結果

## 調整例

顔を少し小さめにして肩を多く入れる:

```bash
.venv/bin/python crop_student_photos.py --face-height-ratio 0.38
```

顔を大きめにする:

```bash
.venv/bin/python crop_student_photos.py --face-height-ratio 0.46
```

上の余白を増やす:

```bash
.venv/bin/python crop_student_photos.py --headroom-ratio 0.65
```

600 x 800pxで作りたい場合:

```bash
.venv/bin/python crop_student_photos.py --dpi 508
```

3cm x 4cmで508dpiにすると、約600 x 800pxになります。

500KB以下の高画質版を作る場合:

```bash
.venv/bin/python crop_student_photos.py --recursive \
  --output-ok output_high_500kb \
  --output-review output_high_500kb_review \
  --preview preview_sheet/high_500kb_preview.jpg \
  --report report_high_500kb.csv \
  --dpi 900 \
  --max-kb 500
```

3cm x 4cmで900dpiにすると、1063 x 1417pxになります。

## 注意

- 元画像は上書きしません。
- 生徒写真を外部サービスへ送信しません。
- `output_review/` は必ず人間が確認してください。
- `input/` や `output_*` は `.gitignore` で除外しています。生徒写真をGitHubへ公開しないでください。
