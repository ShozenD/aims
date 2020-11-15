# aims

## クロッピング関連

Preprocessing/**crop.py**: クロッピング用スクリプト

**Usage**
```
python crop.py -i (input_dir_path) -o (output_dir_path)
```
1. `input_dir_path`: マイクロスコープで撮影した画像が入ったディレクトリー
2. `output_dir_path`: クロップした画像を書き出すためのディレクトリー

## Shallow Keras Model

DeepEngine/**shallow_model.py**: 浅いkerasの肌評価モデル

DeepEngine/**image_sorting.py**: ぼやけている画像を除去するスクリプト（shallow_model.pyで使っている）

**multi_label.csv**: 皮膚科先生の評価のみ入ったラベル（shallow_model.pyで使っている）

## 皮膚癌API

ISIC/**ISIC_makecsv.py**: ISICデータベースから画像idと皮膚癌の種類を取得するスクリプト（APIが遅いため、使っていない）

ISIC/**ISIC_dl_image.py**: ISICデータベースから画像をダウンロードするスクリプト（APIが遅いため、使っていない）

ISIC/**api.py**: ISICからデータベースを呼び出すためのスクリプト

## 皮膚癌データセット
**class_label.csv**: 皮膚癌が７つのクラスに分類されたラベル(0~6)
