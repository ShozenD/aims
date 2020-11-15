# aims

## クロッピング関連

**skinImageCrop.ipynb**: クロッピングのノートブック

**skin_detect.py**: 肌探知アルゴリズム

**crop_sofue.py**: クロッピング用スクリプト

## 浅いkerasモデル

**shallow_model.py**: 浅いkerasの肌評価モデル

**image_sorting.py**: ぼやけている画像を除去するスクリプト（shallow_model.pyで使っている）

**multi_label.csv**: 皮膚科先生の評価のみ入ったラベル（shallow_model.pyで使っている）

## 皮膚癌API

**ISIC_makecsv.py**: ISICデータベースから画像idと皮膚癌の種類を取得するスクリプト（APIが遅いため、使っていない）

**ISIC_dl_image.py**: ISICデータベースから画像をダウンロードするスクリプト（APIが遅いため、使っていない）

**api.py**: ISICからデータベースを呼び出すためのスクリプト

## 皮膚癌データセット

**class_label.csv**: 皮膚癌が７つのクラスに分類されたラベル(0~6)
