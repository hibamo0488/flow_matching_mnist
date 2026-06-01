# Flow Matching MNISTデモ
私が執筆してるこちらのブログで紹介してる内容となります。

## 必要環境/想定環境
- VSCode
- WSL2 or Linux
- Docker
- GPU VRAM 8GB (推奨16GB) <br>
※学習せず推論のみならGPU無しでも動くかも. 未検証.

## セットアップ
1. docker compose build
2. docker compose up
3. コンテナ in
4. 左上Fileから"Open Folder" -> "/workspace"ディレクトリを選択
5. ターミナルで"uv sync"を実行

notebookを開き、カーネル設定する際に必要拡張機能入れるか聞いてくるので適当に入れておく.<br>

セットアップ完了

## ファイル
src/
- train.ipynb: モデルの学習<br>
- infer.ipynb: 学習済みモデルを用いてベクトル場推論 & MNIST画像生成<br>
- fm_animation.ipynb: ノイズ点が指定した分布へ移動していく様子を動画化<br>

※既に動画化(gif)したものは[こちら](./samples/sample_movie)<br>

src/model/
- resnet11.ipynb: ベクトル場を学習/推論するモデル(Resnet 11層)の定義<br>

## 補足
学習済みモデルのサンプルは./samples/sample_modelにあるため、モデルの動作確認するだけであれば./src/infer.ipynbを実行するだけで動作確認できます。<br>



