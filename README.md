## About This branch

this is for testing/brainstorming branch for oncanvas inktool node pressure editing.
so this branch is not intended to pull request.

## About original MyPaint

About original Mypaint : オリジナルのMypaintについて
----
Original MyPaint is a simple drawing and painting program
that works well with Wacom-style graphics tablets.
Its main features are a highly configurable brush engine, speed,
and a fullscreen mode which allows artists to
fully immerse themselves in their work.

* Website: [mypaint.org](http://mypaint.org/)
* Twitter: [@MyPaintApp](https://twitter.com/MyPaintApp)
* Github:
  - [Development "master" branch](https://github.com/mypaint/mypaint)
  - [New issue tracker](https://github.com/mypaint/mypaint/issues)
* Other resources:
  - [Mailing list](https://mail.gna.org/listinfo/mypaint-discuss)
  - [Wiki](https://github.com/mypaint/mypaint/wiki)
  - [Forums](http://forum.intilinux.com/)
  - [Old bug tracker](http://gna.org/bugs/?group=mypaint)
    (patience please: we're migrating bugs across)
  - [Introductory docs for developers](https://github.com/mypaint/mypaint/wiki/Development)

MyPaint is written in Python, C++, and C.
It makes use of the GTK toolkit, version 3.x.
The source is maintained using [git](http://www.git-scm.com),
primarily on Github.

About this my custom build : このカスタムビルドについて
----
This is heavily customized version of the program "MyPaint".
Most customized codes are adhoc, still in testing stage by myself.
I also use this branch as my personal backup. (because private repository needs some money...)

これはかなりカスタマイズされたバージョンのMyPaintです。
ほとんどのカスタマイズコードはその場しのぎのもので、私自身によるテストを行っている状況です。
このブランチは個人的なバックアップ用途としても使っています。（プライベートリポジトリはお金がかかるので…）

Customized features : カスタム機能について
----

#### incremental version save : バージョンセーブ機能
This menu action will add version number to filename automatically and save it.
If filename has a version number already, such as 'foobar_002.ora',
this increment it and save as 'foobar_003.ora'

このメニュー機能はファイル名に自動でバージョン番号を付けてセーブします。

既にバージョン番号が付いている場合には、それに従います。
たとえばfoobar_002.oraだったら、foobar_003.oraとなります。

#### stabilizer : 強い手ブレ補正
When this stabilizer toggle button (placed right side of eraser) turned on,
strokes are much more heavily stabilized 
than set brush preset "Smooth" value to maximum.
This would be quite useful when I want to draw extreamly slow and weak stroke.

消しゴムブラシボタンの横に新設されたトグルボタンを押すことで、
通常の手ブレ補正を最大にするより強い手ブレ補正がかかります。

#### inktool nodes auto-culling : インクツールの自動間引き
With this feature,when inktool capture phase end,then auto-culling is executed.

インクツールのキャプチャフェイズが終了すると、自動で間引きが実行されます。

##### currently impremented auto culling algorithm
search the entire stroke (nodes) and find shortest section,and delete it.
and repeat it until nodes count equal to 'Autocull' scale widget count.

If that scale widget set to minimum(leftmost),it shows 'no' and autoculling disabled.

現在の実装では、全ノードをスキャンして最も短い区間を探し、その中間のノードを削除します。
ノードがAutocullスケールウィジェットで設定された個数になるまで、その削除が繰り返されます。

Autocullスケールを左端に設定すると、表示が「no」になり自動間引きは使用しない状態になります。
