## my_build branch : my custom build of MyPaint


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
This is obsoluted.
廃止しました。

#### inktool node capture sampling period factor setting : インクツールのノードキャプチャサンプリング期間の倍率設定
with 'Capturing period' scale,you can customize node capturing(sampling) period.
it is multiple factor to sampling time/length.

I create this feature,so obsolute auto-culling.

#### inktool - Average points feature : インクツールの制御点を平均化してなだらかにする機能
'Average points' make stroke points less bumped,smooth curve.

#### Per device mode change : デバイスごとのモード変更
A 'Device' referred to here is, for example, Pen tablet styls,or Pen tablet tail eraser,etc.

This feature record the painting 'mode' (not only brush) which used with a device.
And if a device is activated again,previously recorded mode automatically applied.

for example , this feature should be useful in such a case...
1. drawing with inktool.finalize it.
1. oh I failed.I want to erase small part of that stroke!
1. flip pen stylus, to activate pen eraser.
1. (without this feature,you will need to change to freehand mode)
1. erase it
1. flip pen stylus again.
1. (without this feature,you need to change inktool again)




