## my_build branch : my custom build of MyPaint

This is heavily customized version of the program "MyPaint".

Most customized codes are adhoc, still in testing stage by myself.
I also use this branch as my personal backup. 

これはかなりカスタマイズされたバージョンのMyPaintです。
ほとんどのカスタマイズコードはその場しのぎのもので、私自身によるテストを行っている状況です。
このブランチは個人的なバックアップ用途としても使っています。

This branch is not intended to pull request. This is purely experimental build.

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

#### inktool - oncanvas node pressure editing : インクツールの制御点の筆圧をキャンバス上で修正可能
With holding down and drag the selected node, you can change it's pressure.

 * dragging up or left, pressure value is decreased.
 * dragging down or right, pressure value is increased.

this feature will affect to multiple selected nodes.

選択されたノード（複数可）をシフトキーを押しながらドラッグすると、それらのノードの筆圧を変更できます。

#### inktool - node capture sampling period factor setting : インクツールのノードキャプチャサンプリング期間の倍率設定
with 'Capturing period' scale,you can customize node capturing(sampling) period.
it is multiple factor to sampling time/length.

I got inspiration for this feature,so obsolute auto-culling.

インクツールのノードのサンプリング期間を倍率指定で広げることが出来ます。倍率を大きくすることで間隔が広がり、形は大まかになりますが、制御点の数が減って編集はしやすくなります。
場合に応じて使い分けてください。

#### inktool - Average points feature : インクツールの制御点を平均化してなだらかにする機能
'Average points' make stroke points less bumped,smooth curve.

制御点の位置を平均化することでなだらかにします。

#### inktool - Select multiple nodes : インクツールの制御点を複数選択可能に
you can select multiple nodes with holding CONTROL key and click nodes.
With this feature, you can move multiple nodes at once,simultaneously.

also 'Average points' / 'Delete points' / oncanvas node pressure editing , now works with this feature.

コントロールキーを押しながらノードをクリックすることで、制御点を複数選択できます。
平均化・削除・キャンバス状での筆圧修正の機能が、この複数選択状態に対応しています。

#### Per device mode change : デバイスごとのモード変更
A 'Device' referred to here is, for example, Pen tablet styls,or Pen tablet tail eraser,etc.

This feature record the painting 'mode' (not only brush) which used with a device.
And if a device is activated again,previously recorded mode automatically applied.

for example , this feature should be useful in such a case...
 1. drawing with inktool.finalize it.
 1. "oh I failed.I want to erase small part of that stroke!"
 1. flip pen stylus, to activate pen eraser.
 1. (without this feature,you will need to change to freehand mode)
 1. erase it
 1. flip pen stylus again.
 1. (without this feature,you need to change inktool again)
 
This feature may less useful who changes mode with keyboard toggle action,but at least for me this is very handy.

デバイスごとに使用しているモードを記録し、デバイスが戻ってきたら自動変更するようにしました。いままではブラシだけでした。

こんな場合に役立ちます。
 1. インクツールで線を描きます。固定します。
 1. 「おっと線を間違えた。ちょっとだけ消したい！」
 1. ペンスタイラスをひっくり返して消しゴムにします。
 1. （デバイスごとのモード変更なしだと、ここでフリーハンドモードにしなければなりません）
 1. 消します
 1. ペンスタイラスを再びひっくり返してペンモードにします。
 1. (忘れずにインクツールに戻す必要があります）

キーボードでトグルしている人にはあまり使わない機能かもですが、私には便利です。

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

