## my_build branch : my custom build of MyPaint

This is heavily customized version of the program "MyPaint".
[![Translation Status](https://hosted.weblate.org/widgets/mypaint/mypaint/svg-badge.svg)](https://hosted.weblate.org/engage/mypaint/?utm_source=widget)
[![Travis Build Status](https://travis-ci.org/mypaint/mypaint.png?branch=master)](https://travis-ci.org/mypaint/mypaint)
[![Tea-CI Build Status](https://tea-ci.org/api/badges/mypaint/mypaint/status.svg)](https://tea-ci.org/mypaint/mypaint)
>>>>>>> upstream/master

Most customized codes are adhoc, still in testing stage by myself.
I also use this branch as my personal backup. 

これはかなりカスタマイズされたバージョンのMyPaintです。
ほとんどのカスタマイズコードはその場しのぎのもので、私自身によるテストを行っている状況です。
このブランチは個人的なバックアップ用途としても使っています。

This branch is *not* intended to pull request. This is *purely experimental* build.

Customized features : カスタム機能について
----
#### On-Canvas Size Change Mode : オンキャンバスのブラシサイズ変更

You can bind to modifier-key + mouse/stylus button to brush size change now.

it is configured from 'Preference' dialog -> Buttons Tab.then add the mode 'On-canvas Size Change',and bind modifier + button it.

As a default, Shift + Button1 is binded to linemode.You can change this into Brush size change.

With This feature,you can use Krita-like oncanvas brush size change on MyPaint!!

Kritaライクなキャンバス上でのブラシサイズ変更を可能にします。

#### multiple layer selection : レイヤの複数選択
From now, we can select multiple layers.
With this feature, you can ...

 * drag-drop layers into group
 * merge selected layers
 * delete selected layers
 * group selected layers
 * erase a active layer contents with opaque pixels of other selected layers.

Although, You can draw only currently **active** layer, which shown bold and underlined at layers panel.


#### incremental version save : バージョンセーブ機能
This menu action will add version number to filename automatically and save it.
If filename has a version number already, such as 'foobar_002.ora',
this increment it and save as 'foobar_003.ora'

このメニュー機能はファイル名に自動でバージョン番号を付けてセーブします。

既にバージョン番号が付いている場合には、それに従います。
たとえばfoobar_002.oraだったら、foobar_003.oraとなります。

#### project save (experimantal) : プロジェクトセーブ機能
This feature utilize autosave functionality,extreamly faster save your work as 'Project *Directory*',not an 
OpenRaster file.
you can use this from 'Files' -> 'Projects' submenu,

 * Open Project - open project directory previously saved (or extracted Openraster file directory)
 * Open Recent Project - Quick access to recently opened/saved projects
 * Save current project - overwrite current this is almost same as 'Save' menu
 * Save As Project - save current opened work as new project directory 
 * Save Project as New version - save project , and changed (to be overwritten) layers are backuped.
 * Revert Project - **DO NOT CLICK THIS. under construction.** but you can manually revert older layer / strokemap with file manager of your desktop environment.

**This code is extreamly experimental**,so you *might lost your work* when using this feature.

プロジェクトディレクトリに展開して保存し、なおかつそれを使い続ける機能です。
OpenRasterファイルをパッケージする手間がないため高速に保存できます。autosave機能を流用して作られています。

この機能はカスタム機能の中でも**極めて実験的**なもので、使用した場合何らかのバグにより作品が破壊されることもあり得ます。

 * Open Project - プロジェクトを開きます
 * Open Recent Project - 最近使ったプロジェクトを開きます
 * Save current project - 現在のプロジェクトを上書き保存
 * Save As Project - 新規プロジェクトとして保存
 * Save Project as New version - プロジェクトを上書き保存し、その際、上書きが発生するレイヤの画像をすべて退避します。
 * Revert Project - **作成中なのでクリックしないでください** しかし、ファイルマネージャを使って手動で戻すことも可能だと思われます。

#### stabilizer : 強い手ブレ補正
When this stabilizer toggle button (placed right side of eraser) turned on,
strokes are much more heavily stabilized 
than set brush preset "Smooth" value to maximum.
This would be quite useful when I want to draw extreamly slow and weak stroke.

消しゴムブラシボタンの横に新設されたトグルボタンを押すことで、
通常の手ブレ補正を最大にするより強い手ブレ補正がかかります。


#### inktool - oncanvas node pressure editing : インクツールの制御点の筆圧をキャンバス上で修正可能
With holding down and drag the selected node, you can change it's pressure.

 * dragging up or left, pressure value is decreased.
 * dragging down or right, pressure value is increased.

this feature will affect to multiple selected nodes.

選択されたノード（複数可）をシフトキーを押しながらドラッグすると、それらのノードの筆圧を変更できます。

#### inktool - now supports Button-binded actions at CAPTURE phase.

At only CAPTURE phase,inktool supports Button-binded actions now.
this is mainly for Oncanvas-size-change mode.

オンキャンバスでのブラシサイズ変更を念頭に、インクツールがキャプチャフェーズのみはボタンにバインドされたアクションを適用できるようになりました。

#### inktool - node capture sampling period factor setting : インクツールのノードキャプチャサンプリング期間の倍率設定
with 'Capturing period' scale,you can customize node capturing(sampling) period.
it is multiple factor to sampling time/length.

I got inspiration for this feature,so obsolute auto-culling.

インクツールのノードのサンプリング期間を倍率指定で広げることが出来ます。倍率を大きくすることで間隔が広がり、形は大まかになりますが、制御点の数が減って編集はしやすくなります。
場合に応じて使い分けてください。

#### inktool - Average points feature : インクツールの制御点を平均化してなだらかにする機能

 * 'Average Angle' make stroke points less bumped,smooth curve.this feature is multiple-selected-nodes aware.
 * 'Average Distance' make stroke points spaces even.this feature always affects to entire nodes.
 * 'Average Pressure' make nodes pressure smoother.this feature is multiple-selected-nodes aware.

 * 'Average Angle'は、制御点の角度を平均化することでなだらかにします。これは複数選択されているノードに限定することができます。一つしかノードが選ばれていない時は、全体に影響します。
 * 'Average Distance'は、制御点間の距離を均等にします。これは常にノード全体に影響します。
 * 'Average Pressure'は、制御点の筆圧を加重平均化します。

#### inktool - Select multiple nodes : インクツールの制御点を複数選択可能に
you can select multiple nodes with holding CONTROL key and click nodes.
With this feature, you can move multiple nodes at once,simultaneously.

also 'Average points' / 'Delete points' / oncanvas node pressure editing , now works with this feature.

コントロールキーを押しながらノードをクリックすることで、制御点を複数選択できます。
平均化・削除・キャンバス状での筆圧修正の機能が、この複数選択状態に対応しています。

#### Beziertool - Draw Cubic Bezier-sprine curve like inktool : 三次ベジェ曲線を描くためのベジェツール
The Cubic bezier-sprine tool,like 'inkscape' tool.

HOW TO OPERATION:(mostly same as inkscape)
 1. push & hold mouse left button where you want to place the first node.
 1. drag it,then the first control handle to adjust bezier curve created.
 1. when you feel enough,release mouse left button and end drag.
 1. hover mouse cursor to the place where you want to place the next node.
 1. push & hold mouse left button.
 1. and drag.you will decide the next control handle,and you will see the bezier curve.
 1. repeat this and create bezier-sprine curve.
 1. to end editing this curve,click ACCEPT button.

And, you can add node (divide bezier stroke intermidiate) with holding CTRL key and click the stroke.

futhermore,you can set nodes pressure at once by using Pressure Variation curve option widget and "Apply Variation" button (or "ApplyPressureVariationNodes" Action from keyboard)


インクモードと類似したベジェツールモードです。操作方法は
 1. マウス左ボタンをプッシュで最初の点を作成
 1. そのままドラッグで最初の制御ハンドルを伸ばす
 1. マウスボタンを離してドラッグ終了。
 1. 次に置きたい点の位置までホバーさせる
 1. 次の点の位置でマウス左ボタンをプッシュ
 1. そのままドラッグして制御ハンドルを伸ばす
 1. 納得行く曲線を作るまで繰り返す。
 1. ACCEPTボタン（チェックマーク）で完成。この時、少しだけ作業中より精度の高い滑らかな線が描かれる。
 
CTRLキーを押しながらストロークをクリックすることで、その部分に制御点を追加出来ます。

さらに、オプションプレゼンターのPressure Variationウィジェットと、Apply Variationボタン（もしくは ApplyPressureVariationNodes アクションをキーボードから発動）で、現在のストロークの筆圧を一度に設定できます！

#### Polygon fill tool - fill a region surrounded with a (curved) path. : ポリゴンフィルツール
Polygon fill tool, which enable to fill/erase/only fill current opaque pixel with curved polygon.

with holding shift+ctrl and click empty canvas, we can popup "round button palette" around cursor.
and with clicking that button , do fill(check icon)/erase(eraser icon)/erase outside(cut icon)/fill atop(plus icon) operation immidiately.


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

### Getting started

MyPaint has an associated library,
[libmypaint](https://github.com/mypaint/libmypaint),
which is distributed as a sister project on Github.

- libmypaint (>= 1.3.0-alpha.0)

There are several third-party dependencies too:

- scons (>= 2.1.0)
- pygobject
- gtk3 (>= 3.12)
- python (= 2.7) (OSX: python >= 2.7.4)
- swig
- numpy
- pycairo (>= 1.4)
- libpng
- lcms2
- libjson-c (>= 0.11, but the older "libjson" name at ~0.10 will work too)
- librsvg

Recommended: a pressure sensitive input device (graphic tablet)

### Build and Install

All systems differ.
The basic build documentation is divided by
broad class of operating system and software distribution.

* [README\_LINUX.md (chiefly Debian-based systems)](README_LINUX.md)
* [README\_WINDOWS.md (native WIN32/WIN64 using MSYS2)](README_WINDOWS.md)
* [README\_OSX.md (macports - needs review)](README_OSX.md)

### Contributing

The MyPaint project welcomes and encourages participation by everyone.
We want our community to be skilled and diverse,
and we want it to be a community that anybody can feel good about joining.
No matter who you are or what your background is, we welcome you.

Please note that MyPaint is released with a
[Contributor Code of Conduct](CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.

Please see the file [CONTRIBUTING.md](CONTRIBUTING.md)
for details of how you can begin contributing.

### Legal info

MyPaint is Free/Libre/Open Source software.
See [Licenses.md](Licenses.md) for a summary of its licensing.
A list of contributors can be found in the about dialog.

