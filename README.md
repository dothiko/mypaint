## my custom build of MyPaint

About original Mypaint
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

About my custom build
----
This is heavily customized version of the program "MyPaint",
which is explained above.
Most customized codes are adhoc, still in testing stage by myself.

Customized features
----

#### save incremental save
this menu action will add version number to filename automatically and save it.
if filename has a version number already, such as 'foobar_001.ora',
this increment it and save as 'foobar_002.ora'

#### stabilizer
when this stabilizer button (placed right side of eraser) turned on,
strokes are much more heavily stabilized 
than set brush preset "Smooth" value to maximum.
this would be quite useful when I want to draw extreamly slow and weak stroke.

#### inktool nodes auto-culling
when inktool capture phase end,auto-culling is executed,
to avoid manipulate too many nodes.
