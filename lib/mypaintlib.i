%module mypaintlib

%{
#include "mypaintlib.hpp"
%}

%include "common.hpp"

%include "std_vector.i"

namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
}

typedef struct { int x, y, w, h; } Rect;

%include "surface.hpp"
%include "brush.hpp"
%include "mapping.hpp"

%include "python_brush.hpp"
%include "tiledsurface.hpp"

%include "pixops.hpp"
%include "colorring.hpp"
%include "colorchanger_wash.hpp"
%include "colorchanger_crossed_bowl.hpp"
%include "fastpng.hpp"

%ignore  floodfill_color_match;
%include "fill.hpp"

%include "brushsettings.hpp"
%include "gdkpixbuf2numpy.hpp"

%include "fill_guard.hpp"

%init %{
import_array();
%}

