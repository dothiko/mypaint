
#include "fill_guard.hpp"
#include "common.hpp"
#include "fix15.hpp"
#include "fill.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <glib.h>
#include <mypaint-tiled-surface.h>

#ifdef HEAVY_DEBUG
#include <stdio.h>
#endif


#define CENTER_TILE_INDEX 4
#define MAX_CACHE_COUNT 9

#define GET_TILE_INDEX(x, y)  (((x) + 1) + (((y) + 1) * 3))

#define GET_TX_FROM_INDEX(tx, idx) ((tx) + (((idx) % 3) - 1))
#define GET_TY_FROM_INDEX(ty, idx) ((ty) + (((idx) / 3) - 1))

// defining the _Tile wrapper object surface-cache attribute.
#define ATTR_NAME_RGBA "rgba"

inline char* get_tile_pixel(PyArrayObject* tile, const int x, const int y)
{
    // XXX copyed from fill.cpp
    const unsigned int xstride = PyArray_STRIDE(tile, 1);
    const unsigned int ystride = PyArray_STRIDE(tile, 0);
    return (PyArray_BYTES((PyArrayObject*)tile)
            + (y * ystride)
            + (x * xstride));
}
/** 
 * @ Tilecache
 *
 *
 * @detail
 * Base class for Morphology operation.
 * This class does tile cache management, and basic pixel operation
 * over entire tile cache.
 * 
 * Pixel functor class is assigned as a parameter for _iterate_pixel method
 * of this class, and Pixel functor class might uses Kernel functor class
 * to put a bunch of pixel.
 * And, Kernel functor class uses Tilecache derived class to draw pixel
 * with crossing over boundary of each cached tiles.
 */

template <class T>
class Tilecache
{
    protected:
        // Cache array. they are initialized at 
        // init_cached_tiles() method, so no need to
        // zerofill them at constructor.
        PyObject* m_cache_tiles[MAX_CACHE_COUNT];
        PyObject* m_cache_keys[MAX_CACHE_COUNT];

        // Parameters for on-demand cache tile generation. 
        // these values are passed to Numpy-API.
        int m_NPY_TYPE;
        int m_dimension_cnt;
        
        // current center tile location, in tiledict.
        int m_tx; 
        int m_ty; 

        // - Tilecache management methods

        PyObject* _generate_cache_key(int tx, int ty)
        {
            PyObject* pytx = PyInt_FromLong(tx);
            PyObject* pyty = PyInt_FromLong(ty);
            PyObject* key = PyTuple_Pack(2, pytx, pyty);
            Py_DECREF(pytx);
            Py_DECREF(pyty);
            return key;
        }

        PyObject* _generate_tile()
        {
            npy_intp dims[] = {MYPAINT_TILE_SIZE, MYPAINT_TILE_SIZE, m_dimension_cnt};
            return PyArray_ZEROS(3, dims, m_NPY_TYPE, 0);
        }

        // Get target(to be put dilated pixels) tile from relative offsets.
        PyObject* _get_cached_tile(int otx, // otx, oty have relative values ranging from -1 to 1,
                                   int oty, // they are relative offsets from center tile.
                                   bool generate=true) 
        {
            return _get_cached_tile_from_index(GET_TILE_INDEX(otx, oty), generate);
        }

        PyObject* _get_cached_tile_from_index(int index, bool generate)
        {
            if (m_cache_tiles[index] == NULL) {
                if (generate == true) {
                    PyObject* dst_tile = _generate_tile();
                    m_cache_tiles[index] = dst_tile;
#ifdef HEAVY_DEBUG
                    assert(m_cache_keys[index] != NULL);
#endif
                    return dst_tile;
                }
            }
            return m_cache_tiles[index];
        }

        PyObject* _get_cached_key(int index)
        {
            return m_cache_keys[index];
        }

    public:

        Tilecache(int npy_type, int dimension_cnt)
        {
            m_NPY_TYPE = npy_type;
            m_dimension_cnt = dimension_cnt;
        }
        virtual ~Tilecache()
        {
        }
        
        // initialize/finalize cache tile
        // call this and initialize cache before dilate/erode called.
        void init_cached_tiles(PyObject* py_tiledict, int tx, int ty)
        {
            m_tx = tx;
            m_ty = ty;

            // fill the cache array with current dilated dict.
            for (int y=-1; y <= 1; y++) {
                for (int x=-1; x <= 1; x++) {
                    int idx = GET_TILE_INDEX(x, y);
                    PyObject* key = _generate_cache_key(tx+x, ty+y);
                    PyObject* dst_tile = PyDict_GetItem(py_tiledict, key);
                    if (dst_tile) {
                        Py_INCREF(dst_tile); 
                        // With PyDict_GetItem(), tile is just borrowed. 
                        // so incref here.
                        // Otherwise, some tile would be generated in 
                        // this module.
                        // Such 'new tile' would be 'increfed' with 
                        // calling PyDict_SetItem() at _finalize_dilated_tiles().
                        // On the other hand, A  Borrowed tile, it is already 
                        // in tile dictionary, so it would not get increfed 
                        // even PyDict_SetItem().
                        // And all items decrefed in _finalize_cached_tiles().
                        // Therefore we need incref it here.
                        //
                        // keys are generated every time cache is set up, 
                        // so no need to incref.
#ifdef HEAVY_DEBUG
                        assert(dst_tile->ob_refcnt >= 2);
#endif
                    }
                    // if tile does not exist, line below means 
                    // 'initialize with NULL'
                    m_cache_tiles[idx] = dst_tile; 
                    m_cache_keys[idx] = key;
                }
            }
        }
        
        // call this and finalize cache after dilate/erode called.
        unsigned int finalize_cached_tiles(PyObject* py_tiledict)
        {
            unsigned int updated_cnt = 0;
            PyObject* key;

            for (int idx = 0; idx < MAX_CACHE_COUNT; idx++){

                key = m_cache_keys[idx];
#ifdef HEAVY_DEBUG
                assert(key != NULL);
#endif
                // m_cache_tiles might not have actual tile.be careful.
                if (m_cache_tiles[idx] != NULL) {
                    PyObject* dst_tile = (PyObject*)m_cache_tiles[idx];
                    if (PyDict_SetItem(py_tiledict, key, dst_tile) == 0) {
                        updated_cnt++;
                    }
                    Py_DECREF(dst_tile);
                    m_cache_tiles[idx] = NULL;
#ifdef HEAVY_DEBUG
                    assert(dst_tile->ob_refcnt >= 1);
#endif
                }

#ifdef HEAVY_DEBUG
                assert(key->ob_refcnt >= 0);
#endif
                Py_DECREF(key); // key might not used after it is generated.
            
            }

            return updated_cnt;
        }

        // call this setup method before morphology operation starts.
        //void setup_morphology_params(int size, const T* source_pixel);

        // - Pixel manipulating methods

        T* get_pixel(const int cache_index,
                     const unsigned int x, 
                     const unsigned int y,
                     bool generate)
        {
            PyArrayObject* tile = (PyArrayObject*)_get_cached_tile_from_index(
                                                           cache_index, generate);
            if (tile) {
                return (T*)get_tile_pixel(tile, x, y);
            }
            return NULL;
        }

        // get pixel pointer over cached tiles linearly.
        // the origin is center tile(index 4) (0,0),
        // from minimum (-64, -64) to maximum (128, 128).
        //           
        // -64 <-    x    -> 128
        // -64    |0|1|2| 
        //   y    |3|4|5|   the top-left corner of index4 tile is origin (0,0)
        // 128    |6|7|8|
        //    
        // we can access pixels linearly within this range, 
        // without indicating tile cache index or tiledict key(tx, ty). 
        //
        // this method might generate new tile with _get_cached_tile(). 
        T* get_cached_pixel(int cx, int cy, bool generate)
        {
            int tile_index = CENTER_TILE_INDEX;
            
            // 'cache coordinate' is converted to tile local one.
            // and split it RELATIVE tiledict key offset(tx, ty).
            if (cx < 0) {
                tile_index--;
                cx += MYPAINT_TILE_SIZE;
            }
            else if (cx >= MYPAINT_TILE_SIZE) {
                tile_index++;
                cx -= MYPAINT_TILE_SIZE;
            }

            if (cy < 0) {
                tile_index -= 3;
                cy += MYPAINT_TILE_SIZE;
            }
            else if (cy >= MYPAINT_TILE_SIZE) {
                tile_index += 3;
                cy -= MYPAINT_TILE_SIZE;
            }
            
            return get_pixel(tile_index, cx, cy, generate);
        }

        
};




// fix15 dilater class
// This is for ordinary dilating of color pixel tile.
//
class _Dilater_fix15 : public Tilecache<fix15_short_t> {

    protected:
        
        inline void _put_pixel(int rx, int ry, const fix15_short_t* pixel)
        {
            fix15_short_t* dst_pixel = get_cached_pixel(rx, ry, true);

            if (dst_pixel[3] == 0) { // rejecting already written pixel 
                dst_pixel[0] = pixel[0];
                dst_pixel[1] = pixel[1];
                dst_pixel[2] = pixel[2];
                dst_pixel[3] = (fix15_short_t)fix15_one; 

                // XXX  When I tested this functionality , sometimes I met some strange
                // pixel glitches something like 'color noise'.
                // I named it 'alpha priority problem'.
                //
                // The problem is that if there is translucent alpha-blending 
                // pixel at filled tiles, it might produce ugly translucent glitch. 
                // Such translucent pixel in flood-filled pixels would come from 
                // tolerance parameter of floodfill_color_match(). 
                // With just dilating such pixel, there might be large pixel block of
                // translucent. 
                // They are translucent but have alpha value at least greater than zero,
                // so above rejecting(for speeding up) check line 'if(dst_pixel[3] == 0)'
                // reject other much more opaque pixels to write.
                // Thus, in some case, ugly translucent pixels glitch might be appeared.
                //
                // There are some options to avoid it,
                //  a) just use completely opaque pixel for dilation.(fast and easy)
                //  b) just overwrite when alpha value is greater.(a little slower) 
                //  c) mix pixels when more opaque pixel incoming.(much slower?)
                //
                //  I tested (b) and (c), but there still left annoying translucent
                //  pixels, under certain circumstance.
                //  so I decided to use (a), this produces better results most cases.

            }     
            /*
            else if (dst_pixel[3] < src_pixel[3]) {
                fix15_t alpha = (fix15_t)dst_pixel[3] + (fix15_t)pixel[3]; 
                alpha = fix15_mul(alpha, fix15_half);
                dst_pixel[0] = pixel[0];
                dst_pixel[1] = pixel[1];
                dst_pixel[2] = pixel[2];
                dst_pixel[3] = fix15_short_clamp(alpha);
            }        */
        }

        inline void _place_kernel(const int cx, const int cy,
                                  const int dx, const int dy, 
                                  const fix15_short_t* pixel)
        {
            _put_pixel(cx-dx, cy-dy, pixel);
            if (dx != 0 || dy != 0) {
                _put_pixel(cx+dx, cy-dy, pixel);
                _put_pixel(cx-dx, cy+dy, pixel);
                _put_pixel(cx+dx, cy+dy, pixel);
            }
        }

        void _put_square_kernel(const int cx, const int cy, const int size,
                                const fix15_short_t* pixel)
        {
            for (int dy = 0; dy <= size; dy++) {
                for (int dx = 0; dx <= size; dx++) {
                    _place_kernel(cx, cy, dx, dy, pixel);
                }
            }
        }

        void _put_diamond_kernel(const int cx, const int cy, const int size,
                                 const fix15_short_t* pixel)
        {
            for (int dy = 0; dy <= size; dy++) {
                for (int dx = 0; dx <= size-dy; dx++) {
                    _place_kernel(cx, cy, dx, dy, pixel);
                }
            }
        }

    public:

        static const int SQUARE_KERNEL = 0;
        static const int DIAMOND_KERNEL = 1;

        _Dilater_fix15() : Tilecache(NPY_UINT16, 4)
        {
        }

        int dilate(PyObject* py_filled_tile, // the filled src tile. 
                   int tx, int ty,  // the position of py_filled_tile
                   const fix15_short_t* fill_pixel,
                   int size,
                   int kernel_type  // 0 for square kernel, 1 for diamond kernel
                   )
        {
#ifdef HEAVY_DEBUG
            assert(size <= MYPAINT_TILE_SIZE / 2 - 1);
#endif

            int dilated_cnt = 0;

            for (int y=0; y < MYPAINT_TILE_SIZE; y++) {
                for (int x=0; x < MYPAINT_TILE_SIZE; x++) {
                    fix15_short_t* cur_pixel = 
                        (fix15_short_t*)get_tile_pixel(
                                            (PyArrayObject*)py_filled_tile, 
                                            x, y);
                    if (cur_pixel[3] != 0)
                    {
                        switch (kernel_type) {
                            case DIAMOND_KERNEL:
                                _put_diamond_kernel(x, y, size, fill_pixel);
                                break;

                            case SQUARE_KERNEL:
                            default:
                                _put_square_kernel(x, y, size, fill_pixel);
                                break;
                        }
                        dilated_cnt++;
                    }
                }
            }
            
            return dilated_cnt;
        }

};

// _GapFiller : Specialized class for detecting contour, to fill-gap.
// This is for 8bit,flag_based erode/dilate operation.
//
// in new flood_fill(), we use tiles of 8bit bitflag status to figure out 
// original contour, not-filled pixels area(EXIST_MASK) , 
// dilated area from original contour(DILATED_MASK) and
// the result, eroded area from dilated area(ERODED_MASK).

class _GapFiller: public Tilecache<char> {

    protected:
        // pixel status information flag.
        static const char EXIST_MASK = 0x01;
        static const char DILATED_MASK = 0x02; // This means the pixel is just dilated pixel,
                                               // not sure original source pixel.
                                               // it might be empty pixel in source tile.
        static const char ERODED_MASK = 0x04;
        static const char PROCESSED_MASK = 0x08; // This means 'This pixel has original source
                                                    // contour pixel, and dilated'.


        inline void _put_flag(const int cx, const int cy, const char flag)
        {
            char* dst_pixel = get_cached_pixel(cx, cy, true);
            *dst_pixel |= flag;
        }

        inline void _place_dilate_kernel(const int cx, const int cy,
                                  const int dx, const int dy, 
                                  const char flag)
        {
            _put_flag(cx-dx, cy-dy, flag);
            if (dx != 0 || dy != 0) {
                _put_flag(cx+dx, cy-dy, flag);
                _put_flag(cx-dx, cy+dy, flag);
                _put_flag(cx+dx, cy+dy, flag);
            }
        }

        void _put_dilate_kernel(const int cx, const int cy, const int size,
                                const char flag)
        {
            // dilate kernel in this class is always square.
            for (int dy = 0; dy <= size; dy++) {
                for (int dx = 0; dx <= size; dx++) {
                    _place_dilate_kernel(cx, cy, dx, dy, flag);
                }
            }
        }


        inline bool _check_erode_single_pixels(const int cx, const int cy)
        {
            char* dst_pixel;
            dst_pixel = get_cached_pixel(cx, cy, false);
            if (dst_pixel == NULL || 
                    (*dst_pixel & DILATED_MASK) == 0) {
                return false;
            }
            return true;
        }
        inline bool _check_erode_pixels(const int cx, const int cy,
                                        const int dx, const int dy) 
        {
            if ( ! _check_erode_single_pixels(cx - dx, cy - dy))
                    return false;

            if (dx != 0 || dy != 0) {
                if ( ! _check_erode_single_pixels(cx + dx, cy - dy))
                        return false;
                else if ( ! _check_erode_single_pixels(cx - dx, cy + dy))
                        return false;
                else if ( ! _check_erode_single_pixels(cx + dx, cy + dy))
                        return false;
            }

            return true;
        }

        bool _put_erode_kernel(const int cx, const int cy, const int size,
                               const char flag)
        {
            // erode kernel in this class is always diamond.
            for (int dy = 0; dy <= size; dy++) {
                for (int dx = 0; dx <= size-dy; dx++) {
                    if ( ! _check_erode_pixels(cx, cy, dx, dy))
                        return false;
                }
            }
            _put_flag(cx, cy, ERODED_MASK);
            return true;
        }


        // - special information methods
        //
        // Special informations recorded into a tile with
        // setting INFO_MASK bitflag to paticular pixel.
        // And they have various means,according to its location. 
        
        // tile status information flags.
        // This flag is only set to paticular location of tile pixels.
        static const char TILE_INFO_MASK = 0x80;

        static const char DILATED_TILE_MASK = 0x01;
        static const char ERODED_TILE_MASK = 0x02;
        static const char VOID_TILE_MASK = 0x80;  // invalid(does not exist) tile.

        // Maximum count of Tile info flags.
        // VOID_TILE_MASK is not included to this number.
        static const int TILE_INFO_MAX = 2; 
        
        char _get_tile_info(const int tile_index)
        {
            char retflag = 0;
            char flag = 1;
            char* pixel;
#ifdef HEAVY_DEBUG
            assert(tile_index >= 0);
            assert(tile_index < 9);
#endif
            for(int i=0; i < TILE_INFO_MAX; i++) {
                pixel = get_pixel(tile_index, 0, i, false);
                if (pixel == NULL)
                    return VOID_TILE_MASK;

                if (*pixel & TILE_INFO_MASK)
                    retflag |= flag;
                flag = flag << 1;
            }
            return retflag;
        }

        void _set_tile_info(const int tile_index, char flag)
        {
#ifdef HEAVY_DEBUG
            assert(tile_index >= 0);
            assert(tile_index < 9);
#endif
            char* pixel;
            for(int i=0; i < TILE_INFO_MAX && flag != 0; i++) {
                pixel = get_pixel(tile_index, 0, i, false);
                if (pixel == NULL)
                    return; // this tile does not have entity.

                if (flag & 0x01)
                    *pixel |= TILE_INFO_MASK;
                flag = flag >> 1;
            }
        }



        // - contour detecting morphology methods
        //
        // Dilate existing 9 status tiles, to ensure center status tile can get complete dilation.
        // With this dilation, maximum 9+16 = 25 state tiles might be generated.
        // But primary 9 tiles marked as 'dilation executed' and reused,
        // Therefore not so many processing time is consumed.
        void _dilate_contour(int gap_radius)    // growing size from center pixel.
        {
            for (int y = -gap_radius; y < gap_radius + MYPAINT_TILE_SIZE; y++) {
                for (int x = -gap_radius; x < gap_radius + MYPAINT_TILE_SIZE; x++) {
                    char* pixel = get_cached_pixel(x, y, false);
                    if (pixel != NULL  
                        && (*pixel & EXIST_MASK) != 0  
                        && (*pixel & PROCESSED_MASK) == 0) {
                            _put_dilate_kernel(x, y, gap_radius, DILATED_MASK);
                            *pixel |= PROCESSED_MASK;
                    }
                }
            }

            _set_tile_info(CENTER_TILE_INDEX, DILATED_TILE_MASK);
        }

        void _erode_contour(int gap_radius)    // growing size from center pixel.
        {
            // Only center tile should be eroded.
            char tile_info = _get_tile_info(CENTER_TILE_INDEX);
#ifdef HEAVY_DEBUG
            assert((tile_info & VOID_TILE_MASK) == 0);
#endif

            if ((tile_info & ERODED_TILE_MASK) != 0)
                return;
            
            for (int y = -gap_radius; y < gap_radius + MYPAINT_TILE_SIZE; y++) {
                for (int x = -gap_radius; x < gap_radius + MYPAINT_TILE_SIZE; x++) {
                    char* pixel = get_cached_pixel(x, y, false);
                    if (pixel != NULL && (*pixel & DILATED_MASK) != 0) {
                        _put_erode_kernel(x, y, gap_radius, ERODED_MASK);
                      //*pixel &= erase_flag;
                    }
                }
            }

            _set_tile_info(CENTER_TILE_INDEX, ERODED_TILE_MASK);
        }

        // Convert(and initialize) color pixel tile into 8bit status tile.
        // status tile updated with 'EXISTED' flag, but not cleared.
        // So exisitng status tile can keep dilated/eroded flags,
        // which is set another call of this function.
        PyObject* _convert_state_tile(PyObject* py_statedict, // 8bit state dict
                                      PyArrayObject* py_surf_tile, // fix15 color tile from surface
                                      const fix15_short_t* targ_pixel,
                                      const fix15_t tolerance,
                                      PyObject* key) // a tuple of tile location
        {
            

            PyObject* state_tile = PyDict_GetItem(py_statedict, key);

            if (state_tile == NULL) {
                state_tile = _generate_tile();
                PyDict_SetItem(py_statedict, key, state_tile);
                // No need to decref tile & key here, 
                // it should be done at _finalize_cached_tiles()
            }

            for (int py = 0; py < MYPAINT_TILE_SIZE; py++) {
                for (int px = 0; px < MYPAINT_TILE_SIZE; px++) {
                    fix15_short_t* cur_pixel = 
                        (fix15_short_t*)get_tile_pixel(py_surf_tile, px, py);
                    if (floodfill_color_match(cur_pixel, 
                                              targ_pixel, 
                                              tolerance) == 0) {
                        char* state_pixel = 
                            get_tile_pixel((PyArrayObject*)state_tile, px, py);
                        *state_pixel |= EXIST_MASK;
                    }
                }
            }
            
            return state_tile;
        }

        // convert color tiles around (tx, ty) into state flag tiles.
        // also, this setup function sets newly generated status tile into cache.
        void _setup_state_tiles(PyObject* py_statedict, // 8bit state tiles dict
                                PyObject* py_surfdict, // source surface tiles dict
                                const fix15_short_t* targ_pixel,
                                const fix15_t tolerance)
        {
            // py_surfdict is surface tile dictionary, it is not cached in this class.
            // this class have cache array of 'state tiles', not surface one.
            // so,extract source color tiles with python API.
#ifdef HEAVY_DEBUG            
            assert(py_surfdict != NULL);
            assert(py_surfdict != Py_None);
#endif

            for (int i=0; i < 9; i++) {

                if (m_cache_tiles[i] == NULL) {
                    PyObject* key = _get_cached_key(i);
                    PyObject* surf_tile = PyDict_GetItem(py_surfdict, key);
                    if (surf_tile != NULL) {
                        Py_INCREF(surf_tile);

                        // The surf_tile might not be tile, but _Tile wrapper object.
                        // If so, extract cached tile from it.
                        // Fetching tile into cache attribute(rgba) of 
                        // _Tile wrapper object is done at tiledsurface.py
                        int is_tile_obj = PyObject_HasAttrString(surf_tile, ATTR_NAME_RGBA);
                        if (is_tile_obj != 0) {
                            Py_DECREF(surf_tile);// This surf_tile is _Tile

                            surf_tile = PyObject_GetAttrString(surf_tile,
                                                               ATTR_NAME_RGBA);
                            Py_INCREF(surf_tile);// This surf_tile is PyArrayObject
#ifdef HEAVY_DEBUG            
                            assert(surf_tile->ob_refcnt >= 2);
#endif
                        }

                        PyObject* state_tile = 
                            _convert_state_tile(py_statedict, 
                                                (PyArrayObject*)surf_tile, 
                                                targ_pixel, 
                                                tolerance,
                                                key);
                        Py_DECREF(surf_tile);
                        m_cache_tiles[i] = state_tile;
                    }
                }
            }


#ifdef HEAVY_DEBUG            
            assert(m_cache_tiles[CENTER_TILE_INDEX] != NULL);
#endif
        }

    public:

        _GapFiller() : Tilecache(NPY_UINT8, 1)
        {
        }

        void fill_gap(PyObject* py_statedict, // 8bit state tiles dict
                      PyObject* py_surfdict, // source surface tiles dict
                      int tx, int ty, // current tile location
                      const fix15_short_t* targ_pixel,// target pixel for conversion
                                                      // from color tile to status tile.
                      fix15_t tolerance,
                      int gap_radius)
        {
            init_cached_tiles(py_statedict, tx, ty); 

            _setup_state_tiles(py_statedict, 
                               py_surfdict, 
                               targ_pixel,
                               tolerance);
        
            // detecting contour with dilate & erode.
            _dilate_contour(gap_radius);
            
            _erode_contour(gap_radius);

            finalize_cached_tiles(py_statedict); 
        }

#ifdef HEAVY_DEBUG
        // XXX TEST CODES
#endif
        
};

// - Python Interface functions.
//

/** gap_fill:
 *
 * @py_statedict: a Python dictinary, which stores 'state tiles'
 * @py_pre_filled_tile: a Numpy.array object of color tile.
 * @tx, ty: the position of py_pre_filled_tile, in tile coordinate.
 * @targ_r, targ_g, targ_b, targ_a: premult target pixel color
 * @tol: tolerance,[0.0 - 1.0] same as tile_flood_fill().
 * @gap_radius: the filling gap radius.
 * returns: Nothing. returning PyNone always.
 *
 * extract contour into state tiles, 
 * and that state tile is stored into py_statedict, with key of (tx, ty).
 * And later, this state tile used in flood_fill function, to detect
 * ignorable gaps.
 */

PyObject* 
fill_gap(PyObject* py_statedict, // the tiledict for status tiles.
         PyObject* py_surfdict, //  source surface tile dict.
         int tx, int ty,  // the position of py_filled_tile
         int targ_r, int targ_g, int targ_b, int targ_a, //premult target pixel color
         double tol,   // pixel tolerance of filled area.
         int gap_radius) // overflow-preventing closable gap size.
{
#ifdef HEAVY_DEBUG            
    assert(py_statedict != NULL);
    assert(py_surfdict != NULL);
    assert(0.0 <= tol);
    assert(tol <= 1.0);
    assert(gap_radius <= MYPAINT_TILE_SIZE / 2 - 1);
#endif
    // actually, this function is wrapper.
    
    // XXX Morphology object defined as static. 
    // Because this function called each time before a tile flood-filled,
    // I think constructing/destructing cost would not be ignored.
    // Otherwise, we can use something start/end wrapper function and
    // use PyCapsule object...
    static _GapFiller m;


    fix15_short_t targ_pixel[4] = {(fix15_short_t)targ_r,
                                   (fix15_short_t)targ_g,
                                   (fix15_short_t)targ_b,
                                   (fix15_short_t)targ_a};

    // XXX Copied from fill.cpp tile_flood_fill()
    const fix15_t tolerance = (fix15_t)(  MIN(1.0, MAX(0.0, tol))
                                        * fix15_one);

    // limit gap size.
    gap_radius %= (MYPAINT_TILE_SIZE / 2 - 1);

    m.fill_gap(py_statedict, py_surfdict,
               tx, ty, 
               (const fix15_short_t*)targ_pixel,
               tolerance,
               gap_radius);

    Py_RETURN_NONE;// DO NOT FORGET THIS!
      
}

/** dilate_filled_tile:
 *
 * @py_dilated: a Python dictinary, which stores 'dilated color tiles'
 * @py_pre_filled_tile: a Numpy.array object of color tile.
 * @tx, ty: the position of py_pre_filled_tile, in tile coordinate.
 * @grow_size: the growing size of filled area, in pixel.
 * @kernel_type: the shape of dilation. if this is 0
 *               (==Tilecache<T>::SQUARE_KERNEL),
 *               pixels dilated into square shape.
 *               if this is 1 (==Tilecache<T>::DIAMOND_KERNEL), 
 *               pixels dilated into diamond like shape.
 * returns: updated tile counts.
 *
 * dilate flood-filled tile, and store newly generated tiles into 
 * py_dilated dictionary.
 * Usually, dilating operation would mutiple new tiles from one 
 * flood-filled tile.
 * They are stored into py_dilate, and composited other flood-filled 
 * dilation images over and over again.
 * And when the floodfill loop ended,every dilated tiles are composited 
 * to flood-filled tiles.
 * (This composite processing is done in python script)
 * Thus, we can get dilated floodfill image.
 */

// dilate color filled tile. postprocess of flood_fill.
PyObject*
dilate_filled_tile(PyObject* py_dilated, // the tiledict for dilated tiles.
                   PyObject* py_filled_tile, // the filled src tile. 
                   int tx, int ty,  // the position of py_filled_tile
                   double fill_r, double fill_g, double fill_b, //premult pixel color
                   int dilate_size,    // growing size from center pixel.
                   int kernel_type) // 0 for square kernel, 1 for diamond kernel
{
    
    // Morphology object defined as static. 
    // Because this function called each time before a tile flood-filled,
    // so constructing/destructing cost would not be ignored.
    static _Dilater_fix15 d;

#ifdef HEAVY_DEBUG            
    assert(py_dilated != NULL);
    assert(py_filled_tile != NULL);
    assert(dilate_size <= MYPAINT_TILE_SIZE / 2);
    assert(kernel_type == _Dilater_fix15::SQUARE_KERNEL || 
           kernel_type == _Dilater_fix15::DIAMOND_KERNEL);
#endif

    
    // Actually alpha value is not used currently.
    // for future use.
    double alpha=(double)fix15_one;
    fix15_short_t fill_pixel[3] = {(fix15_short_clamp)(fill_r * alpha),
                                   (fix15_short_clamp)(fill_g * alpha),
                                   (fix15_short_clamp)(fill_b * alpha)};
    
    // _Dilater_fix15 class is specialized dilating filled pixels,
    // and uses 'current pixel' for dilation, not fixed/assigned pixel.
    // Therefore this class does not need internal pixel member,
    // dummy pixel is passed to setup_morphology_params().
    
    // limit dilation size.
    dilate_size %= (MYPAINT_TILE_SIZE / 2 - 1);

    d.init_cached_tiles(py_dilated, tx, ty);
    d.dilate(py_filled_tile, tx, ty, fill_pixel, dilate_size, kernel_type);
    d.finalize_cached_tiles(py_dilated);

    Py_RETURN_NONE;
}

#ifdef HEAVY_DEBUG
// XXX TEST CODES
#endif
