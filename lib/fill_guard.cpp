
#include "fill_guard.hpp"
#include "common.hpp"
#include "fix15.hpp"
#include "fill.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <glib.h>
#include <mypaint-tiled-surface.h>
#include <math.h>

#define CENTER_TILE_INDEX 4
#define ABOVE_TILE_INDEX 1
#define BELOW_TILE_INDEX 7
#define LEFT_TILE_INDEX 3
#define RIGHT_TILE_INDEX 5
#define MAX_CACHE_COUNT 9

#define GET_TILE_INDEX(x, y)  (((x) + 1) + (((y) + 1) * 3))

#define GET_TX_FROM_INDEX(tx, idx) ((tx) + (((idx) % 3) - 1))
#define GET_TY_FROM_INDEX(ty, idx) ((ty) + (((idx) / 3) - 1))

// the _Tile wrapper object surface-cache attribute name.
#define ATTR_NAME_RGBA "rgba"

// Maximum operation size. i.e. 'gap-radius'
#define MAX_OPERATION_SIZE ((MYPAINT_TILE_SIZE / 2) - 1)

inline char* get_tile_pixel(PyArrayObject* tile, const int x, const int y)
{
    // XXX almost copyed from fill.cpp::_floodfill_getpixel()
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
 * Base Template class for tile cache operation and basic pixel operation
 * over entire tile cache.
 * 
 * This class has 3x3 tile cache, to ensure morphology operation is exactly
 * done even at the border of the tile.
 * dilating operation would affect surrounding tiles over its border,
 * also, eroding operation would refer surrounding tiles.
 */

template <class T>
class Tilecache
{
    protected:
        // Cache array. They are initialized at 
        // init_cached_tiles() method, so no need to
        // zerofill them at constructor.
        PyObject* m_cache_tiles[MAX_CACHE_COUNT];
        PyObject* m_cache_keys[MAX_CACHE_COUNT];

        // Parameters for on-demand cache tile generation. 
        // These values are passed to Numpy-API.
        int m_NPY_TYPE;
        int m_dimension_cnt;
        
        // Current center tile location, in tiledict.
        int m_tx; 
        int m_ty; 

        // --- Tilecache management methods

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

        // Virtual handler method: used from _search_kernel
        virtual bool is_match_pixel(const int cx, const int cy, 
                                    const T* target_pixel) = 0;

        // Utility method: check pixel existance inside circle-shaped kernel
        // for morphology operation.
        //
        // For morphology operation, we can use same looping logic to search pixels
        // but wanted result is different for each operation.
        // For dilation, we need 
        // 'whether is there any valid pixel inside the kernel or not '.
        // On the other hand, for erosion, we need 
        // 'whether is the entire kernel filled with valid pixels or not'.
        // So use 'target_result' parameter to share codes between
        // these different operations.
        inline bool _search_kernel(const int cx, const int cy, 
                                   const int size,
                                   const T* target_pixel,
                                   bool target_result)
        {
            // dilate kernel in this class is always square.
            int cw;
            double rad;
            bool result;

            for (int dy = size-1; dy >= 0; dy--) {
                rad = acos((double)dy / (double)size);
                cw = (int)(sin(rad) * (double)size);
                for (int dx = 0; dx < cw; dx++) {
                    result = is_match_pixel(cx + dx, cy - dy, target_pixel);
                    if (result == target_result) 
                        return true;

                    if (dx != 0 || dy != 0) {
                        result = is_match_pixel(cx + dx, cy + dy, target_pixel);
                        if (result == target_result) 
                            return true;

                        result = is_match_pixel(cx - dx, cy - dy, target_pixel);
                        if (result == target_result) 
                            return true;

                        result = is_match_pixel(cx - dx, cy + dy, target_pixel);
                        if (result == target_result) 
                            return true;
                    }
                }
            }

            return false;
        }

    public:

        Tilecache(int npy_type, int dimension_cnt)
        {
            // Tile type and dimension are used in _generate_tile(),
            // when a new tile requested.
            // So only setup them, in constructor.
            m_NPY_TYPE = npy_type;
            m_dimension_cnt = dimension_cnt;
        }
        virtual ~Tilecache()
        {
        }
        
        // Initialize/finalize cache tile
        // Call this and initialize cache before dilate/erode called.
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
                        // On the other hand, key object is always generated
                        // above line, so does not need incref.
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
        
        // Call this and finalize cache before exiting interface function.
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
                assert(key->ob_refcnt >= 1);
#endif
                Py_DECREF(key); 
            
            }

            return updated_cnt;
        }

        // --- Pixel manipulating methods

        T* get_pixel(const int cache_index,
                     const unsigned int x, 
                     const unsigned int y,
                     bool generate)
        {
            PyArrayObject* tile = (PyArrayObject*)_get_cached_tile_from_index(
                                                    cache_index, generate);
            if (tile) 
                return (T*)get_tile_pixel(tile, x, y);
            
            return NULL;
        }

        // Get pixel pointer over cached tiles linearly.
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


// Morphology operation class for fix15_short_t pixels.
//
// CAUTION:
//   Cached tile is different for each morphology operation.
//   When doing dilation, we set write TARGET tiles into tile-cache.
//   and need only one tile for read-only reference source(flood-filled) tile.
//
//   When doing erosion, we set read-only SOURCE tiles into tile-cache.
//   and need only one tile for write target.
//
class _Dilation_fix15 : public Tilecache<fix15_short_t> {

    protected:

        virtual bool is_match_pixel(const int cx, const int cy, 
                                    const fix15_short_t* target_pixel) 
        {
            // target_pixel is ignored.
            //
            // And, this class only refer the currently targetted tile,
            // so coordinates which are negative or exceeding tile size
            // should be just ignored.
            if (cx < 0 || cy < 0 
                || cx >= MYPAINT_TILE_SIZE 
                || cy >= MYPAINT_TILE_SIZE) {
               return false;
            } 

            fix15_short_t* cur_pixel = 
                (fix15_short_t*)get_tile_pixel(
                                    m_target_tile,
                                    cx, cy);
            if (cur_pixel[3] != 0)
                return true;
            return false;
        }
        
        inline void _put_pixel(int cx, int cy, const fix15_short_t* pixel)
        {
            fix15_short_t* dst_pixel = get_cached_pixel(cx, cy, true);

            if (dst_pixel[3] == 0) { // rejecting already dilated pixel 
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

        PyArrayObject* m_target_tile;

    public:
        _Dilation_fix15() : Tilecache(NPY_UINT16, 4)
        {
        }

        int dilate(PyObject* py_filled_tile, // the filled src tile. 
                   int tx, int ty,  // the position of py_filled_tile
                   const fix15_short_t* fill_pixel,
                   int size)
        {
            int dilated_cnt = 0;
            m_target_tile = (PyArrayObject*)py_filled_tile;

            for (int cy=-size; cy < MYPAINT_TILE_SIZE+size; cy++) {
                for (int cx=-size; cx < MYPAINT_TILE_SIZE+size; cx++) {
                    // target pixel is unused.
                    if (_search_kernel(cx, cy, size, NULL, true)) {
                        _put_pixel(cx, cy, fill_pixel);
                        dilated_cnt++;
                    }
                }
            }
            
            return dilated_cnt;
        }
};

class _Erosion_fix15 : public Tilecache<fix15_short_t> {

    protected:
        virtual bool is_match_pixel(const int cx, const int cy, 
                                    const fix15_short_t* target_pixel) 
        {
            // target_pixel is ignored.
            const fix15_short_t* ref_pixel = 
                (const fix15_short_t*)get_cached_pixel(cx, cy, false);
            if (ref_pixel != NULL && ref_pixel[3] != 0)
                return true;
            return false;
        }
        
    public:
        _Erosion_fix15() : Tilecache(NPY_UINT16, 4)
        {
        }
        
        int erode(PyObject* py_targ_tile, // the write target tile. 
                  int tx, int ty,  // the position of py_filled_tile
                  const fix15_short_t* fill_pixel,
                  int size)
        {

            int eroded_cnt = 0;

            for (int cy=0; cy < MYPAINT_TILE_SIZE; cy++) {
                for (int cx=0; cx < MYPAINT_TILE_SIZE; cx++) {
                    if ( ! _search_kernel(cx, cy, size, NULL, false)) {
                        fix15_short_t* targ_pixel = 
                            (fix15_short_t*)get_tile_pixel(
                                (PyArrayObject*)py_targ_tile, cx, cy);
                        targ_pixel[0] = fill_pixel[0];
                        targ_pixel[1] = fill_pixel[1];
                        targ_pixel[2] = fill_pixel[2];
                        targ_pixel[3] = (fix15_short_t)fix15_one; 
                        eroded_cnt++;
                    }
                }
            }
            
            return eroded_cnt;
        }

};

// _GapCloser : Specialized class for detecting contour, to fill-gap.
// This is for STATUS_PIXEL type,flag_based erode/dilate operation.
//
// in new flood_fill(), we use tiles of 8bit bitflag status to figure out 
// original contour, not-filled pixels area(EXIST_FLAG) , 
// dilated area from original contour(DILATED_FLAG) and
// the result, eroded area from dilated area(ERODED_FLAG).

#define STATUS_PIXEL char
#define STATUS_PIXEL_NUMPY NPY_UINT8
// when changing STATUS_PIXEL type, DO NOT FORGET TO UPDATE
// NUMPY ARRAY TYPE IN CONSTRUCTOR!!

class _GapCloser: public Tilecache<STATUS_PIXEL> {

    protected:
        // pixel status information flag are now defined as 
        // preprocessor constants, to easily share it with other modules.

        inline void _put_flag(const int cx, const int cy, const STATUS_PIXEL flag)
        {
            STATUS_PIXEL* dst_pixel = get_cached_pixel(cx, cy, true);
            *dst_pixel |= flag;
        }

        virtual bool is_match_pixel(const int cx, const int cy, const STATUS_PIXEL* target_pixel) 
        {
            STATUS_PIXEL* dst_pixel = get_cached_pixel(cx, cy, false);
            if (dst_pixel == NULL 
                || (*dst_pixel & *target_pixel) == 0) {
                return false;
            }
            return true;
        }

        //// special tile information methods
        //
        // Special informations recorded into a tile with
        // setting INFO_FLAG bitflag to paticular pixel.
        // And they have various means,according to its location. 
        //
        // These methods are accessible from outside this class
        // without any instance, now.
        
        // tile status information flags.
        // This flag is only set to paticular location of tile pixels.
        static const STATUS_PIXEL TILE_INFO_FLAG = 0x80;

        static const STATUS_PIXEL DILATED_TILE_FLAG = 0x01;
        //static const STATUS_PIXEL ERODED_TILE_FLAG = 0x02;
        static const STATUS_PIXEL BORDER_PROCESSED_TILE_FLAG = 0x04;
        static const STATUS_PIXEL VALID_TILE_FLAG = 0x80;  // valid(exist) tile.

        // Maximum count of Tile info flags.
        // DO NOT FORGET TO UPDATE THIS, when adding above tile flag constant!
        // VALID_TILE_FLAG is not included to this number.
        static const int TILE_INFO_MAX = 3; 

        STATUS_PIXEL _get_tile_info(PyArrayObject* tile)
        {
            STATUS_PIXEL retflag = 0;
            STATUS_PIXEL flag = 1;
            STATUS_PIXEL* pixel;

#ifdef HEAVY_DEBUG
            assert(tile != NULL);
#endif
            for(int i=0; i < TILE_INFO_MAX; i++) {
                pixel = get_tile_pixel(tile, 0, i);
                if (*pixel & TILE_INFO_FLAG)
                    retflag |= flag;
                flag = flag << 1;
            }
            return retflag | VALID_TILE_FLAG;
        }

        // Utility method.
        // This is NULL-safe, and can follow when the tile
        // dynamically generated.
        STATUS_PIXEL _get_tile_info(int index)
        {
#ifdef HEAVY_DEBUG
            assert(index >= 0);
            assert(index < MAX_CACHE_COUNT);
#endif
            if (m_cache_tiles[index] != NULL)
                return _get_tile_info((PyArrayObject*)m_cache_tiles[index]);
            return 0;
        }

        void _set_tile_info(PyArrayObject* tile, STATUS_PIXEL flag)
        {
#ifdef HEAVY_DEBUG
            assert(tile != NULL);
#endif
            STATUS_PIXEL* pixel;
            for(int i=0; i < TILE_INFO_MAX && flag != 0; i++) {
                pixel = get_tile_pixel(tile, 0, i);
                if (flag & 0x01)
                    *pixel |= TILE_INFO_FLAG;
                flag = flag >> 1;
            }
        }

        // Utility method.
        // This is NULL-safe, and can follow when the tile
        // dynamically generated.
        void _set_tile_info(int index, STATUS_PIXEL flag)
        {
#ifdef HEAVY_DEBUG
            assert(index >= 0);
            assert(index < MAX_CACHE_COUNT);
#endif
            if (m_cache_tiles[index] != NULL)
                _set_tile_info((PyArrayObject*)m_cache_tiles[index], flag);
        }

        // Dilate entire center tile and some area of surrounding 8 tiles, 
        // to ensure center status tile can get complete dilation.
        void _dilate_contour(int gap_radius) 
        {
            STATUS_PIXEL tile_info = _get_tile_info(CENTER_TILE_INDEX);

#ifdef HEAVY_DEBUG
            assert(gap_radius <= MAX_OPERATION_SIZE);
#endif

            if ((tile_info & DILATED_TILE_FLAG) != 0)
                return;

            STATUS_PIXEL flag = EXIST_FLAG;
            
            for (int y = -gap_radius; y < MYPAINT_TILE_SIZE+gap_radius; y++) {
                for (int x = -gap_radius; x < MYPAINT_TILE_SIZE+gap_radius; x++) {
                    STATUS_PIXEL* pixel = get_cached_pixel(x, y, false);
                    if (pixel != NULL   
                        && (*pixel & DILATED_FLAG) != 0) {
                        // pixel exists, but it already dilated.
                        // = does nothing
                        //
                        // otherwise, pixel is not dilated
                        // or, pixel does not exist(==NULL)
                        // we might place pixel.
                    }
                    else if (_search_kernel(x, y, gap_radius, &flag, true)) {
                        // if pixel is NULL (i.e. tile is NULL)
                        // generate it.
                        if (pixel == NULL)
                            pixel = get_cached_pixel(x, y, true);
            
                        *pixel |= DILATED_FLAG;
                    }
                }
            }

            _set_tile_info(CENTER_TILE_INDEX, DILATED_TILE_FLAG);
        }
        
        // Convert(and initialize) color pixel tile into 8bit status tile.
        // status tile updated with 'EXISTED' flag, but not cleared.
        // So exisitng status tile can keep dilated/eroded flags,
        // which is set another call of this function.
        PyObject* _convert_state_tile(PyObject* py_status_dict, // 8bit state dict
                                      PyArrayObject* py_surf_tile, // fix15 color tile from surface
                                      const fix15_short_t* targ_pixel,
                                      const fix15_t tolerance,
                                      PyObject* key) // a tuple of tile location
        {
            

            PyObject* state_tile = PyDict_GetItem(py_status_dict, key);

            if (state_tile == NULL) {
                state_tile = _generate_tile();
                PyDict_SetItem(py_status_dict, key, state_tile);
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
                        STATUS_PIXEL* state_pixel = 
                            get_tile_pixel((PyArrayObject*)state_tile, px, py);
                        *state_pixel |= EXIST_FLAG;
                    }
                }
            }
            
            return state_tile;
        }

        // Before call this method, init_cached_tiles() must be already called.
        // This method converts color tiles around (tx, ty) into state flag tiles.
        // also, this setup function sets newly generated status tile into cache.
        void _setup_state_tiles(PyObject* py_status_dict, // 8bit state tiles dict
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
                            _convert_state_tile(py_status_dict, 
                                                (PyArrayObject*)surf_tile, 
                                                targ_pixel, 
                                                tolerance,
                                                key);
                        Py_DECREF(surf_tile);
                        m_cache_tiles[i] = state_tile;
                    }
                }
            }

        }

    public:

        _GapCloser() : Tilecache(STATUS_PIXEL_NUMPY, 1)
        {
        }

        void close_gap(PyObject* py_status_dict, // 8bit state tiles dict
                       PyObject* py_surfdict, // source surface tiles dict
                       int tx, int ty, // current tile location
                       const fix15_short_t* targ_pixel,// target pixel for conversion
                                                      // from color tile to status tile.
                       fix15_t tolerance,
                       int gap_radius)
        {
#ifdef HEAVY_DEBUG
            assert(gap_radius <= MAX_OPERATION_SIZE);
#endif

            init_cached_tiles(py_status_dict, tx, ty); 

            _setup_state_tiles(py_status_dict, 
                               py_surfdict, 
                               targ_pixel,
                               tolerance);
        
            // Filling gap with dilated contour
            // (contour = not flood-fill targeted pixel area).
            _dilate_contour(gap_radius);

          //// Then, erode the contour to make it much thinner.
          //int erosion_size = gap_radius / 2;
          //if (erosion_size > 0 )
          //    _erode_contour(erosion_size);

            // I also created to make much thinner contour with 
            // skelton morphology, but it was too slow
            // (mypaint hanged up for several seconds) 
            // when floodfill spill out the targeted area.
            // so I gave up it.

            finalize_cached_tiles(py_status_dict); 
        }
        
         
      //void test_dilate(PyObject* py_status_dict, // 8bit state tiles dict
      //                 const int tx, const int ty,
      //                 const int gap_radius)
      //{
      //    init_cached_tiles(py_status_dict, tx, ty); 
      //
      //    // Filling gap with dilated contour
      //    // (contour = not flood-fill targeted pixel area).
      //    _dilate_contour(gap_radius);
      //    int erosion_size = gap_radius / 2;
      //    if (erosion_size > 0 )
      //        _erode_contour(erosion_size);
      //}

    void ensure_dilate_tile(PyArrayObject* targ_tile, 
                            const int dilation_size)
    {
#ifdef HEAVY_DEBUG
            assert(targ_tile == 
                    (PyArrayObject*)m_cache_tiles[CENTER_TILE_INDEX]);
#endif

        int tile_status = _get_tile_info(CENTER_TILE_INDEX);
        STATUS_PIXEL flag = EXIST_FLAG;

        if ( (tile_status & DILATED_TILE_FLAG) == 0)
        {
            for (int y = 0; y < MYPAINT_TILE_SIZE; y++) {
                for (int x = 0; x < MYPAINT_TILE_SIZE; x++) {
                    STATUS_PIXEL* pixel = get_tile_pixel(
                                            targ_tile,
                                            x, y);
                    if ((*pixel & DILATED_FLAG) != 0) {
                        // pixel exists, but it already dilated.
                        // = does nothing
                        //
                        // otherwise, pixel is not dilated
                        // or, pixel does not exist(==NULL)
                        // we might place pixel.
                    }
                    else if (_search_kernel(x, y, dilation_size, &flag, true)) {
                        *pixel |= DILATED_FLAG;
                    }
                }
            }

            _set_tile_info(CENTER_TILE_INDEX, DILATED_TILE_FLAG);
        }
    }

};

// _CountourFiller : Specialized class for painting dilated (gap-connected)
// contour.
//
// This class fills gap-connected contour which is generated 
// by _GapCloser class.
// It is merged with flood-filled pixel tiles, thus entire (thick)
// gap-connected contour and flood-filled pixels are painted.
//
// After that, the painted area should be eroded with _Erosion_fix15 class.
// Thus justfied, mostly same shape of original contour (but gap-connected)
// would be completed.
#define VALID_BORDER_TILE (VALID_TILE_FLAG | BORDER_PROCESSED_TILE_FLAG)

class _ContourFiller: public _GapCloser {

    protected:

        void _fill_pixel_tile(PyArrayObject* pixel_tile,
                             const fix15_short_t* fill_pixel)
        {
            for (int y = 0; y < MYPAINT_TILE_SIZE; y++) {
                for (int x = 0; x < MYPAINT_TILE_SIZE; x++) {
                    STATUS_PIXEL* pixel = get_cached_pixel(x, y, false);
                    if (pixel != NULL   
                        && (*pixel & (DILATED_FLAG | EXIST_FLAG)) != 0) {
                        fix15_short_t* targ_pixel = 
                            (fix15_short_t*)get_tile_pixel(pixel_tile, x, y);
                        targ_pixel[0] = fill_pixel[0];
                        targ_pixel[1] = fill_pixel[1];
                        targ_pixel[2] = fill_pixel[2];
                        targ_pixel[3] = (fix15_short_t)fix15_one; 
                    }
                }
            }
        }

        int _search_continuous_pixel(const int cx, const int cy,
                                     const int ex, const int ey)
        {
            // c_pixel, i.e. current center tile pixel.
            STATUS_PIXEL* c_pixel = get_cached_pixel(cx, cy, false);
            // e_pixel, i.e. exceeding tile border pixel.
            STATUS_PIXEL* e_pixel = get_cached_pixel(ex, ey, false);
            if ( c_pixel != NULL && e_pixel != NULL   
                 && (*c_pixel & DILATED_FLAG) != 0
                 && (*e_pixel & DILATED_FLAG) != 0) {
                *e_pixel |= BORDER_FLAG;
                return 1;
            }
            return 0;
        }

        inline bool _is_valid_tile(const int cache_index)
        {
            STATUS_PIXEL tile_info = _get_tile_info(cache_index);
            if (tile_info != 0 
                && (tile_info & BORDER_PROCESSED_TILE_FLAG) == 0)
                return true;
            return false;
        }

        int _search_borders()
        {
            int cnt = 0;
            // searching the top and bottom ridge of tile
            for (int x = 0; x < MYPAINT_TILE_SIZE; x++) {
                if (_is_valid_tile(ABOVE_TILE_INDEX))
                    cnt += _search_continuous_pixel(x, 0, x, -1);

                if (_is_valid_tile(BELOW_TILE_INDEX))
                    cnt += _search_continuous_pixel(x, MYPAINT_TILE_SIZE-1, 
                                                    x, MYPAINT_TILE_SIZE);
            }
            // searching the both side ridge of tile
            for (int y = 0; y < MYPAINT_TILE_SIZE; y++) {
                if (_is_valid_tile(LEFT_TILE_INDEX))
                    cnt += _search_continuous_pixel(0, y, -1, y);

                if (_is_valid_tile(RIGHT_TILE_INDEX))
                    cnt += _search_continuous_pixel(MYPAINT_TILE_SIZE-1, y,
                                                    MYPAINT_TILE_SIZE, y);
            }
            return cnt;
        }

        int _process_current_tile(PyArrayObject* pixel_tile,
                                  const fix15_short_t* fill_pixel)
        {
            STATUS_PIXEL tile_info = _get_tile_info(CENTER_TILE_INDEX);
            if (tile_info == 0 
                || (tile_info & BORDER_PROCESSED_TILE_FLAG) != 0)
                return 0;

            for (int y = 0; y < MYPAINT_TILE_SIZE; y++) {
                for (int x = 0; x < MYPAINT_TILE_SIZE; x++) {
                    STATUS_PIXEL* pixel = get_cached_pixel(x, y, false);
                    if (pixel != NULL   
                        && (*pixel & BORDER_FLAG) != 0) {
                        _fill_pixel_tile(pixel_tile, fill_pixel);
                        _set_tile_info(CENTER_TILE_INDEX, 
                                       BORDER_PROCESSED_TILE_FLAG); 
                        return _search_borders();
                    }
                }
            }
            return 0;
        }

    public:

        _ContourFiller() : _GapCloser()
        {
        }

        int fill(PyObject* py_status_dict, // 8bit state tiles dict
                  int tx, int ty, // current tile location
                  PyArrayObject* pixel_tile, // target pixel tile(not status flag tile)
                  const fix15_short_t* fill_pixel)
        {
            init_cached_tiles(py_status_dict, tx, ty); 
            int ret_value = _process_current_tile(pixel_tile, fill_pixel);
            finalize_cached_tiles(py_status_dict); 
            return ret_value;
        }


};

//// Python Interface functions.
//

/** close_gap:
 *
 * @py_status_dict: a Python dictinary, which stores 'state tiles'
 * @py_pre_filled_tile: a Numpy.array object of color tile.
 * @tx, ty: the position of py_pre_filled_tile, in tile coordinate.
 * @targ_r, targ_g, targ_b, targ_a: premult target pixel color
 * @tol: tolerance,[0.0 - 1.0] same as tile_flood_fill().
 * @gap_radius: the filling gap radius.
 * returns: Nothing. returning PyNone always.
 *
 * extract contour into state tiles, 
 * and that state tile is stored into py_status_dict, with key of (tx, ty).
 * And later, this state tile used in flood_fill function, to detect
 * ignorable gaps.
 */

PyObject* 
close_gap(PyObject* py_status_dict, // the tiledict for status tiles.
          PyObject* py_surfdict, //  source surface tile dict.
          const int tx, const int ty,  // the position of py_filled_tile
          const int targ_r, const int targ_g, const int targ_b, const int targ_a, //premult target pixel color
          const double tol,   // pixel tolerance of filled area.
          const int gap_radius) // overflow-preventing closable gap size.
{
#ifdef HEAVY_DEBUG            
    assert(py_status_dict != NULL);
    assert(py_surfdict != NULL);
    assert(0.0 <= tol);
    assert(tol <= 1.0);
    assert(gap_radius <= MAX_OPERATION_SIZE);
#endif
    // actually, this function is wrapper.
    
    // XXX Morphology Operation object defined as static. 
    // Because this function called each time before a tile flood-filled,
    // I think constructing/destructing cost cannot be ignored.
    // Otherwise, we can use something start/end wrapper function and
    // use PyCapsule object...
    static _GapCloser m;


    fix15_short_t targ_pixel[4] = {(fix15_short_t)targ_r,
                                   (fix15_short_t)targ_g,
                                   (fix15_short_t)targ_b,
                                   (fix15_short_t)targ_a};

    // XXX Copied from fill.cpp tile_flood_fill()
    const fix15_t tolerance = (fix15_t)(  MIN(1.0, MAX(0.0, tol))
                                        * fix15_one);

    m.close_gap(py_status_dict, py_surfdict,
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
 * @fill_r, fill_g, fill_b: premult filling pixel colors.
 * @dilation_size: the dilation size of filled area, in pixel.
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
                   const int tx, const int ty,  // the position of py_filled_tile 
                   const double fill_r, const double fill_g, const double fill_b, //premult pixel color 
                   const int dilation_size)    // growing size from center pixel.  
{
    // Morphology object defined as static. 
    // Because this function called each time before a tile flood-filled,
    // so constructing/destructing cost would not be ignored.
    static _Dilation_fix15 d;

#ifdef HEAVY_DEBUG            
    assert(py_dilated != NULL);
    assert(py_filled_tile != NULL);
    assert(dilation_size <= MAX_OPERATION_SIZE);
#endif

    
    // Actually alpha value is not used currently.
    // for future use.
    double alpha=(double)fix15_one;
    fix15_short_t fill_pixel[3] = {(fix15_short_clamp)(fill_r * alpha),
                                   (fix15_short_clamp)(fill_g * alpha),
                                   (fix15_short_clamp)(fill_b * alpha)};
    
    // _Dilation_fix15 class is specialized dilating filled pixels,
    // and uses 'current pixel' for dilation, not fixed/assigned pixel.
    // Therefore this class does not need internal pixel member,
    // dummy pixel is passed to setup_morphology_params().

    d.init_cached_tiles(py_dilated, tx, ty);
    d.dilate(py_filled_tile, tx, ty, fill_pixel, dilation_size);
    d.finalize_cached_tiles(py_dilated);

    Py_RETURN_NONE;
}

// erode color filled tile. almost same as dilate_filled_tile()
PyObject*
erode_filled_tile(PyObject* py_srctile_dict, // the tiledict for SOURCE pixel tiles.
                  PyObject* py_targ_tile, // the filled TARGET tile.  
                  const int tx, const int ty,  // the position of py_targ_tile 
                  const double fill_r, const double fill_g, const double fill_b, //premult pixel color 
                  const int erosion_size)    // growing size from center pixel.  
{
    // Morphology object defined as static. 
    // Because this function called each time before a tile flood-filled,
    // so constructing/destructing cost would not be ignored.
    static _Erosion_fix15 e;

#ifdef HEAVY_DEBUG            
    assert(py_srctile_dict != NULL);
    assert(py_targ_tile != NULL);
    assert(erosion_size <= MAX_OPERATION_SIZE);
#endif

    
    // Actually alpha value is not used currently.
    // for future use.
    double alpha=(double)fix15_one;
    fix15_short_t fill_pixel[3] = {(fix15_short_clamp)(fill_r * alpha),
                                   (fix15_short_clamp)(fill_g * alpha),
                                   (fix15_short_clamp)(fill_b * alpha)};
    
    // _Dilation_fix15 class is specialized dilating filled pixels,
    // and uses 'current pixel' for erosion, not fixed/assigned pixel.
    // Therefore this class does not need internal pixel member,
    // dummy pixel is passed to setup_morphology_params().
    
    e.init_cached_tiles(py_srctile_dict, tx, ty);
    e.erode(py_targ_tile, tx, ty, fill_pixel, erosion_size);
    e.finalize_cached_tiles(py_srctile_dict);

    Py_RETURN_NONE;
}


/** contour_fill:
 *
 * @py_status_dict: a Python dictinary, which stores 'state tiles'
 * @tx, ty: the position of py_pre_filled_tile, in tile coordinate.
 * @py_pixel_tile : a Numpy.array object of color pixel tile.
 * @targ_r, targ_g, targ_b, targ_a: premult target pixel color
 * returns: new Border flag count, in Python Interger value.
 *
 * Fill contours which is set BORDER_FLAG. 
 * This means "entire dilated contour + inner area is painted over with one color"
 * After this function called, erode_filled_tile() loop would be called
 * and it would justify dilated flood-fill area shape.
 *
 * This function returns newly set (i.e. not processed yet) Border flag count.
 * That return value accumulated in loop in caller(python function),
 * and that accumulated count is 0, the countour filling loop would end.
 *
 */

PyObject*
contour_fill(PyObject* py_status_dict, // the tiledict for status tiles.
             const int tx, const int ty,  // the position of py_filled_tile
             PyObject* py_pixel_tile, // the TARGET pixel tile.  
             const int targ_r, const int targ_g, const int targ_b, const int targ_a)
{
    static _ContourFiller c;

#ifdef HEAVY_DEBUG            
    assert(py_status_dict != NULL);
    assert(py_pixel_tile != NULL);
#endif
    
    fix15_short_t targ_pixel[4] = {(fix15_short_t)targ_r,
                                   (fix15_short_t)targ_g,
                                   (fix15_short_t)targ_b,
                                   (fix15_short_t)targ_a};


    int border_set_count = c.fill(py_status_dict,
                                  tx, ty,
                                  (PyArrayObject*)py_pixel_tile,
                                  targ_pixel);

    PyObject *result = Py_BuildValue("i", border_set_count);
    return result;
}

/** ensure_dilate_tile:
 *
 * @py_targ_tile : a Numpy.array object of contour status tile.
 * returns: always PyNone
 *
 * This function is to ensure the tile is dilated.
 * Some tiles have contour but not completely dilated.
 * In older version, such tiles is not problem, but now
 * we need more accurate shape over entire surrounding contour.
 * So This function created.
 */
PyObject*
ensure_dilate_tile(PyObject* py_status_dict,
                   const int tx, const int ty,
                   PyObject* py_targ_tile, // the TARGET status tile)  
                   const int dilation_size)
{
    static _GapCloser c;
#ifdef HEAVY_DEBUG            
    assert(py_targ_tile != NULL);
    assert(dilation_size <= MAX_OPERATION_SIZE);
#endif
    c.init_cached_tiles(py_status_dict, tx, ty); 
    c.ensure_dilate_tile((PyArrayObject*)py_targ_tile,
                         dilation_size);
    c.finalize_cached_tiles(py_status_dict); 
    Py_RETURN_NONE;
}

#ifdef HEAVY_DEBUG
/*
PyObject*
test(PyObject* py_status_dict, // the tiledict for status tiles.
     const int tx, const int ty,
     const int dilate_size)    // growing size from center pixel.
{
    static _GapCloser m;

    m.test_dilate(py_status_dict, 
                  tx, ty,
                  dilate_size);

    Py_RETURN_NONE;
}
*/

//// Utility functions for c++ modules

int get_status_flag(PyObject* sts_tile, const int x, const int y)
{
    if(sts_tile) {
        return (int)*((char*)get_tile_pixel((PyArrayObject*)sts_tile, 
                                             x, y));
    }
    return 0;
}

void set_status_flag(PyObject* sts_tile, 
                            const int x, const int y, 
                            const char flag)
{
    if(sts_tile) {
        char* flagptr = (char*)get_tile_pixel((PyArrayObject*)sts_tile, 
                                               x, y);
        *flagptr |= flag;
    }
}
#endif
