
#include "fill_guard.hpp"
#include "common.hpp"
#include "fix15.hpp"
#include "fill.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <glib.h>
#include <mypaint-tiled-surface.h>

//int idx = (x + 1) + (y + 1) * 3;
#define GET_TILE_INDEX(x, y)  (((x) + 1) + (((y) + 1) * 3))
#define CENTER_TILE_INDEX 4

#define GET_TX_FROM_INDEX(tx, idx) ((tx) + (((idx) % 3) - 1))
#define GET_TY_FROM_INDEX(ty, idx) ((ty) + (((idx) / 3) - 1))

// convert 0 to 8 index to 0 to 24
//#define CONVERT_TILE_INDEX(i)  ((((i) % 3) + 6) + ((i / 3) * 5))

// Base class implementation.
//
#define MAX_CACHE_COUNT 9

PyObject* MorphologyBase::s_cache_tiles[MAX_CACHE_COUNT];
PyObject* MorphologyBase::s_cache_keys[MAX_CACHE_COUNT];

PyObject*
MorphologyBase::_generate_cache_key(int tx, int ty)
{
    PyObject* txo = PyInt_FromLong(tx);
    PyObject* tyo = PyInt_FromLong(ty);
    PyObject* key = PyTuple_Pack(2, txo, tyo);
    Py_DECREF(txo);
    Py_DECREF(tyo);
    return key;
}

PyObject* 
MorphologyBase::_generate_tile()
{
    npy_intp dims[] = {MYPAINT_TILE_SIZE, MYPAINT_TILE_SIZE, m_dimension_cnt};
    return PyArray_ZEROS(3, dims, m_NPY_TYPE, 0);
}

PyObject* 
MorphologyBase::_get_cached_key(int index)
{
    return s_cache_keys[index];
}

// Get target(to be put dilated pixels) tile from offsets.
PyObject* MorphologyBase::_get_cached_tile(int otx, // otx, oty have values ranging from -1 to 1,
                                          int oty, // it is offsets from center tile. 
                                          bool generate) 
{
    return _get_cached_tile_from_index(GET_TILE_INDEX(otx, oty), generate);
}

PyObject* 
MorphologyBase::_get_cached_tile_from_index(int index, 
                                            bool generate) 
{
    if (s_cache_tiles[index] == NULL) {
        if (generate == true) {
            PyObject* dst_tile = _generate_tile();
            s_cache_tiles[index] = dst_tile;
#ifdef HEAVY_DEBUG
            assert(s_cache_keys[index] != NULL);
#endif
            /* key is already generated in _init_cached_tiles
            PyObject* key = _generate_cache_key(GET_TX_FROM_INDEX(m_tx, index), 
                                                GET_TY_FROM_INDEX(m_ty, index));
            s_cache_keys[index] = key;
            */
            return dst_tile;
        }
    }
    return s_cache_tiles[index];
}

void 
MorphologyBase::_init_cached_tiles(PyObject* py_tiledict, int tx, int ty)
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
                // Otherwise, some tile would be generated in this module.
                //
                // Such 'new tile' would be 'increfed' with calling PyDict_SetItem() at
                // _finalize_dilated_tiles().
                // On the other hand, A  Borrowed tile, it is already in tile
                // dictionary, so it would not get increfed even PyDict_SetItem().
                // And all items decrefed in _finalize_cached_tiles().
                // Therefore we need incref it here.
                //
                // key is generated every time cache is set up, so no need to incref.
#ifdef HEAVY_DEBUG
                assert(dst_tile->ob_refcnt == 2);
#endif
            }
            s_cache_tiles[idx] = dst_tile; // if tile does not exist, this means 'initialize with NULL'
            s_cache_keys[idx] = key;
        }
    }
}

unsigned int
MorphologyBase::_finalize_cached_tiles(PyObject* py_tiledict)
{

    unsigned int updated_cnt = 0;
    PyObject* key;

    for (int idx = 0; idx < MAX_CACHE_COUNT; idx++){

        key = s_cache_keys[idx];

        // s_cache_tiles might not have actual tile.be careful.
        if (s_cache_tiles[idx] != NULL) {
            PyObject* dst_tile = (PyObject*)s_cache_tiles[idx];
            if (PyDict_SetItem(py_tiledict, key, dst_tile) == 0) {
                updated_cnt++;
            }
            Py_DECREF(dst_tile);
            s_cache_tiles[idx] = NULL;
#ifdef HEAVY_DEBUG
            assert(dst_tile->ob_refcnt == 1);
#endif
        }

        // Different from s_cache_tiles,
        // Each s_cache_keys elements should be valid python object.
        // so every 'key' MUST BE DONE Py_DECREF.
        Py_DECREF(key);
        s_cache_keys[idx] = NULL;

#ifdef HEAVY_DEBUG
        assert(key->ob_refcnt == 1);
#endif
    }

    return updated_cnt;
}

// Call this method before morphology operations starts.
void 
MorphologyBase::setup_morphology_params(int size, const char* source_pixel)
{
    m_size = size % (MYPAINT_TILE_SIZE / 2);
    m_source_pixel = source_pixel;
#ifdef HEAVY_DEBUG
    assert(m_source_pixel != NULL);
#endif
}



char* 
MorphologyBase::_get_pixel(PyObject* array,
                           const unsigned int x, 
                           const unsigned int y)
{
    const unsigned int xstride = PyArray_STRIDE((PyArrayObject*)array, 1);
    const unsigned int ystride = PyArray_STRIDE((PyArrayObject*)array, 0);
    return (PyArray_BYTES((PyArrayObject*)array)
            + (y * ystride)
            + (x * xstride));
}



MorphologyBase::MorphologyBase(int npy_type, int dimension_cnt)
{
    m_NPY_TYPE = npy_type;
    m_dimension_cnt = dimension_cnt;
}

MorphologyBase::~MorphologyBase()
{
}

unsigned int 
MorphologyBase::dilate(PyObject* py_dilated, // the tiledict for dilated tiles.
                       PyObject* py_filled_tile, // the filled src tile. 
                       int tx, int ty,  // the position of py_filled_tile
                       int kernel_type  // 0 for square kernel, 1 for diamond kernel
                       )
{
    // Initialize 3x3 tile cache.
    // This cache is to avoid generating tuple object
    // for each time we refer to neighbor tiles.
     _init_cached_tiles(py_dilated, tx, ty); 

    for (int y=0; y < MYPAINT_TILE_SIZE; y++) {
        for (int x=0; x < MYPAINT_TILE_SIZE; x++) {
            char* cur_pixel = _get_pixel(py_filled_tile, x, y);
            if (_is_dilate_target_pixel(cur_pixel)) {
                switch (kernel_type) {
                    case DIAMOND_KERNEL:
                        _put_dilate_diamond_kernel(x, y); 
                        break;

                    case SQUARE_KERNEL:
                    default:
                        _put_dilate_square_kernel(x, y); 
                        break;
                }
            }
        }
    }
    

    // finally, make tiles cache into python directory object.
    // and dereference them.
    return _finalize_cached_tiles(py_dilated);
}

unsigned int 
MorphologyBase::erode(PyObject* py_dilated, // the tiledict for dilated tiles.
                      PyObject* py_target_tile, // the tile to be drawn eroded pixel. 
                      int tx, int ty,  // the position of py_filled_tile
                      int kernel_type  // 0 for square kernel, 1 for diamond kernel
                      )
{
    // Initialize 3x3 tile cache.
    // This cache is to avoid generating tuple object
    // for each time we refer to neighbor tiles.
     _init_cached_tiles(py_dilated, tx, ty); 

    for (int y=0; y < MYPAINT_TILE_SIZE; y++) {
        for (int x=0; x < MYPAINT_TILE_SIZE; x++) {
            // In erode, pixel check is done in kernel placement method.
            switch (kernel_type) {
                case DIAMOND_KERNEL:
                    _put_erode_diamond_kernel(py_target_tile,
                                              x, y); 
                    break;

                case SQUARE_KERNEL:
                default:
                    _put_erode_square_kernel(py_target_tile,
                                             x, y); 
                    break;
            }
        }
    }

    // finally, make tiles cache into python directory object.
    // and dereference them.
    // we need call this even unchanged, to unref tiles and keys.
    return _finalize_cached_tiles(py_dilated);
}


// fix15 dilater class
// This is for ordinary dilating of color pixel tile.
//
class _Dilater_fix15 : public MorphologyBase {

    private:

        static const fix15_t fix15_half = fix15_one >> 1;// for 'alpha priority problem'

    protected:

        void _put_pixel(char* dst_pixel, const char* src_pixel)
        {
            fix15_short_t* t_dst_pixel = (fix15_short_t*)dst_pixel;
            fix15_short_t* t_src_pixel = (fix15_short_t*)src_pixel;

            t_dst_pixel[0] = t_src_pixel[0];
            t_dst_pixel[1] = t_src_pixel[1];
            t_dst_pixel[2] = t_src_pixel[2];
            t_dst_pixel[3] = t_src_pixel[3];
        }

        void _put_pixel(PyObject* dst_tile,int x, int y,const char* pixel)
        {
            fix15_short_t* dst_pixel = (fix15_short_t*)_get_pixel(dst_tile, x, y);
            fix15_short_t* src_pixel = (fix15_short_t*)pixel;
            
            if (dst_pixel[3] == 0) { // rejecting already written pixel 
                dst_pixel[0] = src_pixel[0];
                dst_pixel[1] = src_pixel[1];
                dst_pixel[2] = src_pixel[2];
                dst_pixel[3] = src_pixel[3];
            }
            else if (dst_pixel[3] < src_pixel[3]) {
                // XXX To solve 'alpha priority problem'
                //
                // 'alpha priority problem' is , I named it , such problem that
                // if there is something translucent alpha-blending pixel at filled
                // tiles, it might produce ugly translucent glitch. 
                // Such translucent pixel would come from tolerance parameter of 
                // floodfill_color_match(). 
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
                //  At this time , I took plan c), but I think it might be enough 
                //  using plan a) for almost painting.
                fix15_t alpha = (fix15_t)dst_pixel[3] + (fix15_t)src_pixel[3]; 
                alpha = fix15_mul(alpha, fix15_half);
                dst_pixel[0] = src_pixel[0];
                dst_pixel[1] = src_pixel[1];
                dst_pixel[2] = src_pixel[2];
                dst_pixel[3] = fix15_short_clamp(alpha);
            }
        } 

        bool _is_dilate_target_pixel(const char* pixel)
        {
            if (((fix15_short_t*)pixel)[3] != 0) {
                m_source_pixel = pixel;// fetch this 'current' pixel!
                return true;
            }
            return false;
        }

        // erode function never used.
        bool _is_erode_target_pixel(const char* pixel)
        {
            return false;
        }

    public:
        _Dilater_fix15() : MorphologyBase(NPY_UINT16, 4)
        {
        }

        unsigned int erode(PyObject* py_dilated, // the tiledict for dilated tiles.
                           PyObject* py_target_tile, // the tile to be drawn eroded pixel. 
                           int tx, int ty,  // the position of py_filled_tile
                           int kernel_type   // 0 for square kernel, 1 for diamond kernel
                           )
        {
            return 0; // this class does not erode.
        }

};


// _Morphology_contour : Specialized class for detecting contour, to fill-gap.
// This is for 8bit,flag_based erode/dilate operation.
//
// in new flood_fill(), we use tiles of 8bit bitflag status to figure out 
// original colored not-filled pixels(EXIST_MASK) , 
// dilated pixels(DILATED_MASK) and
// eroded contour pixels(ERODED_MASK).

class _Morphology_contour: public MorphologyBase {

    protected:
        
        fix15_t m_tolerance;

        // pixel status information flag.
        static const char EXIST_MASK = 0x01;
        static const char DILATED_MASK = 0x02; // This means the pixel is just dilated pixel,
                                               // not sure original source pixel.
                                               // it might be empty pixel in source tile.
        static const char ERODED_MASK = ERODED_FLAG;
        static const char PROCESSED_MASK = 0x08; // This means 'This pixel has original source
                                                 // contour pixel, and dilated'.

        // tile status information flag.
        // This flag is only set to pixel(x, y) = (0, 0).
        static const char DILATED_TILE_MASK = 0x40;   // dilation for this tile is EXECUTED
                                                      // (not completed yet, dilation of surrounding 
                                                      // tiles might affect this tile)
        static const char ERODED_TILE_MASK = 0x20;    // entire erosion for this tile is completed
        
        // And, we can get such special infomation from _get_tile_info() method
        // for future expanision.
        

        // In this class, all writing operation is not writing,
        // but bitwise compositing.
        void _put_pixel(char* dst_pixel, const char* src_pixel)
        {
            *dst_pixel |= *src_pixel;
        }

        void _put_pixel(char* dst_pixel, const char src_flag)
        {
            *dst_pixel |= src_flag;
        }

        void _put_pixel(PyObject* dst_tile,int x, int y,const char* pixel)
        {
            char* dst_pixel = _get_pixel(dst_tile, x, y);
            *dst_pixel |= *pixel;
        } 

        bool _is_equal_pixel(const char* pixel1, const char* pixel2)
        {
            return ((*pixel1 & *pixel2) == *pixel2);
        }

        bool _is_erode_target_pixel(const char* pixel)
        {
            return ((*pixel & DILATED_MASK) != 0);
        }

        bool _is_dilate_target_pixel(const char* pixel)
        {
            // contour pixel is exist, and, it is not processed yet.
            if ((*pixel & EXIST_MASK) != 0)
                return ((*pixel & PROCESSED_MASK) == 0);
            return false;
        }

        // - special information methods
        //
        // Special informations recorded into a tile with
        // setting INFO_MASK bitflag to specified pixel.
        // And they have various means,according to pixel location. 
        
        char _get_tile_info(PyArrayObject* tile)
        {
            char* pixel = _get_pixel((PyObject*)tile, 0, 0);
            return (*pixel & 0xf0);
        }

        void _set_tile_info(PyArrayObject* tile, char flag)
        {
            char* pixel = _get_pixel((PyObject*)tile, 0, 0);
            *pixel |= (flag & 0xf0);
        }



        // - contour detecting morphology methods
        //
        // Dilate existing 9 status tiles, to ensure center status tile can get complete dilation.
        // With this dilation, maximum 9+16 = 25 state tiles might be generated.
        // But primary 9 tiles marked as 'dilation executed' and reused,
        // Therefore not so many processing time is consumed.
        void _dilate_contour(int gap_size,    // growing size from center pixel.
                             int kernel_type)  // 0 for square kernel, 1 for diamond kernel
        {
            
            // Setup internal state variables.
            // to failsafe, limit grow size within half of MYPAINT_TILE_SIZE
            char src_pixel = DILATED_MASK;

            setup_morphology_params(gap_size, 
                                    &src_pixel);

           // int sx, sy, ex, ey, cx, cy;
           // char tile_info;

            for (int y = -gap_size; y < gap_size + MYPAINT_TILE_SIZE; y++)
            {
                for (int x = -gap_size; x < gap_size + MYPAINT_TILE_SIZE; x++)
                {
                    char* t_pixel = _get_cached_pixel(x, y);
                    if (_is_dilate_target_pixel(t_pixel)) {

                        switch (kernel_type) {
                            case DIAMOND_KERNEL:
                                _put_dilate_diamond_kernel(x, y); 
                                break;

                            case SQUARE_KERNEL:
                            default:
                                _put_dilate_square_kernel(x, y); 
                                break;
                        }

                        _put_pixel(t_pixel, PROCESSED_MASK);
                    }
                }
            }

            PyObject* target_tile = _get_cached_tile_from_index(CENTER_TILE_INDEX, true);
            _set_tile_info((PyArrayObject*)target_tile, DILATED_TILE_MASK);
            
        }

        void _erode_contour(int gap_size,    // growing size from center pixel.
                            int kernel_type)  // 0 for square kernel, 1 for diamond kernel
        {
            
            // Only center tile should be eroded.
            PyObject* target_tile = _get_cached_tile_from_index(CENTER_TILE_INDEX, false);
#ifdef HEAVY_DEBUG
            assert(target_tile != NULL);
#endif

            char tile_info = _get_tile_info((PyArrayObject*)target_tile);

            if ((tile_info & ERODED_TILE_MASK) != 0)
                return;

#ifdef HEAVY_DEBUG
            assert((tile_info & DILATED_TILE_MASK) != 0);
#endif
            
            // to failsafe, limit grow size within MYPAINT_MYPAINT_TILE_SIZE
            char src_pixel = ERODED_MASK;

            setup_morphology_params(gap_size, 
                                    &src_pixel);
            
            for (int y=0; y < MYPAINT_TILE_SIZE; y++) {
                for (int x=0; x < MYPAINT_TILE_SIZE; x++) {
                    // In erode, pixel check is done in kernel placement method.
                    switch (kernel_type) {
                        case DIAMOND_KERNEL:
                            _put_erode_diamond_kernel(target_tile,
                                                      x, y); 
                            break;

                        case SQUARE_KERNEL:
                        default:
                            _put_erode_square_kernel(target_tile,
                                                     x, y); 
                            break;
                    }
                }
            }
            
            _set_tile_info((PyArrayObject*)target_tile, ERODED_TILE_MASK);
        }

        // Convert(and initialize) color pixel tile into 8bit status tile.
        // status tile updated with 'EXISTED' flag, but not cleared.
        // So exisitng status tile can keep dilated/eroded flags,
        // which is set another call of this function.
        PyObject* _convert_state_tile(PyObject* py_statedict, // 8bit state dict
                                      PyArrayObject* py_surf_tile, // fix15 color tile from surface
                                      const fix15_short_t* targ_pixel,
                                      PyObject* key) // a tuple of tile location
        {
            

            PyObject* state_tile = PyDict_GetItem(py_statedict, key);
#ifdef HEAVY_DEBUG
            assert(state_tile == NULL);
#endif

            state_tile = _generate_tile();
            PyDict_SetItem(py_statedict, key, state_tile);
            // No need to decref tile & key here, it should be done at _finalize_cached_tiles

            char exist_flag_pixel = EXIST_MASK;
            
            for (int py = 0; py < MYPAINT_TILE_SIZE; py++) {
                for (int px = 0; px < MYPAINT_TILE_SIZE; px++) {
                    fix15_short_t* cur_pixel = (fix15_short_t*)_get_pixel((PyObject*)py_surf_tile, px, py);
                  //if (cur_pixel[0] != targ_pixel[0] ||
                  //    cur_pixel[1] != targ_pixel[1] ||
                  //    cur_pixel[2] != targ_pixel[2] ||
                  //    cur_pixel[3] != targ_pixel[3] ) {
                    if (floodfill_color_match(cur_pixel, targ_pixel, (const fix15_t)m_tolerance) == 0) {
                        _put_pixel(state_tile, px, py, &exist_flag_pixel);
                    }
                }
            }
            
            return state_tile;
        }

        // convert color tiles around (tx, ty) into state flag tiles.
        // also, this setup function sets newly generated status tile into cache.
        void _setup_state_tiles(PyObject* py_statedict, // 8bit state tiles dict
                                PyObject* py_surfdict, // source surface tiles dict
                                const fix15_short_t* targ_pixel)
        {
            // py_surfdict is surface tile dictionary, it is not cached in this class.
            // this class have cache array of 'state tiles', not surface one.
            // so,extract source color tiles with python API.
#ifdef HEAVY_DEBUG            
            assert(py_surfdict != NULL);
            assert(py_surfdict != Py_None);
            assert(s_cache_tiles[CENTER_TILE_INDEX] != NULL);
#endif

            PyObject* attr_name_rgba = Py_BuildValue("s","rgba");

            for (int i=0; i < 9; i++) {

                if (s_cache_tiles[i] == NULL) {
                    PyObject* key = _get_cached_key(i);
                    PyObject* surf_tile = PyDict_GetItem(py_surfdict, key);
                    if (surf_tile != NULL) {

                        // Check whether it is readonly _Tile object or not.
                        int is_tile_obj = PyObject_HasAttr(surf_tile, attr_name_rgba);
                        if (is_tile_obj != 0) {
                            surf_tile = PyObject_GetAttr(surf_tile,
                                                         attr_name_rgba);
                            Py_DECREF(surf_tile);
#ifdef HEAVY_DEBUG            
                            assert(surf_tile->ob_refcnt >= 1);
#endif
                        }

                        PyObject* state_tile = _convert_state_tile(py_statedict, 
                                                                   (PyArrayObject*)surf_tile, 
                                                                   targ_pixel, 
                                                                   key);
                        s_cache_tiles[i] = state_tile;
                    }
                    else if (i == CENTER_TILE_INDEX) {    
                        // surface tile does not exist - 
                        // i.e. user clicked empty tile!
                        // However, we need center state tile
                        // everytime detecting contour.
                        // so just generate it.
                        s_cache_tiles[i] = _generate_tile();

                        // State tiles in other empty place might be needed
                        // as dilation going on, but it would be generated
                        // on demand.
                    }
                }
            }
        }

    public:

        _Morphology_contour() : MorphologyBase(NPY_UINT8, 1)
        {
        }

        void detect_contour(PyObject* py_statedict, // 8bit state tiles dict
                            PyObject* py_surfdict, // source surface tiles dict
                            int tx, int ty, // current tile location
                            const fix15_short_t* targ_pixel,// target pixel for conversion
                                                            // from color tile to status tile.
                            fix15_t tolerance,
                            int gap_size) 
        {
            _init_cached_tiles(py_statedict, tx, ty); 

            m_tolerance = tolerance;

            _setup_state_tiles(py_statedict, 
                               py_surfdict, 
                               targ_pixel);
        
            // detecting contour with dilate & erode.
            _dilate_contour(gap_size,
                            SQUARE_KERNEL);
            
            _erode_contour(gap_size,
                           DIAMOND_KERNEL);
        
            _finalize_cached_tiles(py_statedict); 
        }

        
};

//
// Python Interface functions.
//


/** detect_contour:
 *
 * @py_statedict: a Python dictinary, which stores 'state tiles'
 * @py_pre_filled_tile: a Numpy.array object of color tile.
 * @x, y: start coordinate of floodfill. i.e. the color to be flood-filled.
 * @tx, ty: the position of py_pre_filled_tile, in tile coordinate.
 * returns: Nothing. i.e. returning PyNone always.
 *
 * extract contour into state tiles, 
 * and that state tile is stored into py_statedict, with key of (tx, ty).
 * And later, this state tile used in flood_fill function, to detect
 * ignorable gaps.
 */

PyObject* 
detect_contour(PyObject* py_statedict, // the tiledict for status tiles.
               PyObject* py_surfdict, //  source surface tile dict.
               int tx, int ty,  // the position of py_filled_tile
               int targ_r, int targ_g, int targ_b, int targ_a, //premult target pixel color
               double tol,
               int gap_size)   // ignorable gap size.
{
    // actually, this function is wrapper.
    static _Morphology_contour m;
    fix15_short_t targ_pixel[4] = {(fix15_short_t)targ_r,
                                   (fix15_short_t)targ_g,
                                   (fix15_short_t)targ_b,
                                   (fix15_short_t)targ_a};

    // XXX Copied from fill.cpp tile_flood_fill()
    const fix15_t tolerance = (fix15_t)(  MIN(1.0, MAX(0.0, tol))
                                        * fix15_one);

    m.detect_contour(py_statedict, py_surfdict,
                     tx, ty, 
                     (const fix15_short_t*)targ_pixel,
                     tolerance,
                     gap_size);

    Py_RETURN_NONE;// DO NOT FORGET THIS!
      
}

/** dilate_filled_tile:
 *
 * @py_dilated: a Python dictinary, which stores 'dilated color tiles'
 * @py_pre_filled_tile: a Numpy.array object of color tile.
 * @tx, ty: the position of py_pre_filled_tile, in tile coordinate.
 * @grow_size: the growing size of filled area, in pixel.
 * @kernel_type: the shape of dilation. if this is 0(==MorphologyBase::SQUARE_KERNEL),
 *               pixels dilated into square shape.
 *               if this is 1 (==MorphologyBase::DIAMOND_KERNEL), pixels dilated into
 *               diamond like shape.
 * returns: updated tile counts.
 *
 * dilate flood-filled tile, and store newly generated tiles into py_dilated dictionary.
 * Usually, dilating operation would mutiple new tiles from one flood-filled tile.
 * They are stored into py_dilate, and composited other flood-filled dilation images 
 * over and over again.
 * And when the floodfill loop ended,every dilated tiles are composited to flood-filled tiles.
 * (This processing is done in python script)
 * Thus, we can get dilated floodfill image.
 */

// dilate color filled tile. postprocess of flood_fill.
unsigned int 
dilate_filled_tile(PyObject* py_dilated, // the tiledict for dilated tiles.
                   PyObject* py_filled_tile, // the filled src tile. 
                   int tx, int ty,  // the position of py_filled_tile
                   int dilate_size,    // growing size from center pixel.
                   int kernel_type) // 0 for square kernel, 1 for diamond kernel
{
    static _Dilater_fix15 d;
    char dummy_pixel;
    
    // _Dilater_fix15 class is specialized dilating filled pixels,
    // and uses 'current pixel' for dilation, not fixed/assigned pixel.
    // Therefore this class does not need internal pixel member,
    // dummy pixel is passed to setup_morphology_params().

    d.setup_morphology_params(dilate_size, &dummy_pixel);
    return d.dilate(py_dilated, py_filled_tile, tx, ty, kernel_type);

}


