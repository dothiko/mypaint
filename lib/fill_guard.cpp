
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

// Base class for Morphology operation and tile cache management.
// We need tile cache , to speed up accessing tile pixels with
// linear coordinate.
//
// Defining these template base classes in  another file
// causing ImportError even compiling success, so I wrote 
// all of them in this file.

template <class T>
class MorphologyBase
{
    protected:
        static PyObject* s_cache_tiles[MAX_CACHE_COUNT];
        static PyObject* s_cache_keys[MAX_CACHE_COUNT];

        // parameters for on-demand cache tile generation, 
        // assign to Numpy-API.
        int m_NPY_TYPE;
        int m_dimension_cnt;

        int m_tx; // current center tile location, in tiledict.
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

        PyObject* _get_cached_key(int index)
        {
            return s_cache_keys[index];
        }

        // - pixel processing functors.
        class PixelFunctor {
            public:
                virtual int operator()(MorphologyBase<T>* mb, T* pixel, int x, int y) = 0; // put something internal data to pixel
        };

        // iterate pixels of cached tiles with assigned PixelFunctor.
        int _iterate_pixel(PixelFunctor& func, int gap_size)
        {
            int total_cnt = 0;
            for (int y = -gap_size; y < gap_size + MYPAINT_TILE_SIZE; y++) {
                for (int x = -gap_size; x < gap_size + MYPAINT_TILE_SIZE; x++) {
                    short* t_pixel = get_cached_pixel(x, y, false);
                    total_cnt += func(this, t_pixel, x, y);
                }
            }
            return total_cnt;
        }

    public:

        MorphologyBase(int npy_type, int dimension_cnt)
        {
            m_NPY_TYPE = npy_type;
            m_dimension_cnt = dimension_cnt;
        }
        virtual ~MorphologyBase()
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
                        assert(dst_tile->ob_refcnt >= 2);
#endif
                    }
                    s_cache_tiles[idx] = dst_tile; // if tile does not exist, this means 'initialize with NULL'
                    s_cache_keys[idx] = key;
                }
            }
        }
        
        // call this and finalize cache after dilate/erode called.
        unsigned int finalize_cached_tiles(PyObject* py_tiledict)
        {
            unsigned int updated_cnt = 0;
            PyObject* key;

            for (int idx = 0; idx < MAX_CACHE_COUNT; idx++){

                key = s_cache_keys[idx];
#ifdef HEAVY_DEBUG
                assert(key != NULL);
#endif
                // s_cache_tiles might not have actual tile.be careful.
                if (s_cache_tiles[idx] != NULL) {
                    PyObject* dst_tile = (PyObject*)s_cache_tiles[idx];
                    if (PyDict_SetItem(py_tiledict, key, dst_tile) == 0) {
                        updated_cnt++;
                    }
                    Py_DECREF(dst_tile);
                    s_cache_tiles[idx] = NULL;
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

        T* get_pixel(PyObject* array,
                     const unsigned int x, 
                     const unsigned int y)
        {
            const unsigned int xstride = PyArray_STRIDE((PyArrayObject*)array, 1);
            const unsigned int ystride = PyArray_STRIDE((PyArrayObject*)array, 0);
            return (T*)(PyArray_BYTES((PyArrayObject*)array)
                    + (y * ystride)
                    + (x * xstride));
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
            int otx = 0;
            int oty = 0;
            
            // 'cache coordinate' is converted to tile local one.
            // and split it RELATIVE tiledict key offset(otx, oty).
            if (cx < 0) {
                otx = -1;
                cx += MYPAINT_TILE_SIZE;
            }
            else if (cx >= MYPAINT_TILE_SIZE) {
                otx = 1;
                cx -= MYPAINT_TILE_SIZE;
            }

            if (cy < 0) {
                oty = -1;
                cy += MYPAINT_TILE_SIZE;
            }
            else if (cy >= MYPAINT_TILE_SIZE) {
                oty = 1;
                cy -= MYPAINT_TILE_SIZE;
            }
            
            PyObject* dst_tile = _get_cached_tile(otx, oty, generate);
            if (dst_tile != NULL)
                return get_pixel(dst_tile, cx, cy); 
            return NULL;
        }

        virtual void put_pixel(T* dst_pixel, const T* src_pixel)
        {
            *dst_pixel = *src_pixel;
        }

        virtual void put_pixel(PyObject* dst_tile, int x, int y, const T* pixel)
        {// Utility method.
            put_pixel(get_pixel(dst_tile, x, y), pixel);
        }

        // put pixel with relative-cache coordinate.
        virtual void put_pixel(int rx, int ry, const T* pixel)
        {
            T* targ_pixel = get_cached_pixel(rx, ry, true); // generate tile,if it does not exist
            *targ_pixel = *pixel;
        }
        
        // Utility method , to put type T immidiate value.
        // this might not work for some type.
        virtual void put_pixel(int rx, int ry, T pixel)
        {
            // so, initially this is dummy. to use, override this.
            assert(false /* NOT IMPLEMENTED */);
        }

        virtual bool is_equal_cached_pixel(int rx, int ry, const T* pixel)
        {
            T* targ_pixel = get_cached_pixel(rx, ry, false); 
            if (targ_pixel != NULL)
                return *targ_pixel == *pixel;
            return false;
        }
        
};

template <class T>
PyObject* MorphologyBase<T>::s_cache_tiles[MAX_CACHE_COUNT];

template <class T>
PyObject* MorphologyBase<T>::s_cache_keys[MAX_CACHE_COUNT];

// - Kernel functor.
// Kernel functor is to place a structual element(kernel) into tiles ,
// under various conditions.

template <class T>
class KernelFunctor {
    public:
        virtual int operator()(MorphologyBase<T>* mb, int cx, int cy, int size, const T* pixel){
           return 0; //dummy
        } // put kernel
};

template <class T>
class SquareDilateKernel : public KernelFunctor<T> {

    public:
        virtual int operator()(MorphologyBase<T>* mb, int cx, int cy, int size,const T* pixel)
        {
            for (int dy = 0; dy <= size; dy++) {
                for (int dx = 0; dx <= size; dx++) {
                    if (dx != 0 || dy != 0) {
                        // The first(center) pixel shold be already there.
                        mb->put_pixel(cx-dx, cy-dy, pixel);
                        mb->put_pixel(cx+dx, cy-dy, pixel);
                        mb->put_pixel(cx-dx, cy+dy, pixel);
                        mb->put_pixel(cx+dx, cy+dy, pixel);
                    }
                }
            }
            return 1;
        }
};

template <class T>
class DiamondDilateKernel : public KernelFunctor<T> {

    public:
        virtual int operator()(MorphologyBase<T>* mb, int cx, int cy, int size, const T* pixel)
        {
            for (int dy = 0; dy <= size; dy++) {
                for (int dx = 0; dx <= size-dy; dx++) {
                    if (dx != 0 || dy != 0) {
                        // The first(center) pixel shold be already there.
                        mb->put_pixel(cx-dx, cy-dy, pixel);
                        mb->put_pixel(cx+dx, cy-dy, pixel);
                        mb->put_pixel(cx-dx, cy+dy, pixel);
                        mb->put_pixel(cx+dx, cy+dy, pixel);
                    }
                }
            }
            return 1;
        }
};

template <class T>
class DiamondErodeKernel : public KernelFunctor<T> {
        
        inline int _is_erodable(MorphologyBase<T>* mb, int cx, int cy, int dx, int dy, const T* target_pixel)
        {   
            if (!mb->is_equal_cached_pixel(cx-dx, cy-dy, target_pixel))
                return 0;
            else if (!mb->is_equal_cached_pixel(cx+dx, cy-dy, target_pixel))
                return 0;
            else if (!mb->is_equal_cached_pixel(cx-dx, cy+dy, target_pixel))
                return 0;                                                          
            else if (!mb->is_equal_cached_pixel(cx+dx, cy+dy, target_pixel))
                return 0;
            return 1;
        }

    public:
        virtual int operator()(MorphologyBase<T>* mb, int cx, int cy, int size, 
                               const T* pixel, const T* target_pixel)
        {
            for (int dy = 0; dy <= size; dy++) {
                for (int dx = 0; dx <= size - dy; dx++) {
                    if (_is_erodable(mb, cx, cy, dx, dy, target_pixel) == 0)
                        return 0;
                }
            }

            mb->put_pixel(cx, cy, pixel);
            return 1;
        }
};

template <class T>
class SquareErodeKernel : public DiamondErodeKernel<T> {
        
    public:
        virtual int operator()(MorphologyBase<T>* mb, int cx, int cy, int size, const T* pixel, const T* target_pixel)
        {
            for (int dy = 0; dy <= size; dy++) {
                for (int dx = 0; dx <= size; dx++) {
                    if (_is_erodable(mb, cx, cy, dx, dy, target_pixel) == 0)
                        return 0;
                }
            }

            mb->put_pixel(cx, cy, pixel);
            return 1;
        }
};


// fix15 dilater class
// This is for ordinary dilating of color pixel tile.
//
class _Dilater_fix15 : public MorphologyBase<fix15_short_t> {

    private:

        static const fix15_t fix15_half = fix15_one >> 1;// for 'alpha priority problem'

    protected:
        // currently not used
        virtual void put_pixel(fix15_short_t* dst_pixel, const fix15_short_t* src_pixel)
        {
            dst_pixel[0] = src_pixel[0];
            dst_pixel[1] = src_pixel[1];
            dst_pixel[2] = src_pixel[2];
            dst_pixel[3] = (fix15_short_t)fix15_one; //src_pixel[3];
        }
        
        virtual void put_pixel(int rx, int ry, const fix15_short_t* pixel)
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

        static SquareDilateKernel<fix15_short_t> sq_kernel;
        static DiamondDilateKernel<fix15_short_t> dm_kernel;

    public:

        static const int SQUARE_KERNEL = 0;
        static const int DIAMOND_KERNEL = 1;

        _Dilater_fix15() : MorphologyBase(NPY_UINT16, 4)
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
            KernelFunctor<fix15_short_t>* functor;
            
            switch (kernel_type) {
                case DIAMOND_KERNEL:
                    functor = &dm_kernel;
                    break;

                case SQUARE_KERNEL:
                default:
                    functor = &sq_kernel;
                    break;
            }

           // for (int y=1; y < MYPAINT_TILE_SIZE; y++) {  // why 1?
            for (int y=0; y < MYPAINT_TILE_SIZE; y++) {
                for (int x=0; x < MYPAINT_TILE_SIZE; x++) {
                    fix15_short_t* cur_pixel = get_pixel(py_filled_tile, x, y);
                    if (cur_pixel[3] != 0)
                    {
                        // using cur_pixel is hazardous in some situation,
                        // so use fixed pixel color.
                        (*functor)(this, x, y, size, fill_pixel);
                        dilated_cnt++;
                    }
                }
            }
            
            return dilated_cnt;
        }

};
              
SquareDilateKernel<fix15_short_t> _Dilater_fix15::sq_kernel;
DiamondDilateKernel<fix15_short_t> _Dilater_fix15::dm_kernel;

// _Morphology_contour : Specialized class for detecting contour, to fill-gap.
// This is for 16bit,flag_based erode/dilate operation.
//
// in new flood_fill(), we use tiles of 8bit bitflag status to figure out 
// original colored not-filled pixels(EXIST_MASK) , 
// dilated pixels(DILATED_MASK) and
// eroded contour pixels(ERODED_MASK).

class _Morphology_contour: public MorphologyBase<short> {

    protected:
        // pixel status information flag.
        static const short EXIST_MASK = 0x0001;
        static const short DILATED_MASK = 0x0002; // This means the pixel is just dilated pixel,
                                               // not sure original source pixel.
                                               // it might be empty pixel in source tile.
        static const short ERODED_MASK = 0x0004;

        // bit flags for skelton morphology 
        static const short SKELTON_ERODED_MASK = 0x0008;
        static const short SKELTON_DILATED_MASK = 0x0010;
        static const short SKELTON_RESULT_MASK = 0x0020;
        static const short SKELTON_BASE_MASK = 0x0040;

        static const short PROCESSED_MASK = 0x0100; // This means 'This pixel has original source
                                                    // contour pixel, and dilated'.
        static const short SKELTON_PROCESSED_MASK = 0x0200; // This is skelton processed mask.

        // tile status information flag.
        // This flag is only set to paticular location of tile pixels.
        static const short TILE_INFO_MASK = 0x8000;

        static const short DILATED_TILE_MASK = 0x01;
        static const short ERODED_TILE_MASK = 0x02;
        static const short SKELTON_TILE_MASK = 0x04;
        static const int TILE_INFO_MAX = 3; // change this when new tile info flag created.

        // - kernel functors
        static SquareDilateKernel<short> s_sq_dilate_kernel;
        static DiamondDilateKernel<short> s_dm_dilate_kernel;
        static DiamondErodeKernel<short> s_dm_erode_kernel;

        class SkeltonFinishKernel: public KernelFunctor<short> {
                
            protected:

                bool vert_check(MorphologyBase<short>* mb, int cx, int cy)
                {
                    // checking below 3 pixels are blank or not
                    for(int i=-1; i <= 1 ; i++) {
                        short* cur_pixel = mb->get_cached_pixel(cx+i, cy+1, false);
                        if (cur_pixel != NULL && (*cur_pixel & SKELTON_RESULT_MASK) != 0) {
                            return false; // not blank. exit.
                        } 
                    }

                    // and, 1 pixel further 3 pixels exist or not
                    for(int i=-1; i <= 1 ; i++) {
                        short* cur_pixel = mb->get_cached_pixel(cx+i, cy+2, false);
                        if (cur_pixel != NULL && (*cur_pixel & SKELTON_RESULT_MASK) != 0) {
                            return true;
                        } 
                    }

                    return false;
                }

                bool horz_check(MorphologyBase<short>* mb, int cx, int cy)
                {
                    // checking right 3 pixels are blank or not
                    for(int i=-1; i <= 1 ; i++) {
                        short* cur_pixel = mb->get_cached_pixel(cx+1, cy+i, false);
                        if (cur_pixel != NULL && (*cur_pixel & SKELTON_RESULT_MASK) != 0) {
                            return false; // not blank. exit.
                        } 
                    }

                    // and, 1 pixel further 3 pixels exist or not
                    for(int i=-1; i <= 1 ; i++) {
                        short* cur_pixel = mb->get_cached_pixel(cx+2, cy+i, false);
                        if (cur_pixel != NULL && (*cur_pixel & SKELTON_RESULT_MASK) != 0) {
                            return true;
                        } 
                    }

                    return false;
                }

            public:

                virtual int operator()(MorphologyBase<short>* mb, int cx, int cy, int size, const short* pixel)
                {
                    int ret = 0;
                    if ((*pixel & SKELTON_RESULT_MASK) != 0) {

                        if ( vert_check(mb, cx, cy) ) {
                            mb->put_pixel(cx, cy+1, SKELTON_RESULT_MASK);
                            ret++;
                        }
                        else if (horz_check(mb, cx, cy) ) {
                            mb->put_pixel(cx+1, cy, SKELTON_RESULT_MASK);
                            ret++;
                        }

                    }
                    /*
                    // removing generic eroded flags from center tile 
                    if ( 0 <= cx && cx < MYPAINT_TILE_SIZE &&
                         0 <= cy && cy < MYPAINT_TILE_SIZE ) {
                        short* cur_pixel = (short*)pixel;
                       // *cur_pixel &= (~ERODED_MASK);
                    }
                    */
                    return ret;
                }
        };

        static SkeltonFinishKernel s_finish_kernel;



        // - pixel processing functors.
        // These functors used for skelton morphology, to deal with bitflags.
        class Functor_convert_erodedflag : public PixelFunctor {
            public:
                int operator()(MorphologyBase<short>* mb, short* pixel, int x, int y)
                {
                    if (pixel != NULL) {
                        if (*pixel & ERODED_MASK) {
                            *pixel |= SKELTON_BASE_MASK; // set new dilated mask
                            return 1;
                        }
                        else {
                            *pixel &= (~SKELTON_BASE_MASK);// clear them (although there is no such flags in most cases...)
                        }
                    }
                    return 0;
                }
        };

        class Functor_convert_resultflag : public PixelFunctor {
            public:
                int operator()(MorphologyBase<short>* mb,short* pixel, int x, int y)
                {
                    int cnt = 0;
                    if (pixel != NULL) {
                        if ((*pixel & SKELTON_DILATED_MASK) == 0 && (*pixel & SKELTON_BASE_MASK) != 0)
                        {
                            // skelton pixel == not dilated, but original area
                            *pixel |= SKELTON_RESULT_MASK;
                            cnt = 1;
                        }
                        *pixel &= (~(SKELTON_ERODED_MASK|SKELTON_DILATED_MASK)); // clear skelton work masks
                    }
                    return cnt;
                }
        };

        class Functor_convert_back : public PixelFunctor {
            public:
                int operator()(MorphologyBase<short>* mb,short* pixel, int x, int y)
                {
                    int cnt = 0;
                    if (pixel != NULL) {
                        if ((*pixel & SKELTON_ERODED_MASK) == 0) {
                            *pixel &= (~SKELTON_BASE_MASK);
                        }
                        else {
                            *pixel |= SKELTON_BASE_MASK;
                            cnt = 1;
                        }
                        *pixel &= (~SKELTON_ERODED_MASK); // remove work masks
                    }
                    return cnt;
                }
        };

        /*
        class Functor_convert_final : public PixelFunctor {
            public:
                int operator()(MorphologyBase<short>* mb,short* pixel, int x, int y)
                {
                    if (pixel != NULL) {
                        if ((*pixel & SKELTON_DILATED_MASK) != 0) 
                            *pixel |= SKELTON_RESULT_MASK;
                        *pixel &= (~SKELTON_DILATED_MASK); // remove work masks
                    }
                    return 0;
                }
        };
        */

        class Functor_dilate : public PixelFunctor {
            public:
                short flag;
                int size;

                int operator()(MorphologyBase<short>* mb,short* pixel, int x, int y)
                {
                    if (pixel != NULL && (*pixel & EXIST_MASK) != 0 && (*pixel & PROCESSED_MASK) == 0) {
                        s_sq_dilate_kernel(mb, x, y, size, &flag);
                        *pixel |= PROCESSED_MASK;
                    }
                    return 0;
                }
        };

        class Functor_dilate_skelton : public PixelFunctor {
            public:
                short target_flag;
                KernelFunctor<short>* kernel;  // kernel selectable
                short flag;
                int size;

                int operator()(MorphologyBase<short>* mb,short* pixel, int x, int y)
                {
                    if (pixel != NULL && (*pixel & target_flag) != 0) {
                        (*kernel)(mb, x, y, size, &flag);
                    }
                    return 0;
                }
        };

        class Functor_erode: public PixelFunctor {
            public:
                short target_flag;
                short flag;
                int size;

                int operator()(MorphologyBase<short>* mb,short* pixel, int x, int y)
                {
                    if (pixel != NULL && (*pixel & target_flag) != 0) {
                        s_dm_erode_kernel(mb, x, y, size, &flag, &target_flag);
                    }
                    return 0;
                }
        };

        // Filter functor. enumerate all pixels.
        class Functor_filter: public PixelFunctor {
            public:
                KernelFunctor<short>* kernel;  // selectable kernel for filter.

                int operator()(MorphologyBase<short>* mb,short* pixel, int x, int y)
                {
                    if (pixel != NULL)
                    {
                        (*kernel)(mb, x, y, 0, pixel);
                    }
                    return 0;
                }
        };

        fix15_t m_tolerance;
        short m_erode_target;

        
        // And, we can get such special infomation from _get_tile_info() method
        // for future expanision.
        
        // In this class, all writing operation is not writing,
        // but bitwise compositing.
        virtual void put_pixel(short* dst_pixel, const short* src_pixel)
        {
            *dst_pixel |= *src_pixel;
        }

        // specialized utility method for this class.
        void put_pixel(short* dst_pixel, const short src_flag)
        {
            *dst_pixel |= src_flag;
        }

        virtual void put_pixel(PyObject* dst_tile,int x, int y,const short* pixel)
        {
            short* dst_pixel = get_pixel(dst_tile, x, y);
            *dst_pixel |= *pixel;
        } 

        virtual void put_pixel(int rx, int ry, const short* pixel)
        {
            short* dst_pixel = get_cached_pixel(rx, ry, true);
            *dst_pixel |= *pixel;
        } 

        virtual void put_pixel(int rx, int ry, short pixel)
        {
            short* dst_pixel = get_cached_pixel(rx, ry, true);
            *dst_pixel |= pixel;
        }

        virtual bool is_equal_cached_pixel(int rx, int ry, const short* pixel)
        {
            short* targ_pixel = get_cached_pixel(rx, ry, false); 
            if (targ_pixel != NULL)
                return (*targ_pixel & *pixel) != 0;
            return false;
        }

        // - special information methods
        //
        // Special informations recorded into a tile with
        // setting INFO_MASK bitflag to paticular pixel.
        // And they have various means,according to its location. 
        
        char _get_tile_info(PyObject* tile)
        {
            char retflag = 0;
            char flag = 1;
            short* pixel;
            for(int i=0; i < TILE_INFO_MAX; i++) {
                pixel = get_pixel(tile, 0, i);
                if (*pixel & TILE_INFO_MASK)
                    retflag |= flag;
                flag = flag << 1;
            }
            return retflag;
        }

        void _set_tile_info(PyObject* tile, char flag)
        {
            short* pixel;
            for(int i=0; i < TILE_INFO_MAX && flag != 0; i++) {
                pixel = get_pixel((PyObject*)tile, 0, i);
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
        void _dilate_contour(int gap_size)    // growing size from center pixel.
        {

            Functor_dilate func;
            func.size = gap_size;
            func.flag = DILATED_MASK;

            _iterate_pixel(func, gap_size);

            PyObject* target_tile = _get_cached_tile_from_index(CENTER_TILE_INDEX, true);
            _set_tile_info(target_tile, DILATED_TILE_MASK);
            
        }

        void _erode_contour(int gap_size)    // growing size from center pixel.
        {
            
            // Only center tile should be eroded.
            PyObject* target_tile = _get_cached_tile_from_index(CENTER_TILE_INDEX, false);
#ifdef HEAVY_DEBUG
            assert(target_tile != NULL);
#endif

            char tile_info = _get_tile_info(target_tile);

            if ((tile_info & ERODED_TILE_MASK) != 0)
                return;

#ifdef HEAVY_DEBUG
            assert((tile_info & DILATED_TILE_MASK) != 0);
#endif
            
            // to failsafe, limit grow size within MYPAINT_MYPAINT_TILE_SIZE
            Functor_erode func;
            func.target_flag = DILATED_MASK;
            func.flag = ERODED_MASK;
            func.size = gap_size;

            _iterate_pixel(func, gap_size);
            
            _set_tile_info(target_tile, ERODED_TILE_MASK);
        }

        // Apply skelton morphology (might slow!)
        void _skelton_contour(int gap_size)
        {
            
            // check whether this (center) tile is already 'skeltoned'
            PyObject* target_tile = _get_cached_tile_from_index(CENTER_TILE_INDEX, false);
#ifdef HEAVY_DEBUG
            assert(target_tile != NULL);
#endif
            char tile_info = _get_tile_info(target_tile);
            if ((tile_info & SKELTON_TILE_MASK) != 0)
               return;

            // Functors
            Functor_convert_erodedflag func_conv_eroded;
            Functor_convert_resultflag func_conv_result;
            Functor_erode func_erode;
            func_erode.size = 1;

            Functor_dilate_skelton func_dilate;
            func_dilate.target_flag = SKELTON_ERODED_MASK;
            func_dilate.kernel = &s_dm_dilate_kernel;
            func_dilate.size = 1;

            Functor_convert_back func_conv_back;

            // Initialize, convert eroded bits ERODED_MASK to SKELTON_BASE_MASK. 
            // we need to keep original eroded area because we cannot keep
            // consistency of eroded area over multiple tiles when it is
            // decreased(further eroded) in flood-fill loop.
            // if we lose consistency of eroded state, some tiles get over
            // eroded and fail to produce correct skelton structure.
            _iterate_pixel(func_conv_eroded, gap_size);

            // Skelton loop : maximum MYPAINT_TILE_SIZE count.
            // it should be enough in any case and most case,
            // we'll exit this loop much less count.
            for(int cnt=0; cnt < MYPAINT_TILE_SIZE; cnt++) {
                
                // first of all, erode current eroded flag area.
                //m_source_pixel = &work_pixel;
                func_erode.flag = SKELTON_ERODED_MASK;
                func_erode.target_flag = SKELTON_BASE_MASK;
                _iterate_pixel(func_erode, gap_size);
                

                // change m_source_pixel to WORK2_MASK flag.
                // because, if dilating pixel and its source is same,
                // dilation routines respond its own wrote pixels. 
                //m_source_pixel = &work2_pixel;
                func_dilate.flag = SKELTON_DILATED_MASK;
                _iterate_pixel(func_dilate, gap_size);

                // such erosion -> dilation operation called as 'open', in Morphology.

                // then, mark SKELTON_MASK to pixels which is 
                // (normal_eroded flagged area & not opened_area)
                _iterate_pixel(func_conv_result, gap_size);
                
                // finally, erode current SKELTON_BASE_MASK area
                // into SKELTON_ERODED_MASK
                //m_source_pixel = &work_pixel;
                func_erode.flag = SKELTON_ERODED_MASK;
                _iterate_pixel(func_erode, gap_size);

                
                // then, convert back eroded SKELTON_ERODED_MASK into 
                // SKELTON_BASE_MASK. and loop it again 
                // until there are unprocessed pixels left.
                if (_iterate_pixel(func_conv_back, gap_size) == 0)
                    break;


            }

            
            // Loop ended.
            // we should get the result of skelton morphorogy.
            // But, pixel skelton mophology likely produce 1 pixel gap in some case. 
            // (We can see it even OpenCV samples)
            // It can mess up flood-fill overflow guard. so, connect them with
            // specialized kernel.

            Functor_filter func_filter;
            func_filter.kernel = &s_finish_kernel;
            _iterate_pixel(func_filter, gap_size);

            // Although already center tile has been set ERODED_TILE_MASK flag
            // in most case.
            target_tile = _get_cached_tile_from_index(CENTER_TILE_INDEX, true);
            _set_tile_info(target_tile, SKELTON_TILE_MASK);
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

            short exist_flag_pixel = EXIST_MASK;
            
            for (int py = 0; py < MYPAINT_TILE_SIZE; py++) {
                for (int px = 0; px < MYPAINT_TILE_SIZE; px++) {
                    fix15_short_t* cur_pixel = (fix15_short_t*)get_pixel((PyObject*)py_surf_tile, px, py);
                    if (floodfill_color_match(cur_pixel, targ_pixel, (const fix15_t)m_tolerance) == 0) {
                        put_pixel(state_tile, px, py, &exist_flag_pixel);
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
#endif

            PyObject* attr_name_rgba = Py_BuildValue("s","rgba");

            for (int i=0; i < 9; i++) {

                if (s_cache_tiles[i] == NULL) {
                    PyObject* key = _get_cached_key(i);
                    PyObject* surf_tile = PyDict_GetItem(py_surfdict, key);
                    if (surf_tile != NULL) {
                        Py_INCREF(surf_tile);

                        // Check whether it is readonly _Tile object or not.
                        int is_tile_obj = PyObject_HasAttr(surf_tile, attr_name_rgba);
                        if (is_tile_obj != 0) {
                            Py_DECREF(surf_tile);

                            surf_tile = PyObject_GetAttr(surf_tile,
                                                         attr_name_rgba);
                            Py_INCREF(surf_tile);
#ifdef HEAVY_DEBUG            
                            assert(surf_tile->ob_refcnt >= 1);
#endif
                        }

                        PyObject* state_tile = _convert_state_tile(py_statedict, 
                                                                   (PyArrayObject*)surf_tile, 
                                                                   targ_pixel, 
                                                                   key);
                        Py_DECREF(surf_tile);
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

#ifdef HEAVY_DEBUG            
            assert(s_cache_tiles[CENTER_TILE_INDEX] != NULL);
#endif
        }

    public:

        _Morphology_contour() : MorphologyBase(NPY_UINT16, 1),
                                m_erode_target(DILATED_MASK)
        {
        }

        void detect_contour(PyObject* py_statedict, // 8bit state tiles dict
                            PyObject* py_surfdict, // source surface tiles dict
                            int tx, int ty, // current tile location
                            const fix15_short_t* targ_pixel,// target pixel for conversion
                                                            // from color tile to status tile.
                            fix15_t tolerance,
                            int gap_size,
                            bool do_skelton) 
        {
            init_cached_tiles(py_statedict, tx, ty); 

            m_tolerance = tolerance;

            _setup_state_tiles(py_statedict, 
                               py_surfdict, 
                               targ_pixel);
        
            // detecting contour with dilate & erode.
            _dilate_contour(gap_size);
            
            _erode_contour(gap_size);

            if (do_skelton) 
                _skelton_contour(gap_size);

            finalize_cached_tiles(py_statedict); 
        }

        // XXX TEST CODES
        void test_skelton(PyObject* py_statedict, // 8bit state tiles dict
                            int tx, int ty, // current tile location
                            int gap_size
                            ) 
        {
            init_cached_tiles(py_statedict, tx, ty); 

            _skelton_contour(gap_size);

            finalize_cached_tiles(py_statedict); 
        }
        
};


SquareDilateKernel<short> _Morphology_contour::s_sq_dilate_kernel;
DiamondDilateKernel<short> _Morphology_contour::s_dm_dilate_kernel;
DiamondErodeKernel<short> _Morphology_contour::s_dm_erode_kernel;
_Morphology_contour::SkeltonFinishKernel _Morphology_contour::s_finish_kernel;
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
               double tol,   // pixel tolerance of filled area.
               int gap_size, // overflow-preventing closable gap size.
               int do_skelton) // use skelton morphology(slow)
{
#ifdef HEAVY_DEBUG            
    assert(py_statedict != NULL);
    assert(py_surfdict != NULL);
    assert(0.0 <= tol);
    assert(tol <= 1.0);
    assert(gap_size <= MYPAINT_TILE_SIZE / 2 - 1);
#endif
    // actually, this function is wrapper.
    
    // XXX Morphology object defined as static. 
    // Because this function called each time before a tile flood-filled,
    // so constructing/destructing cost would not be ignored.
    // Otherwise, we can use something start/end wrapper function and
    // use PyCapsule object...
    static _Morphology_contour m;


    fix15_short_t targ_pixel[4] = {(fix15_short_t)targ_r,
                                   (fix15_short_t)targ_g,
                                   (fix15_short_t)targ_b,
                                   (fix15_short_t)targ_a};

    // XXX Copied from fill.cpp tile_flood_fill()
    const fix15_t tolerance = (fix15_t)(  MIN(1.0, MAX(0.0, tol))
                                        * fix15_one);

    // limit gap size.
    gap_size %= (MYPAINT_TILE_SIZE / 2 - 1);

    m.detect_contour(py_statedict, py_surfdict,
                     tx, ty, 
                     (const fix15_short_t*)targ_pixel,
                     tolerance,
                     gap_size,
                     do_skelton != 0);

    Py_RETURN_NONE;// DO NOT FORGET THIS!
      
}

/** dilate_filled_tile:
 *
 * @py_dilated: a Python dictinary, which stores 'dilated color tiles'
 * @py_pre_filled_tile: a Numpy.array object of color tile.
 * @tx, ty: the position of py_pre_filled_tile, in tile coordinate.
 * @grow_size: the growing size of filled area, in pixel.
 * @kernel_type: the shape of dilation. if this is 0(==MorphologyBase<T>::SQUARE_KERNEL),
 *               pixels dilated into square shape.
 *               if this is 1 (==MorphologyBase<T>::DIAMOND_KERNEL), pixels dilated into
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

    //d.setup_morphology_params(dilate_size, &dummy_pixel);
    d.init_cached_tiles(py_dilated, tx, ty);
    d.dilate(py_filled_tile, tx, ty, fill_pixel, dilate_size, kernel_type);
    //d.dilate(py_filled_tile, tx, ty, dilate_size, kernel_type);
    d.finalize_cached_tiles(py_dilated);

    Py_RETURN_NONE;
}

// XXX TEST CODES
PyObject* test_skelton(PyObject* py_statedict, // the tiledict for dilated tiles.
                       int tx, int ty,  // the position of py_filled_tile
                       int gap_size  // overflow-preventing closable gap size.
                       )
{
    static _Morphology_contour m;

    m.test_skelton(py_statedict, 
                   tx, ty, 
                   gap_size);

    Py_RETURN_NONE;// DO NOT FORGET THIS!
}

