#!/usr/bin/env python

from __future__ import division, print_function

from time import time
import sys
import os
import gc

import numpy as np

os.chdir(os.path.dirname(__file__))
sys.path.insert(0, '..')

from lib import mypaintlib, tiledsurface, brush, document, command, helpers
import guicontrol

# loadtxt is known to leak memory, thus we run it only once
# http://projects.scipy.org/numpy/ticket/1356
painting30sec_events = np.loadtxt('painting30sec.dat')

LEAK_EXIT_CODE = 33


def mem():
    gc.collect()
    with open('/proc/self/statm') as statm:
        return int(statm.read().split()[0])


def check_garbage(msg='uncollectable garbage left over from previous tests'):
    gc.collect()
    garbage = []
    for obj in gc.garbage:
        # ignore garbage generated by numpy loadtxt command
        if hasattr(obj, 'filename') and obj.filename == 'painting30sec.dat':
            continue
        garbage.append(obj)
    assert not garbage, 'uncollectable garbage left over from previous tests: %s' % garbage


def iterations():
    check_garbage()

    max_mem = 0
    max_mem_stable = 0
    max_mem_increasing = 0
    leak = True
    m1 = 0
    for i in range(options.max_iterations):
        yield i
        if options.debug:
            if i == 3:
                check_garbage()
                helpers.record_memory_leak_status()
            if i == 4 or i == 5:
                helpers.record_memory_leak_status(print_diff=True)
        m2 = mem()
        print('iteration %02d/%02d: %d pages used (%+d)' % (
            i + 1,
            options.max_iterations,
            m2,
            m2 - m1))
        m1 = m2
        if m2 > max_mem:
            max_mem = m2
            max_mem_stable = 0
            max_mem_increasing += 1
            if max_mem_increasing == options.required:
                print('maximum was always increasing for', max_mem_increasing,
                      'iterations')
                break
        else:
            max_mem_stable += 1
            max_mem_increasing = 0
            if max_mem_stable == options.required:
                print('maximum was stable for', max_mem_stable, 'iterations')
                leak = False
                break

    check_garbage()

    if leak:
        print('memory leak found')
        sys.exit(LEAK_EXIT_CODE)
    else:
        print('no leak found')

all_tests = {}


def leaktest(f):
    "decorator to declare leak test functions"
    all_tests[f.__name__] = f
    return f


#@leaktest
def provoke_leak():
    for i in iterations():
        # note: interestingly this leaky only shows in the later iterations
        #       (and very small leaks might not be detected)
        setattr(gc, 'my_test_leak_%d' % i, np.zeros(50000))


@leaktest
def noleak():
    for i in iterations():
        setattr(gc, 'my_test_leak', np.zeros(50000))


@leaktest
def document_alloc():
    for i in iterations():
        doc = document.Document()
        doc.cleanup()


@leaktest
def surface_alloc():
    for i in iterations():
        tiledsurface.Surface()


def paint_doc(doc):
    events = painting30sec_events
    t_old = events[0][0]
    layer = doc.layer_stack.current
    for i, (t, x, y, pressure) in enumerate(events):
        dtime = t - t_old
        t_old = t
        layer.stroke_to(doc.brush, x, y, pressure, 0.0, 0.0, dtime)


@leaktest
def save_test():
    doc = document.Document()
    paint_doc(doc)
    for i in iterations():
        doc.save('test_leak.ora')
        doc.save('test_leak.png')
        doc.save('test_leak.jpg')
    doc.cleanup()


@leaktest
def repeated_loading():
    doc = document.Document()
    for i in iterations():
        doc.load('bigimage.ora')
    doc.cleanup()


@leaktest
def paint_save_clear():
    doc = document.Document()
    for i in iterations():
        paint_doc(doc)
        doc.save('test_leak.ora')
        doc.clear()
    doc.cleanup()


def paint_gui(gui):
    """
    Paint with a constant number of frames per recorded second.
    Not entirely realistic, but gives good and stable measurements.
    """
    FPS = 30
    gui_doc = gui.app.doc
    model = gui_doc.model
    tdw = gui_doc.tdw

    b = gui.app.brushmanager.get_brush_by_name('redbrush')
    gui.app.brushmanager.select_brush(b)

    events = list(painting30sec_events)
    t_old = 0.0
    t_last_redraw = 0.0
    for t, x, y, pressure in events:
        if t > t_last_redraw + 1.0/FPS:
            gui.wait_for_gui()
            t_last_redraw = t
        dtime = t - t_old
        t_old = t
        x, y = tdw.display_to_model(x, y)
        gui_doc.modes.top.stroke_to(model, dtime, x, y, pressure, 0.0, 0.0)


@leaktest
def gui_test():
    # NOTE: this an all-in-one GUI test as a workaround for the
    # problem that the GUI does not cleanly terminate after the test fork()
    gui = guicontrol.GUI()
    gui.wait_for_idle()
    gui.app.filehandler.open_file(u'bigimage.ora')
    gui_doc = gui.app.doc
    for i in iterations():
        gui.app.filehandler.open_file(u'smallimage.ora')
        gui.wait_for_idle()
        paint_gui(gui)
        gui.app.filehandler.save_file(u'test_save.ora')
        gui.scroll()
        gui_doc.zoom(gui_doc.ZOOM_OUTWARDS)
        gui.scroll()
        gui_doc.zoom(gui_doc.ZOOM_INWARDS)

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    from optparse import OptionParser
    parser = OptionParser('usage: %prog [options] [test1 test2 test3 ...]')
    parser.add_option(
        '-a',
        '--all',
        action='store_true',
        default=False,
        help='run all tests'
    )
    parser.add_option(
        '-l',
        '--list',
        action='store_true',
        default=False,
        help='list all available tests'
    )
    parser.add_option(
        '-d',
        '--debug',
        action='store_true',
        default=False,
        help='print leak analysis (slow)'
    )
    parser.add_option(
        '-e',
        '--exit',
        action='store_true',
        default=False,
        help='exit at first error'
    )
    parser.add_option(
        '-r',
        '--required',
        type='int',
        default=15,
        help='iterations required to draw a conclusion (default: 15)'
    )
    parser.add_option(
        '-m',
        '--max-iterations',
        type='int',
        default=100,
        help='maximum number of iterations (default: 100)'
    )
    options, tests = parser.parse_args()

    if options.list:
        for name in sorted(all_tests.keys()):
            print(name)
        sys.exit(0)

    if options.required >= options.max_iterations:
        print('requiring more good iterations than the iteration limit makes '
              'no sense')
        sys.exit(1)

    if not tests:
        if options.all:
            tests = list(all_tests)
        else:
            parser.print_help()
            sys.exit(1)

    for t in tests:
        if t not in all_tests:
            print('Unknown test:', t)
            sys.exit(1)

    results = []
    for t in tests:
        child_pid = os.fork()
        if not child_pid:
            print('---')
            print('running test "%s"' % t)
            print('---')
            all_tests[t]()
            sys.exit(0)

        pid, status = os.wait()
        exitcode = os.WEXITSTATUS(status)
        if options.exit and exitcode != 0:
            sys.exit(1)
        results.append(exitcode)

    everything_okay = True
    print()
    print('=== SUMMARY ===')
    for t, exitcode in zip(tests, results):
        if exitcode == 0:
            print(t, 'OK')
        else:
            everything_okay = False
            if exitcode == LEAK_EXIT_CODE:
                print(t, 'LEAKING')
            else:
                print(t, 'EXCEPTION')
    if not everything_okay:
        sys.exit(1)