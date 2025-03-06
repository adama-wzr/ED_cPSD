/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.8.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../mainwindow.h"
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.8.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {
struct qt_meta_tag_ZN6WorkerE_t {};
} // unnamed namespace


#ifdef QT_MOC_HAS_STRINGDATA
static constexpr auto qt_meta_stringdata_ZN6WorkerE = QtMocHelpers::stringData(
    "Worker",
    "resultReady",
    "",
    "const QString*",
    "result",
    "enableButtons",
    "disableButtons"
);
#else  // !QT_MOC_HAS_STRINGDATA
#error "qtmochelpers.h not found or too old."
#endif // !QT_MOC_HAS_STRINGDATA

Q_CONSTINIT static const uint qt_meta_data_ZN6WorkerE[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    1,   32,    2, 0x06,    1 /* Public */,
       5,    0,   35,    2, 0x06,    3 /* Public */,
       6,    0,   36,    2, 0x06,    4 /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

Q_CONSTINIT const QMetaObject Worker::staticMetaObject = { {
    QMetaObject::SuperData::link<QThread::staticMetaObject>(),
    qt_meta_stringdata_ZN6WorkerE.offsetsAndSizes,
    qt_meta_data_ZN6WorkerE,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_tag_ZN6WorkerE_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<Worker, std::true_type>,
        // method 'resultReady'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<const QString *, std::false_type>,
        // method 'enableButtons'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'disableButtons'
        QtPrivate::TypeAndForceComplete<void, std::false_type>
    >,
    nullptr
} };

void Worker::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<Worker *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->resultReady((*reinterpret_cast< std::add_pointer_t<const QString*>>(_a[1]))); break;
        case 1: _t->enableButtons(); break;
        case 2: _t->disableButtons(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _q_method_type = void (Worker::*)(const QString * );
            if (_q_method_type _q_method = &Worker::resultReady; *reinterpret_cast<_q_method_type *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
        {
            using _q_method_type = void (Worker::*)();
            if (_q_method_type _q_method = &Worker::enableButtons; *reinterpret_cast<_q_method_type *>(_a[1]) == _q_method) {
                *result = 1;
                return;
            }
        }
        {
            using _q_method_type = void (Worker::*)();
            if (_q_method_type _q_method = &Worker::disableButtons; *reinterpret_cast<_q_method_type *>(_a[1]) == _q_method) {
                *result = 2;
                return;
            }
        }
    }
}

const QMetaObject *Worker::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Worker::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ZN6WorkerE.stringdata0))
        return static_cast<void*>(this);
    return QThread::qt_metacast(_clname);
}

int Worker::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void Worker::resultReady(const QString * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void Worker::enableButtons()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void Worker::disableButtons()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}
namespace {
struct qt_meta_tag_ZN10MainWindowE_t {};
} // unnamed namespace


#ifdef QT_MOC_HAS_STRINGDATA
static constexpr auto qt_meta_stringdata_ZN10MainWindowE = QtMocHelpers::stringData(
    "MainWindow",
    "updateFileText",
    "",
    "togglePartLabel2D",
    "togglePoreLabel2D",
    "saveInput2D",
    "clearText2D",
    "updateFileText3D",
    "toggleInput3D",
    "togglePartLabel3D",
    "togglePoreLabel3D",
    "saveInput3D",
    "clearText3D",
    "clearTextRun",
    "runSim",
    "findOpFolder",
    "handleFinish",
    "disableButtons",
    "enableButtons"
);
#else  // !QT_MOC_HAS_STRINGDATA
#error "qtmochelpers.h not found or too old."
#endif // !QT_MOC_HAS_STRINGDATA

Q_CONSTINIT static const uint qt_meta_data_ZN10MainWindowE[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
      17,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    0,  116,    2, 0x08,    1 /* Private */,
       3,    0,  117,    2, 0x08,    2 /* Private */,
       4,    0,  118,    2, 0x08,    3 /* Private */,
       5,    0,  119,    2, 0x08,    4 /* Private */,
       6,    0,  120,    2, 0x08,    5 /* Private */,
       7,    0,  121,    2, 0x08,    6 /* Private */,
       8,    0,  122,    2, 0x08,    7 /* Private */,
       9,    0,  123,    2, 0x08,    8 /* Private */,
      10,    0,  124,    2, 0x08,    9 /* Private */,
      11,    0,  125,    2, 0x08,   10 /* Private */,
      12,    0,  126,    2, 0x08,   11 /* Private */,
      13,    0,  127,    2, 0x08,   12 /* Private */,
      14,    0,  128,    2, 0x08,   13 /* Private */,
      15,    0,  129,    2, 0x08,   14 /* Private */,
      16,    0,  130,    2, 0x08,   15 /* Private */,
      17,    0,  131,    2, 0x08,   16 /* Private */,
      18,    0,  132,    2, 0x08,   17 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

Q_CONSTINIT const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_ZN10MainWindowE.offsetsAndSizes,
    qt_meta_data_ZN10MainWindowE,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_tag_ZN10MainWindowE_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<MainWindow, std::true_type>,
        // method 'updateFileText'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'togglePartLabel2D'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'togglePoreLabel2D'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'saveInput2D'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'clearText2D'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'updateFileText3D'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'toggleInput3D'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'togglePartLabel3D'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'togglePoreLabel3D'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'saveInput3D'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'clearText3D'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'clearTextRun'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'runSim'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'findOpFolder'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'handleFinish'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'disableButtons'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        // method 'enableButtons'
        QtPrivate::TypeAndForceComplete<void, std::false_type>
    >,
    nullptr
} };

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<MainWindow *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->updateFileText(); break;
        case 1: _t->togglePartLabel2D(); break;
        case 2: _t->togglePoreLabel2D(); break;
        case 3: _t->saveInput2D(); break;
        case 4: _t->clearText2D(); break;
        case 5: _t->updateFileText3D(); break;
        case 6: _t->toggleInput3D(); break;
        case 7: _t->togglePartLabel3D(); break;
        case 8: _t->togglePoreLabel3D(); break;
        case 9: _t->saveInput3D(); break;
        case 10: _t->clearText3D(); break;
        case 11: _t->clearTextRun(); break;
        case 12: _t->runSim(); break;
        case 13: _t->findOpFolder(); break;
        case 14: _t->handleFinish(); break;
        case 15: _t->disableButtons(); break;
        case 16: _t->enableButtons(); break;
        default: ;
        }
    }
    (void)_a;
}

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ZN10MainWindowE.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 17)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 17;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 17)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 17;
    }
    return _id;
}
QT_WARNING_POP
