#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFile>
#include <QFileDialog>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include "data_structs.hpp"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

// worker thread class definition
class Worker : public QThread
{
    Q_OBJECT
public:
    // constructor and destructor
    Worker(QObject *parent = nullptr);
    ~Worker();
    // function call to run
    void runSim(const QString string);
    // Simulation related functions
    int Sim2D_ui(options* opts);
    int Sim3D_ui(options* opts);
    // pore and part 2D
    int pore2D_SD_ui(options *opts,
                     sizeInfo2D *info,
                     char *P,
                     char POI);
    int part2D_SD_ui(options *opts,
                     sizeInfo2D *info,
                     char *P,
                     char POI);
    // pore and part 3D
    int part3D_SD_ui(options *opts,
                     sizeInfo *info,
                     char *P,
                     int POI);
    int pore3D_SD_ui(options *opts,
                     sizeInfo *info,
                     char *P,
                     int POI);
signals:
    void resultReady(const QString *result);
    void enableButtons();
    void disableButtons();
protected:
    void run() override;
private:
    QMutex mutex;
    QWaitCondition condition;
    bool restart = false;
    bool abort = false;
    QString foldername;
};

// main window class definition

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    // ~MainWindow();
private slots:
    // 2D slots
    void updateFileText();
    void togglePartLabel2D();
    void togglePoreLabel2D();
    void saveInput2D();
    void clearText2D();
    // 3D slots
    void updateFileText3D();
    void toggleInput3D();
    void togglePartLabel3D();
    void togglePoreLabel3D();
    void saveInput3D();
    void clearText3D();
    // Run slots
    void clearTextRun();
    void runSim();
    void findOpFolder();
    void handleFinish();
    void disableButtons();
    void enableButtons();  
private:
    void handleResult(const QString *string);
    Ui::MainWindow *ui;
    Worker workerThread;
    bool restart;
    bool abort;
    QString foldername;
};

#endif // MAINWINDOW_H
