#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFile>
#include <QFileDialog>
#include <QtConcurrent>
#include <QThread>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class Worker : public QObject
{
    Q_OBJECT
public slots:
    // simulation related slots
    void runSim();
    void stopSim();
signals:
    // simulation related signals
    void resultReady(const QString &result);
};

class MainWindow : public QMainWindow
{
    Q_OBJECT
    QThread workerThread;
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
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
public slots:
    void handleResult(const QString &);
private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
