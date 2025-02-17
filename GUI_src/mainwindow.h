#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFile>
#include <QFileDialog>
#include <QtConcurrent>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

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
    void runSim();
    void findOpFolder();
    void handleFinish();
private:
    Ui::MainWindow *ui;
    QFutureWatcher<int> watcher;
    QFuture<int> future;
};

#endif // MAINWINDOW_H
