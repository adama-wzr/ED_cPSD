#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QThread>
#include <QtConcurrent>
#include "ED_PSD_CPU.hpp"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    // gen button
    connect(ui->GenButton2D,&QPushButton::clicked, this, &MainWindow::updateFileText);
    // gen button 3D
    connect(ui->GenButton3D,&QPushButton::clicked, this, &MainWindow::updateFileText3D);
    // toggle 2D pore/part labels
    connect(ui->partSD,&QCheckBox::checkStateChanged, this, &MainWindow::togglePartLabel2D);
    connect(ui->poreSD,&QCheckBox::checkStateChanged, this, &MainWindow::togglePoreLabel2D);
    // toggle 3D pore/part labels
    connect(ui->partSD_3D, &QCheckBox::checkStateChanged, this, &MainWindow::togglePartLabel3D);
    connect(ui->poreSD_3D, &QCheckBox::checkStateChanged, this, &MainWindow::togglePoreLabel3D);
    // toggle Input Type 3D
    connect(ui->inputType3D, &QComboBox::currentIndexChanged, this, &MainWindow::toggleInput3D);
    // save input 2D
    connect(ui->saveButton2D, &QPushButton::clicked, this, &MainWindow::saveInput2D);
    // save input 3D
    connect(ui->saveButton3D, &QPushButton::clicked, this, &MainWindow::saveInput3D);
    // Find operating folder
    connect(ui->searchFolder, &QPushButton::clicked, this, &MainWindow::findOpFolder);
    // connect future watcher with future
    connect(&watcher, &QFutureWatcher<int>::finished, this, &MainWindow::handleFinish);
    // connect clear button 2D
    connect(ui->clearButton2D, &QPushButton::clicked, this, &MainWindow::clearText2D);
    // connect clear button 3D
    connect(ui->clearButton3D, &QPushButton::clicked, this, &MainWindow::clearText3D);
    // Run code
    connect(ui->runButton, &QPushButton::clicked, this, &MainWindow::runSim);
}

int simOpts(const QString &string)
{
    options opts;

    opts.folderName = (char *)malloc(sizeof(char)*1000);
    strcpy(opts.folderName, string.toStdString().c_str());

    char inputTextFile[100];
    sprintf(inputTextFile, "input.txt");

    bool fileFlag = false;

    std::filesystem::path dir (opts.folderName);
    std::filesystem::path file (inputTextFile);
    std::filesystem::path full_path = dir / file;

    if(FILE *file = fopen(full_path.generic_string().c_str(), "r")){
        fclose(file);
        fileFlag = true;
    }else
    {
        fileFlag = false;
    }

    if (!fileFlag)
    {
        qInfo("Could not find input file. Exiting now.");
        return 1;
    }

    readInput(inputTextFile, &opts);

    if(opts.verbose)
    {
        printOpts(&opts);
    }

    if(opts.nD == 2)
    {
        Sim2D(&opts);
    }
    else if(opts.nD == 3)
    {
        Sim3D(&opts);
    }

    free(opts.folderName);
    return 0;
}

// Run sim

void MainWindow::runSim()
{

    ui->runtimeMessages->setText("Attempting to run simulation.\n");

    char* runtimeFolder = (char *)malloc(sizeof(char)*1000);

    if (ui->opFolder->text().isEmpty())
    {
        ui->runtimeMessages->append("Error: no operating folder selected!\n");
        free(runtimeFolder);
        return;
    }else
    {
        strcpy(runtimeFolder, ui->opFolder->text().toStdString().c_str());
        ui->runtimeMessages->append("Folder: " + ui->opFolder->text() + "/\n");
    }

    ui->runtimeMessages->append("Simulation running, please wait.\n");

    // disable buttons and text

    ui->runButton->setEnabled(false);
    ui->stopButton->setEnabled(true);

    ui->searchFolder->setEnabled(false);
    ui->opFolder->setReadOnly(true);

    // QFutureWatcher<int> watcher;
    // connect the watcher with handleFinish function
    // connect(&watcher, &QFutureWatcher<int>::finished, this, &MainWindow::handleFinish);

    // QFuture<int> future = QtConcurrent::run(simOpts, ui->opFolder->text());
    future = QtConcurrent::run(simOpts, ui->opFolder->text());
    // watch the concurrent execution
    watcher.setFuture(future);

    free(runtimeFolder);
    return;
}

void MainWindow::handleFinish()
{
    int setStatus = future.result();
    ui->runButton->setEnabled(true);
    ui->stopButton->setEnabled(false);

    ui->searchFolder->setEnabled(true);
    ui->opFolder->setReadOnly(false);

    if (setStatus == 0)
    {
        ui->runtimeMessages->append("Execution Finished Successfully!");
    }else
    {
        ui->runtimeMessages->append("Execution Finished with error.");
        ui->runtimeMessages->append("Check command line output for more details.");
    }


    return;
}

// Find operating folder

void MainWindow::findOpFolder()
{
    QString opFolderName = QFileDialog::getExistingDirectory(this);
    ui->opFolder->setText(opFolderName);
    return;
}

// save input file 2D

void MainWindow::saveInput2D()
{
    QString folderName = QFileDialog::getExistingDirectory(this);
    QFile myFile(folderName + "\\input.txt");
    if(myFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        myFile.write(ui->textDisplay2D->toPlainText().toStdString().c_str());
    }
    myFile.close();
    return;
}

void MainWindow::clearText2D()
{
    ui->textDisplay2D->clear();
    return;
}

// save input file 3D

void MainWindow::saveInput3D()
{
    QString folderName = QFileDialog::getExistingDirectory(this);
    QFile myFile(folderName + "\\input.txt");
    if(myFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        myFile.write(ui->textDisplay3D->toPlainText().toStdString().c_str());
    }
    myFile.close();
    return;
}

void MainWindow::clearText3D()
{
    ui->textDisplay3D->clear();
    return;
}

// Toggle Input Type 3D

void MainWindow::toggleInput3D()
{
    int index = 0;
    if(ui->inputType3D->currentText() == "Stack .jpg")
        index = 0;
    else if(ui->inputType3D->currentText() == ".csv")
        index = 1;
    ui->stackedWidget_input3D->setCurrentIndex(index);
    return;
}


// Toggle Part/Pore Label 2D

void MainWindow::togglePoreLabel2D()
{
    if(ui->poreSD->isChecked())
    {
        ui->poreLabel->setCheckable(true);
        ui->poreLabel_Out->setReadOnly(false);
        ui->poreSD_Out->setReadOnly(false);
    } else
    {
        ui->poreLabel->setCheckable(false);
        ui->poreLabel_Out->setReadOnly(true);
        ui->poreSD_Out->setReadOnly(true);
    }
    return;
}

void MainWindow::togglePoreLabel3D()
{
    if(ui->poreSD_3D->isChecked())
    {
        ui->poreLabel_3D->setCheckable(true);
        ui->poreLabel_Out3D->setReadOnly(false);
        ui->poreSD_Out3D->setReadOnly(false);
    } else
    {
        ui->poreLabel_3D->setCheckable(false);
        ui->poreLabel_Out3D->setReadOnly(true);
        ui->poreSD_Out3D->setReadOnly(true);
    }
    return;
}

void MainWindow::togglePartLabel2D()
{
    if(ui->partSD->isChecked())
    {
        ui->partLabel->setCheckable(true);
        ui->partLabel_Out->setReadOnly(false);
        ui->partSD_Out->setReadOnly(false);
    } else
    {
        ui->partLabel->setCheckable(false);
        ui->partLabel_Out->setReadOnly(true);
        ui->partSD_Out->setReadOnly(true);
    }
    return;
}

void MainWindow::togglePartLabel3D()
{
    if(ui->poreSD_3D->isChecked())
    {
        ui->partLabel_3D->setCheckable(true);
        ui->partLabel_Out3D->setReadOnly(false);
        ui->partSD_Out3D->setReadOnly(false);
    } else
    {
        ui->partLabel_3D->setCheckable(false);
        ui->partLabel_Out3D->setReadOnly(true);
        ui->partSD_Out3D->setReadOnly(true);
    }
    return;
}


// Update text

void MainWindow::updateFileText3D()
{
    // header
    QString header = "Input Parameters:";
    ui->textDisplay3D->setText(header);

    QString nD = "nD: 3";

    ui->textDisplay3D->append(nD);

    // file extensions

    QString csv = ".csv";

    // InputType (1 = stack, 0 = .csv)

    QString inputType = "inputType: ";
    int index = 0;
    if(ui->inputType3D->currentText() == "Stack .jpg")
        index = 1;
    else if(ui->inputType3D->currentText() == ".csv")
        index = 0;
    ui->textDisplay3D->append(inputType + QString::number(index));

    if(index == 1)
    {
        // stack size
        QString stackSize = "nSlices: ";
        ui->textDisplay3D->append(stackSize + QString::number(ui->StackSize3D->value()));

        // leading zeroes
        QString leadZero = "leadZero: ";
        ui->textDisplay3D->append(leadZero + QString::number(ui->LeadingZeros3D->value()));

        // threshold
        QString TH = "TH: ";
        ui->textDisplay3D->append(TH + QString::number(ui->Gray_TH_3D->value()));
    } else if(index == 0)
    {
        // check if user entered file name
        QString filenameLabel = "inputFilename: ";
        if(ui->FilenameInput3D_csv->text().isEmpty())
        {
            QString error = "Error: please enter input file name.";
            ui->textDisplay3D->setText(error);
            return;
        }else
        {
            ui->textDisplay3D->append(filenameLabel + ui->FilenameInput3D_csv->text() + csv);
            QString width = "width: ";
            QString depth = "depth: ";
            QString height = "height: ";

            ui->textDisplay3D->append(width + QString::number(ui->width3D->value()));
            ui->textDisplay3D->append(height + QString::number(ui->height3D->value()));
            ui->textDisplay3D->append(depth + QString::number(ui->depth3D->value()));
        }
    }

    // check what code modes to run

    if(ui->poreSD_3D->isChecked())
    {
        QString poreOut = "poreOut: ";
        if (ui->poreSD_Out3D->text().isEmpty())
        {
            QString defaultPoreOut = "defaultPoreOut.csv";
            ui->textDisplay3D->append(poreOut + defaultPoreOut);
        }else
        {
            ui->textDisplay3D->append(poreOut + ui->poreSD_Out3D->text() + csv);
        }
    }

    // pore labels

    if(ui->poreLabel_3D->isChecked())
    {
        QString poreLabelOut = "poreLabelOut: ";
        if (ui->poreLabel_Out3D->text().isEmpty())
        {
            QString default_poreLabelOut = "def_poreLabelOut.csv";
            ui->textDisplay3D->append(poreLabelOut + default_poreLabelOut);
        }
        else
        {
            ui->textDisplay3D->append(poreLabelOut + ui->poreLabel_Out3D->text() + csv);
        }
    }

    // particle-size distribution

    if(ui->partSD_3D->isChecked())
    {
        QString partOut = "partOut: ";
        if (ui->partSD_Out3D->text().isEmpty())
        {
            QString default_partSDOut = "def_partSDOut.csv";
            ui->textDisplay3D->append(partOut + default_partSDOut);
        }else
        {
            ui->textDisplay3D->append(partOut + ui->partSD_Out3D->text() + csv);
        }
    }

    // particle labels

    if(ui->partLabel_3D->isChecked())
    {
        QString partLabelOut = "partLabelOut: ";
        if (ui->partLabel_Out3D->text().isEmpty())
        {
            QString default_partLabelOut = "def_partLabelOut.csv";
            ui->textDisplay3D->append(partLabelOut + default_partLabelOut);
        }else
        {
            ui->textDisplay3D->append(partLabelOut + ui->partLabel_Out3D->text() + csv);
        }
    }

    // Check nThreads

    QString nThreadsLabel = "nThreads: ";
    ui->textDisplay3D->append(nThreadsLabel + QString::number(ui->nThreads3D->value()));

    // Max Radius:

    QString MaxR_Label = "maxR: ";
    ui->textDisplay3D->append(MaxR_Label + QString::number(ui->maxR_3D->value()));

    // verbose

    QString verbose_label = "verbose: ";
    ui->textDisplay3D->append(verbose_label + QString::number(ui->verbose3D->isChecked()));

    // Radius Offset

    QString offsetR_3D_label = "offsetR: ";
    ui->textDisplay3D->append(offsetR_3D_label + QString::number(ui->r_off_3D->value()));

    return;
}


void MainWindow::updateFileText()
{
    // header
    QString header = "Input Parameters:";
    ui->textDisplay2D->setText(header);

    QString nD = "nD: 2";

    ui->textDisplay2D->append(nD);

    // file extensions

    QString csv = ".csv";
    QString jpg = ".jpg";

    // Input file name

    QString s = "inputFilename: ";
    if(ui->FilenameInput2D->text().isEmpty())
    {
        QString error = "Error: invalid input file name.";
        ui->textDisplay2D->setText(error);
        return;
    }
    ui->textDisplay2D->append(s + ui->FilenameInput2D->text() + jpg);

    // Thresholding (Grayscale)

    QString TH2D_Label = "TH: ";
    ui->textDisplay2D->append(TH2D_Label + QString::number(ui->Gray_TH->value()));

    // check what code to run

    // pore-size distribution

    if(ui->poreSD->isChecked())
    {
        QString poreOut = "poreOut: ";
        if (ui->poreSD_Out->text().isEmpty())
        {
            QString defaultPoreOut = "defaultPoreOut.csv";
            ui->textDisplay2D->append(poreOut + defaultPoreOut);
        }else
        {
            ui->textDisplay2D->append(poreOut + ui->poreSD_Out->text() + csv);
        }
    }

    // pore labels

    if(ui->poreLabel->isChecked())
    {
        QString poreLabelOut = "poreLabelOut: ";
        if (ui->poreLabel_Out->text().isEmpty())
        {
            QString default_poreLabelOut = "def_poreLabelOut.csv";
            ui->textDisplay2D->append(poreLabelOut + default_poreLabelOut);
        }
        else
        {
            ui->textDisplay2D->append(poreLabelOut + ui->poreLabel_Out->text() + csv);
        }
    }

    // particle-size distribution

    if(ui->partSD->isChecked())
    {
        QString partOut = "partOut: ";
        if (ui->partSD_Out->text().isEmpty())
        {
            QString default_partSDOut = "def_partSDOut.csv";
            ui->textDisplay2D->append(partOut + default_partSDOut);
        }else
        {
            ui->textDisplay2D->append(partOut + ui->partSD_Out->text() + csv);
        }
    }

    // particle labels

    if(ui->partLabel->isChecked())
    {
        QString partLabelOut = "partLabelOut: ";
        if (ui->partLabel_Out->text().isEmpty())
        {
            QString default_partLabelOut = "def_partLabelOut.csv";
            ui->textDisplay2D->append(partLabelOut + default_partLabelOut);
        }else
        {
            ui->textDisplay2D->append(partLabelOut + ui->partLabel_Out->text() + csv);
        }
    }

    // Check nThreads

    QString nThreadsLabel = "nThreads: ";
    ui->textDisplay2D->append(nThreadsLabel + QString::number(ui->nThreads->value()));

    // Max Radius:

    QString MaxR_Label = "maxR: ";
    ui->textDisplay2D->append(MaxR_Label + QString::number(ui->maxR_2D->value()));

    // verbose

    QString verbose_label = "verbose: ";
    ui->textDisplay2D->append(verbose_label + QString::number(ui->verbose2D->isChecked()));

    // Radius Offset

    QString offsetR_2D_label = "offsetR: ";
    ui->textDisplay2D->append(offsetR_2D_label + QString::number(ui->r_off_2D->value()));

    // Batch Flag

    QString batchLabel2D = "BatchSim: ";
    ui->textDisplay2D->append(batchLabel2D + "0");

    return;
}

MainWindow::~MainWindow()
{
    delete ui;
}


