/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.8.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QTabWidget *tabWidget;
    QWidget *tab;
    QTextEdit *textDisplay2D;
    QScrollArea *scrollArea_4;
    QWidget *scrollAreaWidgetContents_4;
    QGridLayout *gridLayout_5;
    QSpinBox *Gray_TH;
    QLabel *label_14;
    QLabel *FilenameLabel2D;
    QLabel *label_7;
    QCheckBox *poreSD;
    QLineEdit *partLabel_Out;
    QLabel *input2D_ext;
    QLabel *label_13;
    QLineEdit *partSD_Out;
    QLabel *label_11;
    QSpinBox *maxR_2D;
    QLabel *label_15;
    QCheckBox *poreLabel;
    QLineEdit *poreSD_Out;
    QSpinBox *r_off_2D;
    QLabel *label;
    QCheckBox *partLabel;
    QLineEdit *poreLabel_Out;
    QSpinBox *nThreads;
    QLabel *label_8;
    QLabel *label_4;
    QCheckBox *verbose2D;
    QLabel *label_9;
    QLabel *label_10;
    QCheckBox *partSD;
    QLabel *label_3;
    QLineEdit *FilenameInput2D;
    QLabel *label_2;
    QLabel *label_5;
    QWidget *horizontalLayoutWidget;
    QHBoxLayout *horizontalLayout;
    QPushButton *GenButton2D;
    QPushButton *clearButton2D;
    QPushButton *saveButton2D;
    QWidget *tab_2;
    QTextEdit *textDisplay3D;
    QStackedWidget *stackedWidget_input3D;
    QWidget *jpgStack;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QGridLayout *gridLayout_2;
    QLabel *label_17;
    QLabel *label_18;
    QSpinBox *StackSize3D;
    QSpinBox *Gray_TH_3D;
    QLabel *label_16;
    QSpinBox *LeadingZeros3D;
    QWidget *csv;
    QScrollArea *scrollArea_3;
    QWidget *scrollAreaWidgetContents_3;
    QLineEdit *FilenameInput3D_csv;
    QLabel *input2D_ext_2;
    QLabel *FilenameLabel2D_2;
    QWidget *layoutWidget;
    QGridLayout *gridLayout_4;
    QLabel *label_33;
    QSpinBox *height3D;
    QLabel *label_34;
    QSpinBox *width3D;
    QLabel *label_35;
    QSpinBox *depth3D;
    QLabel *label_6;
    QComboBox *inputType3D;
    QCheckBox *verbose3D;
    QLabel *label_30;
    QScrollArea *scrollArea_2;
    QWidget *scrollAreaWidgetContents_2;
    QGridLayout *gridLayout_3;
    QLabel *label_28;
    QLabel *label_32;
    QLabel *label_20;
    QLabel *label_23;
    QLabel *label_24;
    QLineEdit *poreSD_Out3D;
    QLineEdit *partLabel_Out3D;
    QLabel *label_27;
    QLabel *label_26;
    QLabel *label_25;
    QCheckBox *partSD_3D;
    QCheckBox *partLabel_3D;
    QLabel *label_29;
    QCheckBox *poreLabel_3D;
    QLineEdit *poreLabel_Out3D;
    QLabel *label_19;
    QCheckBox *poreSD_3D;
    QLineEdit *partSD_Out3D;
    QLabel *label_21;
    QLabel *label_22;
    QSpinBox *maxR_3D;
    QSpinBox *r_off_3D;
    QSpinBox *nThreads3D;
    QLabel *label_31;
    QWidget *horizontalLayoutWidget_2;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *GenButton3D;
    QPushButton *clearButton3D;
    QPushButton *saveButton3D;
    QWidget *runTab;
    QLineEdit *opFolder;
    QLabel *label_12;
    QPushButton *searchFolder;
    QTextEdit *runtimeMessages;
    QLabel *label_36;
    QWidget *horizontalLayoutWidget_3;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *stopButton;
    QPushButton *clearRunButton;
    QPushButton *runButton;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(800, 600);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        tabWidget = new QTabWidget(centralwidget);
        tabWidget->setObjectName("tabWidget");
        tabWidget->setGeometry(QRect(10, 10, 771, 561));
        tab = new QWidget();
        tab->setObjectName("tab");
        textDisplay2D = new QTextEdit(tab);
        textDisplay2D->setObjectName("textDisplay2D");
        textDisplay2D->setGeometry(QRect(380, 30, 361, 411));
        textDisplay2D->setReadOnly(true);
        scrollArea_4 = new QScrollArea(tab);
        scrollArea_4->setObjectName("scrollArea_4");
        scrollArea_4->setGeometry(QRect(10, 20, 341, 501));
        scrollArea_4->setSizeAdjustPolicy(QAbstractScrollArea::SizeAdjustPolicy::AdjustToContents);
        scrollArea_4->setWidgetResizable(true);
        scrollAreaWidgetContents_4 = new QWidget();
        scrollAreaWidgetContents_4->setObjectName("scrollAreaWidgetContents_4");
        scrollAreaWidgetContents_4->setGeometry(QRect(0, 0, 325, 619));
        gridLayout_5 = new QGridLayout(scrollAreaWidgetContents_4);
        gridLayout_5->setObjectName("gridLayout_5");
        Gray_TH = new QSpinBox(scrollAreaWidgetContents_4);
        Gray_TH->setObjectName("Gray_TH");
        Gray_TH->setMinimum(1);
        Gray_TH->setMaximum(255);
        Gray_TH->setSingleStep(1);
        Gray_TH->setValue(127);

        gridLayout_5->addWidget(Gray_TH, 5, 0, 1, 1, Qt::AlignmentFlag::AlignLeft);

        label_14 = new QLabel(scrollAreaWidgetContents_4);
        label_14->setObjectName("label_14");

        gridLayout_5->addWidget(label_14, 21, 0, 1, 1);

        FilenameLabel2D = new QLabel(scrollAreaWidgetContents_4);
        FilenameLabel2D->setObjectName("FilenameLabel2D");

        gridLayout_5->addWidget(FilenameLabel2D, 0, 0, 1, 1);

        label_7 = new QLabel(scrollAreaWidgetContents_4);
        label_7->setObjectName("label_7");

        gridLayout_5->addWidget(label_7, 12, 1, 1, 1);

        poreSD = new QCheckBox(scrollAreaWidgetContents_4);
        poreSD->setObjectName("poreSD");

        gridLayout_5->addWidget(poreSD, 7, 0, 1, 1);

        partLabel_Out = new QLineEdit(scrollAreaWidgetContents_4);
        partLabel_Out->setObjectName("partLabel_Out");
        partLabel_Out->setReadOnly(true);

        gridLayout_5->addWidget(partLabel_Out, 18, 0, 1, 1);

        input2D_ext = new QLabel(scrollAreaWidgetContents_4);
        input2D_ext->setObjectName("input2D_ext");

        gridLayout_5->addWidget(input2D_ext, 2, 1, 1, 1);

        label_13 = new QLabel(scrollAreaWidgetContents_4);
        label_13->setObjectName("label_13");

        gridLayout_5->addWidget(label_13, 3, 0, 1, 1);

        partSD_Out = new QLineEdit(scrollAreaWidgetContents_4);
        partSD_Out->setObjectName("partSD_Out");
        partSD_Out->setReadOnly(true);

        gridLayout_5->addWidget(partSD_Out, 14, 0, 1, 1);

        label_11 = new QLabel(scrollAreaWidgetContents_4);
        label_11->setObjectName("label_11");

        gridLayout_5->addWidget(label_11, 19, 0, 1, 1);

        maxR_2D = new QSpinBox(scrollAreaWidgetContents_4);
        maxR_2D->setObjectName("maxR_2D");
        maxR_2D->setMinimum(1);
        maxR_2D->setMaximum(100000);
        maxR_2D->setValue(100);

        gridLayout_5->addWidget(maxR_2D, 24, 0, 1, 1, Qt::AlignmentFlag::AlignLeft);

        label_15 = new QLabel(scrollAreaWidgetContents_4);
        label_15->setObjectName("label_15");

        gridLayout_5->addWidget(label_15, 23, 0, 1, 1);

        poreLabel = new QCheckBox(scrollAreaWidgetContents_4);
        poreLabel->setObjectName("poreLabel");
        poreLabel->setCheckable(false);

        gridLayout_5->addWidget(poreLabel, 8, 0, 1, 1);

        poreSD_Out = new QLineEdit(scrollAreaWidgetContents_4);
        poreSD_Out->setObjectName("poreSD_Out");
        poreSD_Out->setReadOnly(true);

        gridLayout_5->addWidget(poreSD_Out, 12, 0, 1, 1);

        r_off_2D = new QSpinBox(scrollAreaWidgetContents_4);
        r_off_2D->setObjectName("r_off_2D");

        gridLayout_5->addWidget(r_off_2D, 22, 0, 1, 1, Qt::AlignmentFlag::AlignLeft);

        label = new QLabel(scrollAreaWidgetContents_4);
        label->setObjectName("label");

        gridLayout_5->addWidget(label, 6, 0, 1, 1);

        partLabel = new QCheckBox(scrollAreaWidgetContents_4);
        partLabel->setObjectName("partLabel");
        partLabel->setCheckable(false);

        gridLayout_5->addWidget(partLabel, 10, 0, 1, 1);

        poreLabel_Out = new QLineEdit(scrollAreaWidgetContents_4);
        poreLabel_Out->setObjectName("poreLabel_Out");
        poreLabel_Out->setReadOnly(true);

        gridLayout_5->addWidget(poreLabel_Out, 16, 0, 1, 1);

        nThreads = new QSpinBox(scrollAreaWidgetContents_4);
        nThreads->setObjectName("nThreads");
        nThreads->setMinimum(1);
        nThreads->setMaximum(1280);

        gridLayout_5->addWidget(nThreads, 20, 0, 1, 1, Qt::AlignmentFlag::AlignLeft);

        label_8 = new QLabel(scrollAreaWidgetContents_4);
        label_8->setObjectName("label_8");

        gridLayout_5->addWidget(label_8, 14, 1, 1, 1);

        label_4 = new QLabel(scrollAreaWidgetContents_4);
        label_4->setObjectName("label_4");

        gridLayout_5->addWidget(label_4, 15, 0, 1, 1);

        verbose2D = new QCheckBox(scrollAreaWidgetContents_4);
        verbose2D->setObjectName("verbose2D");
        verbose2D->setChecked(true);

        gridLayout_5->addWidget(verbose2D, 26, 0, 1, 1);

        label_9 = new QLabel(scrollAreaWidgetContents_4);
        label_9->setObjectName("label_9");

        gridLayout_5->addWidget(label_9, 16, 1, 1, 1);

        label_10 = new QLabel(scrollAreaWidgetContents_4);
        label_10->setObjectName("label_10");

        gridLayout_5->addWidget(label_10, 18, 1, 1, 1);

        partSD = new QCheckBox(scrollAreaWidgetContents_4);
        partSD->setObjectName("partSD");

        gridLayout_5->addWidget(partSD, 9, 0, 1, 1);

        label_3 = new QLabel(scrollAreaWidgetContents_4);
        label_3->setObjectName("label_3");

        gridLayout_5->addWidget(label_3, 13, 0, 1, 1);

        FilenameInput2D = new QLineEdit(scrollAreaWidgetContents_4);
        FilenameInput2D->setObjectName("FilenameInput2D");

        gridLayout_5->addWidget(FilenameInput2D, 2, 0, 1, 1);

        label_2 = new QLabel(scrollAreaWidgetContents_4);
        label_2->setObjectName("label_2");

        gridLayout_5->addWidget(label_2, 11, 0, 1, 1);

        label_5 = new QLabel(scrollAreaWidgetContents_4);
        label_5->setObjectName("label_5");

        gridLayout_5->addWidget(label_5, 17, 0, 1, 1);

        scrollArea_4->setWidget(scrollAreaWidgetContents_4);
        horizontalLayoutWidget = new QWidget(tab);
        horizontalLayoutWidget->setObjectName("horizontalLayoutWidget");
        horizontalLayoutWidget->setGeometry(QRect(370, 450, 381, 71));
        horizontalLayout = new QHBoxLayout(horizontalLayoutWidget);
        horizontalLayout->setObjectName("horizontalLayout");
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        GenButton2D = new QPushButton(horizontalLayoutWidget);
        GenButton2D->setObjectName("GenButton2D");

        horizontalLayout->addWidget(GenButton2D);

        clearButton2D = new QPushButton(horizontalLayoutWidget);
        clearButton2D->setObjectName("clearButton2D");

        horizontalLayout->addWidget(clearButton2D);

        saveButton2D = new QPushButton(horizontalLayoutWidget);
        saveButton2D->setObjectName("saveButton2D");

        horizontalLayout->addWidget(saveButton2D);

        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName("tab_2");
        textDisplay3D = new QTextEdit(tab_2);
        textDisplay3D->setObjectName("textDisplay3D");
        textDisplay3D->setGeometry(QRect(380, 20, 371, 421));
        textDisplay3D->setReadOnly(true);
        stackedWidget_input3D = new QStackedWidget(tab_2);
        stackedWidget_input3D->setObjectName("stackedWidget_input3D");
        stackedWidget_input3D->setGeometry(QRect(10, 50, 351, 211));
        jpgStack = new QWidget();
        jpgStack->setObjectName("jpgStack");
        scrollArea = new QScrollArea(jpgStack);
        scrollArea->setObjectName("scrollArea");
        scrollArea->setGeometry(QRect(19, 9, 311, 131));
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName("scrollAreaWidgetContents");
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 309, 129));
        gridLayout_2 = new QGridLayout(scrollAreaWidgetContents);
        gridLayout_2->setObjectName("gridLayout_2");
        label_17 = new QLabel(scrollAreaWidgetContents);
        label_17->setObjectName("label_17");

        gridLayout_2->addWidget(label_17, 1, 0, 1, 1);

        label_18 = new QLabel(scrollAreaWidgetContents);
        label_18->setObjectName("label_18");

        gridLayout_2->addWidget(label_18, 2, 0, 1, 1);

        StackSize3D = new QSpinBox(scrollAreaWidgetContents);
        StackSize3D->setObjectName("StackSize3D");
        StackSize3D->setMinimum(1);
        StackSize3D->setMaximum(100000);

        gridLayout_2->addWidget(StackSize3D, 0, 1, 1, 2);

        Gray_TH_3D = new QSpinBox(scrollAreaWidgetContents);
        Gray_TH_3D->setObjectName("Gray_TH_3D");
        Gray_TH_3D->setMinimum(1);
        Gray_TH_3D->setMaximum(255);
        Gray_TH_3D->setSingleStep(1);
        Gray_TH_3D->setValue(127);

        gridLayout_2->addWidget(Gray_TH_3D, 2, 1, 1, 2);

        label_16 = new QLabel(scrollAreaWidgetContents);
        label_16->setObjectName("label_16");

        gridLayout_2->addWidget(label_16, 0, 0, 1, 1);

        LeadingZeros3D = new QSpinBox(scrollAreaWidgetContents);
        LeadingZeros3D->setObjectName("LeadingZeros3D");
        LeadingZeros3D->setMaximum(7);

        gridLayout_2->addWidget(LeadingZeros3D, 1, 1, 1, 2);

        scrollArea->setWidget(scrollAreaWidgetContents);
        stackedWidget_input3D->addWidget(jpgStack);
        csv = new QWidget();
        csv->setObjectName("csv");
        scrollArea_3 = new QScrollArea(csv);
        scrollArea_3->setObjectName("scrollArea_3");
        scrollArea_3->setGeometry(QRect(10, 10, 311, 181));
        scrollArea_3->setWidgetResizable(true);
        scrollAreaWidgetContents_3 = new QWidget();
        scrollAreaWidgetContents_3->setObjectName("scrollAreaWidgetContents_3");
        scrollAreaWidgetContents_3->setGeometry(QRect(0, 0, 309, 179));
        FilenameInput3D_csv = new QLineEdit(scrollAreaWidgetContents_3);
        FilenameInput3D_csv->setObjectName("FilenameInput3D_csv");
        FilenameInput3D_csv->setGeometry(QRect(10, 24, 261, 31));
        input2D_ext_2 = new QLabel(scrollAreaWidgetContents_3);
        input2D_ext_2->setObjectName("input2D_ext_2");
        input2D_ext_2->setGeometry(QRect(276, 30, 41, 20));
        FilenameLabel2D_2 = new QLabel(scrollAreaWidgetContents_3);
        FilenameLabel2D_2->setObjectName("FilenameLabel2D_2");
        FilenameLabel2D_2->setGeometry(QRect(10, 4, 201, 16));
        layoutWidget = new QWidget(scrollAreaWidgetContents_3);
        layoutWidget->setObjectName("layoutWidget");
        layoutWidget->setGeometry(QRect(20, 60, 121, 101));
        gridLayout_4 = new QGridLayout(layoutWidget);
        gridLayout_4->setObjectName("gridLayout_4");
        gridLayout_4->setContentsMargins(0, 0, 0, 0);
        label_33 = new QLabel(layoutWidget);
        label_33->setObjectName("label_33");

        gridLayout_4->addWidget(label_33, 0, 0, 1, 1);

        height3D = new QSpinBox(layoutWidget);
        height3D->setObjectName("height3D");
        height3D->setMinimum(1);
        height3D->setMaximum(10000);

        gridLayout_4->addWidget(height3D, 0, 1, 1, 1);

        label_34 = new QLabel(layoutWidget);
        label_34->setObjectName("label_34");

        gridLayout_4->addWidget(label_34, 1, 0, 1, 1);

        width3D = new QSpinBox(layoutWidget);
        width3D->setObjectName("width3D");
        width3D->setMinimum(1);
        width3D->setMaximum(10000);

        gridLayout_4->addWidget(width3D, 1, 1, 1, 1);

        label_35 = new QLabel(layoutWidget);
        label_35->setObjectName("label_35");

        gridLayout_4->addWidget(label_35, 2, 0, 1, 1);

        depth3D = new QSpinBox(layoutWidget);
        depth3D->setObjectName("depth3D");
        depth3D->setMinimum(1);
        depth3D->setMaximum(10000);

        gridLayout_4->addWidget(depth3D, 2, 1, 1, 1);

        scrollArea_3->setWidget(scrollAreaWidgetContents_3);
        stackedWidget_input3D->addWidget(csv);
        label_6 = new QLabel(tab_2);
        label_6->setObjectName("label_6");
        label_6->setGeometry(QRect(27, 12, 111, 16));
        inputType3D = new QComboBox(tab_2);
        inputType3D->addItem(QString());
        inputType3D->addItem(QString());
        inputType3D->setObjectName("inputType3D");
        inputType3D->setGeometry(QRect(127, 8, 101, 22));
        verbose3D = new QCheckBox(tab_2);
        verbose3D->setObjectName("verbose3D");
        verbose3D->setGeometry(QRect(250, 10, 101, 20));
        verbose3D->setChecked(true);
        label_30 = new QLabel(tab_2);
        label_30->setObjectName("label_30");
        label_30->setGeometry(QRect(25, 40, 141, 16));
        scrollArea_2 = new QScrollArea(tab_2);
        scrollArea_2->setObjectName("scrollArea_2");
        scrollArea_2->setGeometry(QRect(20, 270, 331, 251));
        scrollArea_2->setWidgetResizable(true);
        scrollAreaWidgetContents_2 = new QWidget();
        scrollAreaWidgetContents_2->setObjectName("scrollAreaWidgetContents_2");
        scrollAreaWidgetContents_2->setGeometry(QRect(0, 0, 315, 491));
        gridLayout_3 = new QGridLayout(scrollAreaWidgetContents_2);
        gridLayout_3->setObjectName("gridLayout_3");
        label_28 = new QLabel(scrollAreaWidgetContents_2);
        label_28->setObjectName("label_28");

        gridLayout_3->addWidget(label_28, 18, 0, 1, 1);

        label_32 = new QLabel(scrollAreaWidgetContents_2);
        label_32->setObjectName("label_32");

        gridLayout_3->addWidget(label_32, 10, 0, 1, 1);

        label_20 = new QLabel(scrollAreaWidgetContents_2);
        label_20->setObjectName("label_20");

        gridLayout_3->addWidget(label_20, 12, 0, 1, 1);

        label_23 = new QLabel(scrollAreaWidgetContents_2);
        label_23->setObjectName("label_23");

        gridLayout_3->addWidget(label_23, 7, 0, 1, 1);

        label_24 = new QLabel(scrollAreaWidgetContents_2);
        label_24->setObjectName("label_24");

        gridLayout_3->addWidget(label_24, 14, 0, 1, 1);

        poreSD_Out3D = new QLineEdit(scrollAreaWidgetContents_2);
        poreSD_Out3D->setObjectName("poreSD_Out3D");
        poreSD_Out3D->setReadOnly(true);

        gridLayout_3->addWidget(poreSD_Out3D, 6, 0, 1, 1);

        partLabel_Out3D = new QLineEdit(scrollAreaWidgetContents_2);
        partLabel_Out3D->setObjectName("partLabel_Out3D");
        partLabel_Out3D->setReadOnly(true);

        gridLayout_3->addWidget(partLabel_Out3D, 13, 0, 1, 1);

        label_27 = new QLabel(scrollAreaWidgetContents_2);
        label_27->setObjectName("label_27");

        gridLayout_3->addWidget(label_27, 9, 1, 1, 1);

        label_26 = new QLabel(scrollAreaWidgetContents_2);
        label_26->setObjectName("label_26");

        gridLayout_3->addWidget(label_26, 5, 0, 1, 1);

        label_25 = new QLabel(scrollAreaWidgetContents_2);
        label_25->setObjectName("label_25");

        gridLayout_3->addWidget(label_25, 11, 1, 1, 1);

        partSD_3D = new QCheckBox(scrollAreaWidgetContents_2);
        partSD_3D->setObjectName("partSD_3D");

        gridLayout_3->addWidget(partSD_3D, 3, 0, 1, 1);

        partLabel_3D = new QCheckBox(scrollAreaWidgetContents_2);
        partLabel_3D->setObjectName("partLabel_3D");
        partLabel_3D->setCheckable(false);

        gridLayout_3->addWidget(partLabel_3D, 4, 0, 1, 1);

        label_29 = new QLabel(scrollAreaWidgetContents_2);
        label_29->setObjectName("label_29");

        gridLayout_3->addWidget(label_29, 13, 1, 1, 1);

        poreLabel_3D = new QCheckBox(scrollAreaWidgetContents_2);
        poreLabel_3D->setObjectName("poreLabel_3D");
        poreLabel_3D->setCheckable(false);

        gridLayout_3->addWidget(poreLabel_3D, 2, 0, 1, 1);

        poreLabel_Out3D = new QLineEdit(scrollAreaWidgetContents_2);
        poreLabel_Out3D->setObjectName("poreLabel_Out3D");
        poreLabel_Out3D->setReadOnly(true);

        gridLayout_3->addWidget(poreLabel_Out3D, 11, 0, 1, 1);

        label_19 = new QLabel(scrollAreaWidgetContents_2);
        label_19->setObjectName("label_19");

        gridLayout_3->addWidget(label_19, 0, 0, 1, 2);

        poreSD_3D = new QCheckBox(scrollAreaWidgetContents_2);
        poreSD_3D->setObjectName("poreSD_3D");

        gridLayout_3->addWidget(poreSD_3D, 1, 0, 1, 1);

        partSD_Out3D = new QLineEdit(scrollAreaWidgetContents_2);
        partSD_Out3D->setObjectName("partSD_Out3D");
        partSD_Out3D->setReadOnly(true);

        gridLayout_3->addWidget(partSD_Out3D, 9, 0, 1, 1);

        label_21 = new QLabel(scrollAreaWidgetContents_2);
        label_21->setObjectName("label_21");

        gridLayout_3->addWidget(label_21, 6, 1, 1, 1);

        label_22 = new QLabel(scrollAreaWidgetContents_2);
        label_22->setObjectName("label_22");

        gridLayout_3->addWidget(label_22, 16, 0, 1, 1);

        maxR_3D = new QSpinBox(scrollAreaWidgetContents_2);
        maxR_3D->setObjectName("maxR_3D");
        QSizePolicy sizePolicy(QSizePolicy::Policy::Fixed, QSizePolicy::Policy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(maxR_3D->sizePolicy().hasHeightForWidth());
        maxR_3D->setSizePolicy(sizePolicy);
        maxR_3D->setMinimum(1);
        maxR_3D->setMaximum(100000);
        maxR_3D->setValue(100);

        gridLayout_3->addWidget(maxR_3D, 19, 0, 1, 2);

        r_off_3D = new QSpinBox(scrollAreaWidgetContents_2);
        r_off_3D->setObjectName("r_off_3D");
        sizePolicy.setHeightForWidth(r_off_3D->sizePolicy().hasHeightForWidth());
        r_off_3D->setSizePolicy(sizePolicy);

        gridLayout_3->addWidget(r_off_3D, 17, 0, 1, 2);

        nThreads3D = new QSpinBox(scrollAreaWidgetContents_2);
        nThreads3D->setObjectName("nThreads3D");
        sizePolicy.setHeightForWidth(nThreads3D->sizePolicy().hasHeightForWidth());
        nThreads3D->setSizePolicy(sizePolicy);
        nThreads3D->setMinimum(1);
        nThreads3D->setMaximum(1280);

        gridLayout_3->addWidget(nThreads3D, 15, 0, 1, 2);

        scrollArea_2->setWidget(scrollAreaWidgetContents_2);
        label_31 = new QLabel(tab_2);
        label_31->setObjectName("label_31");
        label_31->setGeometry(QRect(26, 248, 141, 16));
        horizontalLayoutWidget_2 = new QWidget(tab_2);
        horizontalLayoutWidget_2->setObjectName("horizontalLayoutWidget_2");
        horizontalLayoutWidget_2->setGeometry(QRect(380, 450, 381, 71));
        horizontalLayout_2 = new QHBoxLayout(horizontalLayoutWidget_2);
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        GenButton3D = new QPushButton(horizontalLayoutWidget_2);
        GenButton3D->setObjectName("GenButton3D");

        horizontalLayout_2->addWidget(GenButton3D);

        clearButton3D = new QPushButton(horizontalLayoutWidget_2);
        clearButton3D->setObjectName("clearButton3D");

        horizontalLayout_2->addWidget(clearButton3D);

        saveButton3D = new QPushButton(horizontalLayoutWidget_2);
        saveButton3D->setObjectName("saveButton3D");

        horizontalLayout_2->addWidget(saveButton3D);

        tabWidget->addTab(tab_2, QString());
        runTab = new QWidget();
        runTab->setObjectName("runTab");
        opFolder = new QLineEdit(runTab);
        opFolder->setObjectName("opFolder");
        opFolder->setGeometry(QRect(10, 30, 261, 22));
        opFolder->setReadOnly(true);
        label_12 = new QLabel(runTab);
        label_12->setObjectName("label_12");
        label_12->setGeometry(QRect(10, 10, 261, 16));
        searchFolder = new QPushButton(runTab);
        searchFolder->setObjectName("searchFolder");
        searchFolder->setGeometry(QRect(280, 30, 41, 21));
        runtimeMessages = new QTextEdit(runTab);
        runtimeMessages->setObjectName("runtimeMessages");
        runtimeMessages->setGeometry(QRect(370, 50, 371, 471));
        runtimeMessages->setReadOnly(true);
        label_36 = new QLabel(runTab);
        label_36->setObjectName("label_36");
        label_36->setGeometry(QRect(370, 30, 261, 16));
        horizontalLayoutWidget_3 = new QWidget(runTab);
        horizontalLayoutWidget_3->setObjectName("horizontalLayoutWidget_3");
        horizontalLayoutWidget_3->setGeometry(QRect(0, 440, 361, 80));
        horizontalLayout_3 = new QHBoxLayout(horizontalLayoutWidget_3);
        horizontalLayout_3->setObjectName("horizontalLayout_3");
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        stopButton = new QPushButton(horizontalLayoutWidget_3);
        stopButton->setObjectName("stopButton");
        stopButton->setEnabled(false);
        QFont font;
        font.setPointSize(12);
        stopButton->setFont(font);
        stopButton->setCheckable(false);
        stopButton->setFlat(false);

        horizontalLayout_3->addWidget(stopButton);

        clearRunButton = new QPushButton(horizontalLayoutWidget_3);
        clearRunButton->setObjectName("clearRunButton");
        clearRunButton->setFont(font);

        horizontalLayout_3->addWidget(clearRunButton);

        runButton = new QPushButton(horizontalLayoutWidget_3);
        runButton->setObjectName("runButton");
        runButton->setFont(font);

        horizontalLayout_3->addWidget(runButton);

        tabWidget->addTab(runTab, QString());
        MainWindow->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);
        stackedWidget_input3D->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "ED-cPSD", nullptr));
        label_14->setText(QCoreApplication::translate("MainWindow", "Radius Offset:", nullptr));
        FilenameLabel2D->setText(QCoreApplication::translate("MainWindow", "Enter input file name below:", nullptr));
        label_7->setText(QCoreApplication::translate("MainWindow", ".csv", nullptr));
        poreSD->setText(QCoreApplication::translate("MainWindow", "Pore-size distribution", nullptr));
        input2D_ext->setText(QCoreApplication::translate("MainWindow", ".jpg", nullptr));
        label_13->setText(QCoreApplication::translate("MainWindow", "Grayscale Threshold", nullptr));
        label_11->setText(QCoreApplication::translate("MainWindow", "CPU Cores:", nullptr));
        label_15->setText(QCoreApplication::translate("MainWindow", "Max. Radius:", nullptr));
        poreLabel->setText(QCoreApplication::translate("MainWindow", "Pore Labels", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "Select expected outputs:", nullptr));
        partLabel->setText(QCoreApplication::translate("MainWindow", "Particle Labels", nullptr));
        label_8->setText(QCoreApplication::translate("MainWindow", ".csv", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "File for saving pore labels:", nullptr));
        verbose2D->setText(QCoreApplication::translate("MainWindow", "Verbose", nullptr));
        label_9->setText(QCoreApplication::translate("MainWindow", ".csv", nullptr));
        label_10->setText(QCoreApplication::translate("MainWindow", ".csv", nullptr));
        partSD->setText(QCoreApplication::translate("MainWindow", "Particle-size distribution", nullptr));
        label_3->setText(QCoreApplication::translate("MainWindow", "File name for saving particle-size distribution:", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "File name for saving pore-size distribution:", nullptr));
        label_5->setText(QCoreApplication::translate("MainWindow", "File for saving particle labels:", nullptr));
        GenButton2D->setText(QCoreApplication::translate("MainWindow", "Generate", nullptr));
        clearButton2D->setText(QCoreApplication::translate("MainWindow", "Clear", nullptr));
        saveButton2D->setText(QCoreApplication::translate("MainWindow", "Save", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab), QCoreApplication::translate("MainWindow", "2D", nullptr));
        label_17->setText(QCoreApplication::translate("MainWindow", "Leading Zeroes:", nullptr));
        label_18->setText(QCoreApplication::translate("MainWindow", "Grayscale Threshold", nullptr));
        label_16->setText(QCoreApplication::translate("MainWindow", "Stack Size:", nullptr));
        input2D_ext_2->setText(QCoreApplication::translate("MainWindow", ".csv", nullptr));
        FilenameLabel2D_2->setText(QCoreApplication::translate("MainWindow", "Enter input file name below:", nullptr));
        label_33->setText(QCoreApplication::translate("MainWindow", "Height:", nullptr));
        label_34->setText(QCoreApplication::translate("MainWindow", "Width:", nullptr));
        label_35->setText(QCoreApplication::translate("MainWindow", "Depth:", nullptr));
        label_6->setText(QCoreApplication::translate("MainWindow", "Enter Input Type:", nullptr));
        inputType3D->setItemText(0, QCoreApplication::translate("MainWindow", "Stack .jpg", nullptr));
        inputType3D->setItemText(1, QCoreApplication::translate("MainWindow", ".csv", nullptr));

        verbose3D->setText(QCoreApplication::translate("MainWindow", "Verbose", nullptr));
        label_30->setText(QCoreApplication::translate("MainWindow", "Input Specific Settings:", nullptr));
        label_28->setText(QCoreApplication::translate("MainWindow", "Max. Radius:", nullptr));
        label_32->setText(QCoreApplication::translate("MainWindow", "File for saving pore labels:", nullptr));
        label_20->setText(QCoreApplication::translate("MainWindow", "File for saving particle labels:", nullptr));
        label_23->setText(QCoreApplication::translate("MainWindow", "File name for saving particle-size distribution:", nullptr));
        label_24->setText(QCoreApplication::translate("MainWindow", "Number of CPU Cores: ", nullptr));
        label_27->setText(QCoreApplication::translate("MainWindow", ".csv", nullptr));
        label_26->setText(QCoreApplication::translate("MainWindow", "File name for saving pore-size distribution:", nullptr));
        label_25->setText(QCoreApplication::translate("MainWindow", ".csv", nullptr));
        partSD_3D->setText(QCoreApplication::translate("MainWindow", "Particle-size distribution", nullptr));
        partLabel_3D->setText(QCoreApplication::translate("MainWindow", "Particle Labels", nullptr));
        label_29->setText(QCoreApplication::translate("MainWindow", ".csv", nullptr));
        poreLabel_3D->setText(QCoreApplication::translate("MainWindow", "Pore Labels", nullptr));
        label_19->setText(QCoreApplication::translate("MainWindow", "Select expected outputs:", nullptr));
        poreSD_3D->setText(QCoreApplication::translate("MainWindow", "Pore-size distribution", nullptr));
        label_21->setText(QCoreApplication::translate("MainWindow", ".csv", nullptr));
        label_22->setText(QCoreApplication::translate("MainWindow", "Radius Offset:", nullptr));
        label_31->setText(QCoreApplication::translate("MainWindow", "General Settings:", nullptr));
        GenButton3D->setText(QCoreApplication::translate("MainWindow", "Generate", nullptr));
        clearButton3D->setText(QCoreApplication::translate("MainWindow", "Clear", nullptr));
        saveButton3D->setText(QCoreApplication::translate("MainWindow", "Save", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QCoreApplication::translate("MainWindow", "3D", nullptr));
        label_12->setText(QCoreApplication::translate("MainWindow", "Operating Folder (where input file is):", nullptr));
        searchFolder->setText(QCoreApplication::translate("MainWindow", "...", nullptr));
        label_36->setText(QCoreApplication::translate("MainWindow", "Runtime Messages:", nullptr));
        stopButton->setText(QCoreApplication::translate("MainWindow", "Stop", nullptr));
        clearRunButton->setText(QCoreApplication::translate("MainWindow", "Clear", nullptr));
        runButton->setText(QCoreApplication::translate("MainWindow", "Run", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(runTab), QCoreApplication::translate("MainWindow", "Run", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
