#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtSerialPort/QSerialPort> //Proporciona funciones basicas para acceder a puertos seriales.


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void ftimer();

private slots:

    void on_barraCanal0Min_valueChanged(int value);

    void on_barraCanal0Max_valueChanged(int value);

    void on_barraCanal1Min_valueChanged(int value);

    void on_barraCanal1Max_valueChanged(int value);

    void on_barraCanal2Min_valueChanged(int value);

    void on_barraCanal2Max_valueChanged(int value);

    void on_buttonBuscar_clicked();

    void on_buttonIniciar_clicked();

    void on_radioButtonRojo_clicked();

    void on_radioButtonVerde_clicked();

    void on_radioButtonAzul_clicked();

    void on_commandLinkButton_clicked();

private:
    Ui::MainWindow *ui;

    QSerialPort *arduino; //Permitira usar el puerto serial (Arduino) en cualquier función de mainWindow.cpp.
    bool arduino_esta_conectado = false; //Variable que indicara si el Arduino se conecta o no.
    void conectarArduino(); //Funcion privada para conectar Arduino.
    void recepcionSerialAsyncrona();
};

#endif // MAINWINDOW_H
