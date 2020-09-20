#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include "mat2qimage.h"
#include <QFileDialog> //Permite usar La clase QFileDialog proporciona un cuadro de diálogo que permite a los usuarios seleccionar archivos o directorios.
#include <QFile> //Permite usar la clase QFile para leer y escribir archivos.
#include <QTimer> //Permite usar la clase Qtimer  que proporciona una interfaz de programación de alto nivel para los temporizadores.
#include <QDebug>  //Permite usar la clase QDebug para salida de informacion en el depurador.
#include <QSerialPortInfo> //Permite usar la clase QSerialPortInfo para proporcionar informacion sobre los puertos existentes.
#include <QSerialPort> //Permite usar la clase QSerialPort que proporciona la funciones de acceso a los puertos seriales.
using namespace cv;
using namespace std;

Mat IMAGEN;
Mat IMAGENchica;
Mat ImagenHSV;
Mat ImagenFiltrada;
Mat ImagenFiltradaVerde;
Mat ImagenFiltradaRoja;
Mat ImagenFiltradaAzul;
Mat drawing;
Mat drawingAzul;
Mat drawingVerde;
Mat drawingRoja;

int umbralVerde = 1000;
int umbralAzul = 1000;
int umbralRojo = 1000;
int canal = 0;

int canal0Min[3] = {0,   0,   60};
int canal0Max[3] = {81,  60, 255};
int canal1Min[3] = {0,   53,  68};
int canal1Max[3] = {122, 255, 152};
int canal2Min[3] = {118, 0,   0};
int canal2Max[3] = {255, 37,  145};

int contadorVerde, contadorRojo, contadorAzul;

bool bandera1 = true;
bool bandera2 = true;
bool primeraVez = true;
bool primeraVez1 = true;
bool primeraVez2 = true;


VideoCapture camaraLocal(1); //Camara a seleccionar.
RNG rng(12345); //Generador de numeros aleatorios.

void MainWindow::ftimer(){
    Point2f centro, centro1, centro2;

    //Valores de los canales para filtrar el color.
    canal = 0*ui->radioButtonRojo->isChecked() + 1*ui->radioButtonVerde->isChecked() + 2*ui->radioButtonAzul->isChecked();
    canal0Min[canal] = ui->barraCanal0Min->value();
    canal0Max[canal] = ui->barraCanal0Max->value();
    canal1Min[canal] = ui->barraCanal1Min->value();
    canal1Max[canal] = ui->barraCanal1Max->value();
    canal2Min[canal] = ui->barraCanal2Min->value();
    canal2Max[canal] = ui->barraCanal2Max->value();

    //Para Camara web o ip
    if(ui->checkBox->isChecked()){ //Verifica si el check-box esta activados.
        if(camaraLocal.isOpened()){ //Verifica si la camara esta abierta.
            camaraLocal >> IMAGEN; //Abre la imagen de la camara en el Mat IMAGEN.
        }
        else{ //En caso de no abrir la camara se creara un matriz de error.
            Mat erroMatriz (400,400, CV_8UC3, Scalar(255,255,255));
            IMAGEN = erroMatriz;
            camaraLocal.open(1);
        }

        //Paso # 1 - Adaptar la imagen a la etiqueta
        cv::resize(IMAGEN,IMAGENchica,Size(150,150));

        //Paso # 2 - Procesar la imagen anterior para obtener HSV
        cvtColor(IMAGENchica,ImagenHSV,CV_BGR2Lab);

        //Paso # 3 - Mostrar las imagenes en sus respectivas etiquetas
        QImage qImage = Mat2QImage(IMAGENchica);
        QPixmap pixmap = QPixmap::fromImage(qImage);
        ui->labelColor->clear();
        ui->labelColor->setPixmap(pixmap);

        qImage = Mat2QImage(ImagenHSV);
        pixmap = QPixmap::fromImage(qImage);
        ui->labelHSV->clear();
        ui->labelHSV->setPixmap(pixmap);
    }

    //Procesar imagen para filtro de ventana
    if(!ImagenHSV.empty()){ //Verifica que la ImagenHSV no sea vacia.

        Scalar colorMinR = Scalar(canal0Min[0], canal1Min[0], canal2Min[0]);
        Scalar colorMaxR = Scalar(canal0Max[0], canal1Max[0], canal2Max[0]);
        Scalar colorMinV = Scalar(canal0Min[1], canal1Min[1], canal2Min[1]);
        Scalar colorMaxV = Scalar(canal0Max[1], canal1Max[1], canal2Max[1]);
        Scalar colorMinA = Scalar(canal0Min[2], canal1Min[2], canal2Min[2]);
        Scalar colorMaxA = Scalar(canal0Max[2], canal1Max[2], canal2Max[2]);

//******Filtrado del color Rojo*********************************************************************************************//
        inRange(IMAGENchica, Scalar(colorMinR), Scalar(colorMaxR), ImagenFiltradaRoja);
        vector<vector<Point> > contornosRoja;
        vector<Vec4i> jerarquiaRoja;

        Mat copiaImagenFiltradaRoja;
        copiaImagenFiltradaRoja = ImagenFiltradaRoja;
        findContours(copiaImagenFiltradaRoja, contornosRoja, jerarquiaRoja, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) ); //Funcion para encontrar contornos, depende del modo.
        drawingRoja = Mat::zeros(copiaImagenFiltradaRoja.size(), CV_8UC3 ); //Matriz negra de tres canales.

        int area = 0;
        int areaMaximaA = 0;
        int areaMaximaV = 0;
        int areaMaximaR = 0;
        int indiceJ = -1;

        for(int i = 0; i < (int)contornosRoja.size(); i++){ //Para encontrar todos los contornos.
            if(jerarquiaRoja [i][2]!= -1) { //Para encontrar jerarquias, buscando padres.
                drawContours (drawingRoja, contornosRoja, i, Scalar(0, 0, 255), 1, 8, jerarquiaRoja, 0); //Dibuja el contorno.
                area = contourArea(contornosRoja[i]); //Calcula el area.

                if(area > areaMaximaR){
                    areaMaximaR = area;
                    indiceJ = i;
                }

                if(indiceJ >= 0){
                    Moments momento0;
                    //Encontrar el momento del contorno indice J.
                    momento0 = moments(contornosRoja[indiceJ]);
                    //Calcular el centro del momento
                    centro = Point2f( static_cast<float>(momento0.m10/momento0.m00), static_cast<float>(momento0.m01/momento0.m00));
                    //Dibuja un circulo
                    circle(drawingRoja, centro, 10, Scalar(255, 255, 255), 2, 8, 0);
                }
            }
        }

        if (areaMaximaR >= umbralRojo && bandera1 == true){ //Banderas para solo mandar un dato.
            qDebug()<<"Color encontrado rojo";
            if(bandera2){
                bandera1 = false;
                bandera2 = false;

                if(arduino_esta_conectado && arduino->isWritable()){
                    arduino->write("d180p");
                }
            }
        }

        if(areaMaximaA < umbralAzul || areaMaximaV < umbralVerde || areaMaximaR < umbralRojo){
            bandera2 = true;
        }

//***************************************************************************************************************************//

//******Filtrado del color Verde*********************************************************************************************//
        inRange(IMAGENchica, Scalar(colorMinV), Scalar(colorMaxV), ImagenFiltradaVerde);
        vector<vector<Point> > contornosVerde;
        vector<Vec4i> jerarquiaVerde;

        Mat copiaImagenFiltradaVerde;
        copiaImagenFiltradaVerde = ImagenFiltradaVerde;
        findContours(copiaImagenFiltradaVerde, contornosVerde, jerarquiaVerde, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) ); //Funcion para encontrar contornos, depende del modo.
        drawingVerde = Mat::zeros(copiaImagenFiltradaVerde.size(), CV_8UC3 ); //Matriz negra de tres canales.

        area = 0;
        indiceJ = -1;

        for(int i = 0; i < (int)contornosVerde.size(); i++){ //Para encontrar todos los contornos.
            if(jerarquiaVerde [i][2]!= -1) { //Para encontrar jerarquias, buscando padres.
                drawContours (drawingVerde, contornosVerde, i, Scalar(0, 255, 0), 1, 8, jerarquiaVerde, 0); //Dibuja el contorno.
                area = contourArea(contornosVerde[i]); //Calcula el area.

                if(area > areaMaximaV){
                    areaMaximaV = area;
                    indiceJ = i;
                }

                if(indiceJ >= 0){
                    Moments momento1;
                    //Encontrar el momento del contorno indice J.
                    momento1 = moments(contornosVerde[indiceJ]);
                    //Calcular el centro del momento
                    centro1 = Point2f( static_cast<float>(momento1.m10/momento1.m00), static_cast<float>(momento1.m01/momento1.m00) );
                    //Dibuja un circulo
                    circle(drawingVerde, centro1, 10, Scalar(255, 255, 255), 2, 8, 0);
                }
            }
        }

        if (areaMaximaV >= umbralVerde && bandera1 == true){ //Banderas para solo mandar un dato.
            qDebug()<<"Color encontrado verde";
            if(bandera2){
                bandera1 = false;
                bandera2 = false;

                if(arduino_esta_conectado && arduino->isWritable()){
                    arduino->write("c120p");
                }
            }
        }


//***************************************************************************************************************************//

//******Filtrado del color Azul*********************************************************************************************//
        inRange(IMAGENchica, Scalar(colorMinA), Scalar(colorMaxA), ImagenFiltradaAzul);
        vector<vector<Point> > contornosAzul;
        vector<Vec4i> jerarquiaAzul;

        Mat copiaImagenFiltradaAzul;
        copiaImagenFiltradaAzul = ImagenFiltradaAzul;
        findContours(copiaImagenFiltradaAzul, contornosAzul, jerarquiaAzul, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) ); //Funcion para encontrar contornos, depende del modo.
        drawingAzul = Mat::zeros(copiaImagenFiltradaAzul.size(), CV_8UC3 ); //Matriz negra de tres canales.

        area = 0;
        indiceJ = -1;

        for(int i = 0; i < (int)contornosAzul.size(); i++){ //Para encontrar todos los contornos.
            if(jerarquiaAzul [i][2]!= -1) { //Para encontrar jerarquias, buscando padres.
                drawContours (drawingAzul, contornosAzul, i, Scalar(255, 0, 0), 1, 8, jerarquiaAzul, 0); //Dibuja el contorno.
                area = contourArea(contornosAzul[i]); //Calcula el area.

                if(area > areaMaximaA){
                    areaMaximaA = area;
                    indiceJ = i;
                }

                if(indiceJ >= 0){
                    Moments momento2;
                    //Encontrar el momento del contorno indice J.
                    momento2 = moments(contornosAzul[indiceJ]);
                    //Calcular el centro del momento
                    centro2 = Point2f( static_cast<float>(momento2.m10/momento2.m00), static_cast<float>(momento2.m01/momento2.m00));
                    //Dibuja un circulo
                    circle(drawingAzul, centro2, 10, Scalar(255, 255, 255), 2, 8, 0);
                }
            }
        }

        if (areaMaximaA >= umbralAzul && bandera1 == true){ //Banderas para solo mandar un dato.
            qDebug()<<"Color encontrado azul";
            if(bandera2){
                bandera1 = false;
                bandera2 = false;

                if(arduino_esta_conectado && arduino->isWritable()){
                    arduino->write("b50p");
                }
            }
        }

//***************************************************************************************************************************//

       if(ui->checkBox_2->isChecked()){
           if(centro.x < 110){
               primeraVez = true;
           }

           else{
               if(primeraVez){
                   contadorRojo++;
                   ui->lcdNumberRojo->display(contadorRojo);
                   primeraVez = false;
               }
           }

           if(centro1.x < 110){
               primeraVez1 = true;
           }

           else{
               if(primeraVez1){
                   contadorVerde++;
                   ui->lcdNumberVerde->display(contadorVerde);
                   primeraVez1 = false;

               }
           }

           if(centro2.x < 110){
               primeraVez2 = true;
           }

           else{
               if(primeraVez2){
                   contadorAzul++;
                   ui->lcdNumberAzul->display(contadorAzul);
                   primeraVez2 = false;
               }
           }
       }

        //Dibujar linea

        switch(canal){
            case 0:
                ImagenFiltrada = ImagenFiltradaRoja;
                drawing = drawingRoja;
                line(drawing, Point(110,0), Point(110,149), Scalar(255,255,255), 2, 8, 0);
            break;
            case 1:
                ImagenFiltrada = ImagenFiltradaVerde;
                drawing = drawingVerde;
                line(drawing, Point(110,0), Point(110,149), Scalar(255,255,255), 2, 8, 0);
                break;
            case 2:
                ImagenFiltrada = ImagenFiltradaAzul;
                drawing = drawingAzul;
                line(drawing, Point(110,0), Point(110,149), Scalar(255,255,255), 2, 8, 0);
                break;
            }


        //Mostrar las imagenes en sus respectivas etiquetas
        QImage qImage = Mat2QImage(drawing);
        QPixmap pixmap = QPixmap::fromImage(qImage);
        ui->labelContornos->clear();
        ui->labelContornos->setPixmap(pixmap);

        qImage = Mat2QImage(ImagenFiltrada);
        pixmap = QPixmap::fromImage(qImage);
        ui->labelInrange->clear();
        ui->labelInrange->setPixmap(pixmap);

    }
}

void MainWindow::recepcionSerialAsyncrona(){ //Funcion comunicacion asincrona

    if(arduino_esta_conectado && arduino->isReadable()){ //Verifica si el arduino esta conectado y si se puede leer.
        QByteArray datosLeidos = arduino->readLine(); //Lee una linea de texto y la asigna a un arreglo de bytes "datosLeidos".
        //qDebug() << datosLeidos << endl;
        bandera1 = true;
    }
}

void MainWindow::conectarArduino(){ //Funcion para buscar y conectar Arduino.

    //Parte #1 - Declaracion inicial de las variables
    arduino_esta_conectado = false; //Si arduino se puede conectar, cambia a true.
    arduino = new QSerialPort(this); //Crea un puerto serial virutal QT.
    connect(arduino, &QSerialPort::readyRead, this, &MainWindow::recepcionSerialAsyncrona); //Conecta el puerto virtual para ser leido y con comunicacion serial asincrona.
    QString nombreDispositivoSerial = ""; //Guarda el ultimo puerto que encuentre la parte 2.
    int nombreProductID = 0; //Guarda el "product ID" del arduino conectado MEGA o UNO R3.

    //Parte #2 - Buscar puertos con los identificadores de Arduino
    qDebug() << "Puertos disponibles: " << QSerialPortInfo::availablePorts().length(); //Muestra en el depurador el numero de puertos disponibles.
    foreach (const QSerialPortInfo &serialPortInfo, QSerialPortInfo::availablePorts()){ // Ejecucion para cuando exista informacion sobre puertos existentes.
        qDebug() << "Identificador del fabricante (VENDOR ID): " << serialPortInfo.hasVendorIdentifier(); //Muestra en el depurador si el vendroID fue identificado true o false.
        if(serialPortInfo.hasVendorIdentifier()){ //Verifica si el vendor de arduino fue identificado.
            qDebug() << "ID Vendedor " << serialPortInfo.vendorIdentifier(); //Muestra en el depurador el "ID Vendedor".
            qDebug() << "ID Producto: " << serialPortInfo.productIdentifier(); //Muestra en el depurador el "ID Producto"
            if(serialPortInfo.productIdentifier() == 66 || serialPortInfo.productIdentifier() == 67){ //Verifica si el "ID producto" corresponde a un arduino MEGA o UNO R3.
                arduino_esta_conectado = true; //Si arduino esta conectado.
                nombreDispositivoSerial = serialPortInfo.portName(); //Guarda el "portName" en la variable "nombreDispositivoSerial".
                nombreProductID = serialPortInfo.productIdentifier(); //Guarda el "productIdentifier" en la variable "nombreProductID".
            }
        }
    }

    //Parte #3 - Conexion
    if(arduino_esta_conectado){ //Verifica si el arduino esta conectado.
        arduino ->setPortName(nombreDispositivoSerial); //Asigna al puerto el nombre guardado en "nombreDispositivoSerial".
        arduino->open(QIODevice::ReadWrite); //Abre el puerto con permisos de lecura y escritura.
        arduino->setDataBits(QSerialPort::Data8); //Configura el puerto con 8 bits de datos.
        arduino ->setBaudRate(QSerialPort::Baud115200); //Configura la velocidad de transmision.
        arduino->setParity(QSerialPort::NoParity); //Configura el bit de paridad.
        arduino->setStopBits(QSerialPort::OneStop); //Configura el bit de parada.
        arduino->setFlowControl(QSerialPort::NoFlowControl); //Configura el control de flujo.
        ui->label_A->clear(); //Limpia la etiqueta donde se muestra el nombre del dispositivo conectado.
        qDebug() << "Producto: " << nombreProductID; //Muestra en el depurador "nombreProductID".
        if(nombreProductID == 67){
            ui->label_A->setText("Arduino UNO R3 conectado"); //Verifica si "nombreProductID" corresponde 66 o 67 y escribira en la etiqueta el nombre del correspodiente arduino.
        }
        else if(nombreProductID == 66){
            ui->label_A->setText("Arduino Mega conectado");
        }
        else {
            ui->label_A->setText("Error 3"); // En caso de no ser 66 o 67 escribira en al etiqueta "error 3".
        }
    }

    else { //En caso de no estar conectado ningun arduino.
        ui->label_A->clear(); //Limpia la etiqueta donde se muestra el nombre del dispositivo conectado.
        ui->label_A->setText("No hay arduino");
    }
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QTimer *cronometro = new QTimer(this);
    connect(cronometro,SIGNAL(timeout()),this,SLOT(ftimer()));
    cronometro->start(100);

    ui->barraCanal0Min->setValue(canal0Min[0]);
    ui->barraCanal0Max->setValue(canal0Max[0]);
    ui->barraCanal1Min->setValue(canal1Min[0]);
    ui->barraCanal1Max->setValue(canal1Max[0]);
    ui->barraCanal2Min->setValue(canal2Min[0]);
    ui->barraCanal2Max->setValue(canal2Max[0]);

}

MainWindow::~MainWindow()
{
    arduino->close(); //Cierra el puerto.
    delete ui;
}


void MainWindow::on_barraCanal0Min_valueChanged(int value)
{
    ui->lcdCanal0Min->display(value);
}

void MainWindow::on_barraCanal0Max_valueChanged(int value)
{
    ui->lcdCanal0Max->display(value);
}

void MainWindow::on_barraCanal1Min_valueChanged(int value)
{
    ui->lcdCanal1Min->display(value);
}

void MainWindow::on_barraCanal1Max_valueChanged(int value)
{
    ui->lcdCanal1Max->display(value);
}

void MainWindow::on_barraCanal2Min_valueChanged(int value)
{
    ui->lcdCanal2Min->display(value);
}

void MainWindow::on_barraCanal2Max_valueChanged(int value)
{
    ui->lcdCanal2Max->display(value);
}

void MainWindow::on_buttonBuscar_clicked()
{
    conectarArduino();
    ui->buttonIniciar->setEnabled(true);
    ui->checkBox->setEnabled(true);
    ui->checkBox_2->setEnabled(true);
    ui->commandLinkButton->setEnabled(true);

}

void MainWindow::on_buttonIniciar_clicked()
{
    if(arduino_esta_conectado && arduino->isWritable()){
        arduino->write("ap");
    }
}

void MainWindow::on_radioButtonRojo_clicked()
{
    ui->barraCanal0Min->setValue(canal0Min[0]);
    ui->barraCanal0Max->setValue(canal0Max[0]);
    ui->barraCanal1Min->setValue(canal1Min[0]);
    ui->barraCanal1Max->setValue(canal1Max[0]);
    ui->barraCanal2Min->setValue(canal2Min[0]);
    ui->barraCanal2Max->setValue(canal2Max[0]);
}

void MainWindow::on_radioButtonVerde_clicked()
{
    ui->barraCanal0Min->setValue(canal0Min[1]);
    ui->barraCanal0Max->setValue(canal0Max[1]);
    ui->barraCanal1Min->setValue(canal1Min[1]);
    ui->barraCanal1Max->setValue(canal1Max[1]);
    ui->barraCanal2Min->setValue(canal2Min[1]);
    ui->barraCanal2Max->setValue(canal2Max[1]);
}

void MainWindow::on_radioButtonAzul_clicked()
{
    ui->barraCanal0Min->setValue(canal0Min[2]);
    ui->barraCanal0Max->setValue(canal0Max[2]);
    ui->barraCanal1Min->setValue(canal1Min[2]);
    ui->barraCanal1Max->setValue(canal1Max[2]);
    ui->barraCanal2Min->setValue(canal2Min[2]);
    ui->barraCanal2Max->setValue(canal2Max[2]);
}

void MainWindow::on_commandLinkButton_clicked()
{
    contadorRojo = 0;
    contadorVerde = 0;
    contadorAzul = 0;
    ui->lcdNumberAzul->display(0);
    ui->lcdNumberVerde->display(0);
    ui->lcdNumberRojo->display(0);
}
