#include <pqxx/pqxx>          // Библиотека для работы с PostgreSQL
#include <opencv2/opencv.hpp> // Основные модули OpenCV
#include <opencv2/dnn.hpp>    // Модуль OpenCV для нейросетей (DNN)
#include <vector>
#include <cmath>
#include <iostream>
#include "database.h"         // Пользовательский хедер для работы с БД

using namespace cv;
using namespace cv::dnn;
using namespace std;

// ---------------------------
// Настройки программы
// ---------------------------
const float FACE_CONFIDENCE_THRESHOLD = 0.7f; // Минимальная уверенность детектора лиц для обработки
const Size PROCESSING_SIZE(640, 480);         // Размер кадра для обработки (снижение нагрузки на нейросети)

// ---------------------------
// Функция получения эмбеддинга лица + L2-нормализация
// ---------------------------
vector<float> get_face_embedding(Net& net, Mat face) {
    Mat blob;

    // Изменяем размер лица до 112x112 (размер модели ArcFace)
    resize(face, face, Size(112, 112));

    // Переводим BGR в RGB (модель обучена на RGB)
    cvtColor(face, face, COLOR_BGR2RGB);

    // Создаем blob из изображения для подачи в DNN
    // Нормализация: (x - 127.5) / 127.5, swapRB=true
    blobFromImage(face, blob, 1.0 / 127.5, Size(112, 112),
        Scalar(127.5, 127.5, 127.5), true, false);

    // Передаем blob в нейросеть ArcFace
    net.setInput(blob);

    // Получаем эмбеддинг (выход слоя)
    Mat embedding = net.forward();

    // Перевод Mat в вектор float
    vector<float> vec(embedding.ptr<float>(), embedding.ptr<float>() + embedding.total());

    // ---------------------------
    // L2-нормализация (вектор единичной длины)
    // ---------------------------
    float norm = 0;
    for (float v : vec) norm += v * v; // Сумма квадратов
    norm = sqrt(norm);                 // Корень суммы квадратов
    for (float& v : vec) v /= norm;    // Делим каждый элемент на норму

    return vec; // Возвращаем нормализованный эмбеддинг
}

// ---------------------------
// Основная функция
// ---------------------------
int main() {
    // Загружаем модели
    Net net = readNetFromONNX("res/arcface.onnx"); // Модель ArcFace для эмбеддингов
    Net faceDetector = readNetFromCaffe("res/deploy.prototxt",
        "res/res10_300x300_ssd_iter_140000.caffemodel"); // SSD для детекции лиц

    // Подключаемся к PostgreSQL
    pqxx::connection conn("dbname=face_recognition user=postgres password=1234 host=localhost");

    // Захват видео с камеры
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return -1;
    }

    // Устанавливаем размер кадров для камеры
    cap.set(CAP_PROP_FRAME_WIDTH, PROCESSING_SIZE.width);
    cap.set(CAP_PROP_FRAME_HEIGHT, PROCESSING_SIZE.height);

    Mat frame;                     // Текущий кадр
    bool show_help = true;         // Флаг отображения подсказок
    vector<Rect> last_face_locations;       // Последние координаты лиц
    vector<vector<float>> last_face_embeddings; // Последние эмбеддинги лиц

    // Главный цикл
    while (true) {
        cap >> frame;              // Считываем кадр
        if (frame.empty()) break;  // Если кадр пустой — выходим

        // ---------------------------
        // Детекция лиц на кадре
        // ---------------------------
        Mat resizedFrame;
        resize(frame, resizedFrame, PROCESSING_SIZE); // уменьшаем размер для скорости

        Mat blob = blobFromImage(resizedFrame, 1.0, Size(300, 300),
            Scalar(104, 177, 123), false, false); // Blob для SSD
        faceDetector.setInput(blob);
        Mat detections = faceDetector.forward();    // Прогон через SSD

        last_face_locations.clear();   // Очищаем предыдущие лица
        last_face_embeddings.clear();  // Очищаем предыдущие эмбеддинги

        const float* data = detections.ptr<float>(); // Доступ к данным детекции
        const int num_detections = detections.size[2]; // Количество обнаружений

        // Перебор всех детекций
        for (int i = 0; i < num_detections; ++i) {
            float confidence = data[i * 7 + 2]; // Уверенность детекции
            if (confidence > FACE_CONFIDENCE_THRESHOLD) { // Только уверенные лица
                const float* det = data + i * 7;
                int x1 = static_cast<int>(det[3] * resizedFrame.cols);
                int y1 = static_cast<int>(det[4] * resizedFrame.rows);
                int x2 = static_cast<int>(det[5] * resizedFrame.cols);
                int y2 = static_cast<int>(det[6] * resizedFrame.rows);

                Rect faceRect(x1, y1, x2 - x1, y2 - y1);

                // Проверка границ лица, чтобы не выйти за кадр
                if (faceRect.x < 0 || faceRect.y < 0 ||
                    faceRect.x + faceRect.width > resizedFrame.cols ||
                    faceRect.y + faceRect.height > resizedFrame.rows) {
                    continue;
                }

                Mat face = resizedFrame(faceRect).clone(); // Вырезаем лицо

                // Получаем эмбеддинг лица
                vector<float> embedding = get_face_embedding(net, face);

                // Масштабируем координаты обратно к исходному кадру
                Rect scaledRect(
                    faceRect.x * frame.cols / resizedFrame.cols,
                    faceRect.y * frame.rows / resizedFrame.rows,
                    faceRect.width * frame.cols / resizedFrame.cols,
                    faceRect.height * frame.rows / resizedFrame.rows
                );

                last_face_locations.push_back(scaledRect); // Сохраняем координаты
                last_face_embeddings.push_back(embedding); // Сохраняем эмбеддинг
            }
        }

        // ---------------------------
        // Проверка лиц в базе и отрисовка
        // ---------------------------
        for (size_t i = 0; i < last_face_locations.size(); ++i) {
            auto [name, similarity] = check_face_in_db(conn, last_face_embeddings[i]);

            rectangle(frame, last_face_locations[i], Scalar(0, 255, 0), 2); // Рисуем прямоугольник вокруг лица

            putText(frame, name,
                Point(last_face_locations[i].x, last_face_locations[i].y - 10),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2); // Имя

            putText(frame, format("%.0f", similarity),
                Point(last_face_locations[i].x,
                    last_face_locations[i].y + last_face_locations[i].height + 15),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1); // Процент сходства
        }

        // ---------------------------
        // Отображение помощи на экране
        // ---------------------------
        if (show_help) {
            putText(frame, "Help:", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
            putText(frame, "'a' - Add face", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
            putText(frame, "'d' - Delete face", Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
            putText(frame, "'h' - Toggle help", Point(10, 120), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
            putText(frame, "'q' - Quit", Point(10, 150), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
        }

        imshow("Face Recognition", frame); // Показываем кадр

        // ---------------------------
        // Обработка клавиш
        // ---------------------------
        int key = waitKey(30);
        if (key == 'q') break;            // Выход
        else if (key == 'h') show_help = !show_help; // Переключение подсказки
        else if (key == 'a' && !last_face_locations.empty()) { // Добавление лица
            cout << "Enter name to add: ";
            string name;
            getline(cin, name);

            if (!name.empty()) {
                add_face_to_db(conn, name, last_face_embeddings[0]);
                cout << "Face added to database!" << endl;
            }
        }
        else if (key == 'd' && !last_face_locations.empty()) { // Удаление лица
            auto [name, similarity] = check_face_in_db(conn, last_face_embeddings[0]);
            if (name != "Unknown") {
                delete_face_from_db(conn, name);
                cout << "Face '" << name << "' deleted from database!" << endl;
            }
            else {
                cout << "No matching face in database to delete" << endl;
            }
        }
    }

    cap.release(); // Освобождаем камеру
    return 0;
}
