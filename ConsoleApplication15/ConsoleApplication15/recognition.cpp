#include <pqxx/pqxx>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include "database.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Настройки
const float FACE_CONFIDENCE_THRESHOLD = 0.7f;
const Size PROCESSING_SIZE(640, 480);

// Получение эмбеддинга + нормализация L2
vector<float> get_face_embedding(Net& net, Mat face) {
    Mat blob;
    resize(face, face, Size(112, 112));
    cvtColor(face, face, COLOR_BGR2RGB);
    blobFromImage(face, blob, 1.0 / 127.5, Size(112, 112),
        Scalar(127.5, 127.5, 127.5), true, false);

    net.setInput(blob);
    Mat embedding = net.forward();

    vector<float> vec(embedding.ptr<float>(),
        embedding.ptr<float>() + embedding.total());

    // L2 нормализация
    float norm = 0;
    for (float v : vec) norm += v * v;
    norm = sqrt(norm);
    for (float& v : vec) v /= norm;

    return vec;
}

int main() {
    // Загрузка моделей
    Net net = readNetFromONNX("res/arcface.onnx");
    Net faceDetector = readNetFromCaffe("res/deploy.prototxt",
        "res/res10_300x300_ssd_iter_140000.caffemodel");

    // Подключение к БД
    pqxx::connection conn("dbname=face_recognition user=postgres password=1234 host=localhost");

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, PROCESSING_SIZE.width);
    cap.set(CAP_PROP_FRAME_HEIGHT, PROCESSING_SIZE.height);

    Mat frame;
    bool show_help = true;
    vector<Rect> last_face_locations;
    vector<vector<float>> last_face_embeddings;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Детекция лиц
        Mat resizedFrame;
        resize(frame, resizedFrame, PROCESSING_SIZE);

        Mat blob = blobFromImage(resizedFrame, 1.0, Size(300, 300),
            Scalar(104, 177, 123), false, false);
        faceDetector.setInput(blob);
        Mat detections = faceDetector.forward();

        last_face_locations.clear();
        last_face_embeddings.clear();

        const float* data = detections.ptr<float>();
        const int num_detections = detections.size[2];

        for (int i = 0; i < num_detections; ++i) {
            float confidence = data[i * 7 + 2];
            if (confidence > FACE_CONFIDENCE_THRESHOLD) {
                const float* det = data + i * 7;
                int x1 = static_cast<int>(det[3] * resizedFrame.cols);
                int y1 = static_cast<int>(det[4] * resizedFrame.rows);
                int x2 = static_cast<int>(det[5] * resizedFrame.cols);
                int y2 = static_cast<int>(det[6] * resizedFrame.rows);

                Rect faceRect(x1, y1, x2 - x1, y2 - y1);
                if (faceRect.x < 0 || faceRect.y < 0 ||
                    faceRect.x + faceRect.width > resizedFrame.cols ||
                    faceRect.y + faceRect.height > resizedFrame.rows) {
                    continue;
                }

                Mat face = resizedFrame(faceRect).clone();

                // Получаем эмбеддинг
                vector<float> embedding = get_face_embedding(net, face);

                // Масштабируем координаты обратно
                Rect scaledRect(
                    faceRect.x * frame.cols / resizedFrame.cols,
                    faceRect.y * frame.rows / resizedFrame.rows,
                    faceRect.width * frame.cols / resizedFrame.cols,
                    faceRect.height * frame.rows / resizedFrame.rows
                );

                last_face_locations.push_back(scaledRect);
                last_face_embeddings.push_back(embedding);
            }
        }

        // Проверка лиц в базе и отрисовка
        for (size_t i = 0; i < last_face_locations.size(); ++i) {
            auto [name, similarity] = check_face_in_db(conn, last_face_embeddings[i]);

            rectangle(frame, last_face_locations[i], Scalar(0, 255, 0), 2);

            putText(frame, name,
                Point(last_face_locations[i].x, last_face_locations[i].y - 10),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

            putText(frame, format("%.0f", similarity),
                Point(last_face_locations[i].x,
                    last_face_locations[i].y + last_face_locations[i].height + 15),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);
        }

        // Отображение помощи
        if (show_help) {
            putText(frame, "Help:", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
            putText(frame, "'a' - Add face", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
            putText(frame, "'d' - Delete face", Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
            putText(frame, "'h' - Toggle help", Point(10, 120), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
            putText(frame, "'q' - Quit", Point(10, 150), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
        }

        imshow("Face Recognition", frame);

        int key = waitKey(30);
        if (key == 'q') break;
        else if (key == 'h') show_help = !show_help;
        else if (key == 'a' && !last_face_locations.empty()) {
            cout << "Enter name to add: ";
            string name;
            getline(cin, name);

            if (!name.empty()) {
                add_face_to_db(conn, name, last_face_embeddings[0]);
                cout << "Face added to database!" << endl;
            }
        }
        else if (key == 'd' && !last_face_locations.empty()) {
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

    cap.release();
    return 0;
}
