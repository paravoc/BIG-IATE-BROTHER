#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace cv::dnn;
using namespace std;
namespace fs = filesystem;

// Загрузка модели для извлечения признаков лиц
Net loadArcFaceModel(const string& modelPath) {
    Net net = readNetFromONNX(modelPath);
    if (net.empty()) {
        cerr << "Failed to load ArcFace model!" << endl;
        exit(-1);
    }
    return net;
}

// Получение эмбеддингов лиц
vector<Mat> getFaceEmbeddings(Net& arcface, const vector<Mat>& faces) {
    vector<Mat> embeddings;
    for (const auto& face : faces) {
        Mat rgb;
        cvtColor(face, rgb, COLOR_BGR2RGB);
        resize(rgb, rgb, Size(112, 112));
        Mat blob = blobFromImage(rgb, 1.0 / 128.0, Size(112, 112), Scalar(127.5, 127.5, 127.5), true, false);
        arcface.setInput(blob);
        embeddings.push_back(arcface.forward().clone());
    }
    return embeddings;
}

// Загрузка базы известных лиц
void loadKnownFaces(const string& facesPath, Net& arcface, vector<Mat>& knownEncodings, vector<string>& knownNames) {
    for (const auto& entry : fs::directory_iterator(facesPath)) {
        if (entry.is_regular_file()) {
            Mat img = imread(entry.path().string());
            if (!img.empty()) {
                // Детекция лица (здесь можно использовать Haar Cascade или другую модель)
                Rect faceRect(0, 0, img.cols, img.rows); // Просто используем все изображение
                Mat face = img(faceRect).clone();

                // Получаем эмбеддинг
                vector<Mat> embeddings = getFaceEmbeddings(arcface, { face });
                if (!embeddings.empty()) {
                    knownEncodings.push_back(embeddings[0]);
                    knownNames.push_back(entry.path().stem().string());
                }
            }
        }
    }
}

// Сравнение эмбеддингов (косинусное сходство)
double cosineSimilarity(const Mat& a, const Mat& b) {
    return a.dot(b) / (norm(a) * norm(b));
}

int main() {
    // Пути к моделям и базе лиц
    string arcfacePath = "arcface.onnx";
    string facesPath = "faces";

    // Загрузка модели ArcFace
    Net arcface = loadArcFaceModel(arcfacePath);

    // Загрузка базы известных лиц
    vector<Mat> knownEncodings;
    vector<string> knownNames;
    loadKnownFaces(facesPath, arcface, knownEncodings, knownNames);

    if (knownEncodings.empty()) {
        cerr << "No known faces loaded!" << endl;
        return -1;
    }

    // Инициализация видеопотока
    VideoCapture video(0);
    if (!video.isOpened()) {
        cerr << "Error opening camera!" << endl;
        return -1;
    }

    // Масштаб для уменьшения кадра (для производительности)
    float scale = 2.0;

    Mat frame;
    while (video.read(frame)) {
        // Уменьшение размера кадра
        Mat resizedFrame;
        resize(frame, resizedFrame, Size(frame.cols / scale, frame.rows / scale));

        // Детекция лиц (используем Haar Cascade для простоты)
        CascadeClassifier faceCascade;
        if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
            cerr << "Error loading face cascade!" << endl;
            return -1;
        }

        vector<Rect> faceLocations;
        Mat gray;
        cvtColor(resizedFrame, gray, COLOR_BGR2GRAY);
        faceCascade.detectMultiScale(gray, faceLocations, 1.1, 4);

        // Обработка каждого обнаруженного лица
        for (const auto& faceLoc : faceLocations) {
            // Получение эмбеддинга для текущего лица
            Mat face = resizedFrame(faceLoc).clone();
            vector<Mat> embeddings = getFaceEmbeddings(arcface, { face });

            if (!embeddings.empty()) {
                Mat unknownEncoding = embeddings[0];

                // Сравнение с известными лицами
                string name = "Unknown";
                double bestSimilarity = 0.4; // Порог распознавания

                for (size_t i = 0; i < knownEncodings.size(); ++i) {
                    double similarity = cosineSimilarity(unknownEncoding, knownEncodings[i]);
                    if (similarity > bestSimilarity) {
                        bestSimilarity = similarity;
                        name = knownNames[i];
                    }
                }

                // Отрисовка результатов
                Rect scaledFaceLoc(
                    faceLoc.x * scale,
                    faceLoc.y * scale,
                    faceLoc.width * scale,
                    faceLoc.height * scale
                );

                rectangle(frame, scaledFaceLoc, Scalar(0, 0, 255), 2);
                putText(frame, name,
                    Point(scaledFaceLoc.x, scaledFaceLoc.y + scaledFaceLoc.height + 20),
                    FONT_HERSHEY_DUPLEX, 0.8, Scalar(255, 255, 255), 1);
            }
        }

        // Отображение количества лиц
        string facesCount = "Faces: " + to_string(faceLocations.size());
        putText(frame, facesCount, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow("Face Recognition", frame);

        if (waitKey(1) == 'q') break;
    }

    return 0;
}