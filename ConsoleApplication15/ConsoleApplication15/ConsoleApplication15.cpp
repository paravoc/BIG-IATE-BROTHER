#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include"PostgresFaceDB.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;
namespace fs = filesystem;

// Загрузка модели ArcFace
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
    Net detector = readNetFromCaffe("res/deploy.prototxt", "res/res10_300x300_ssd_iter_140000.caffemodel");

    for (const auto& entry : fs::directory_iterator(facesPath)) {
        if (entry.is_regular_file()) {
            Mat img = imread(entry.path().string());
            if (!img.empty()) {
                Mat blob = blobFromImage(img, 1.0, Size(300, 300), Scalar(104, 177, 123));
                detector.setInput(blob);
                Mat detections = detector.forward();

                const float* data = detections.ptr<float>();
                const int num_detections = detections.size[2];

                for (int i = 0; i < num_detections; ++i) {
                    float confidence = data[i * 7 + 2];
                    if (confidence > 0.7) {
                        const float* det = data + i * 7;
                        int x1 = static_cast<int>(det[3] * img.cols);
                        int y1 = static_cast<int>(det[4] * img.rows);
                        int x2 = static_cast<int>(det[5] * img.cols);
                        int y2 = static_cast<int>(det[6] * img.rows);

                        // Корректировка координат
                        x1 = max(0, x1); y1 = max(0, y1);
                        x2 = min(img.cols - 1, x2);
                        y2 = min(img.rows - 1, y2);

                        if (x2 > x1 && y2 > y1) {
                            Rect faceRect(x1, y1, x2 - x1, y2 - y1);
                            Mat face = img(faceRect).clone();
                            vector<Mat> embeddings = getFaceEmbeddings(arcface, { face });

                            if (!embeddings.empty()) {
                                knownEncodings.push_back(embeddings[0]);
                                knownNames.push_back(entry.path().stem().string());
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Косинусное сходство
double cosineSimilarity(const Mat& a, const Mat& b) {
    return a.dot(b) / (norm(a) * norm(b));
}

int main() {

    PostgresFaceDB faceDB("dbname=face_recognition user=postgres password=1234");
    if (!faceDB.initialize()) {
        std::cerr << "Failed to initialize database" << std::endl;
        return -1;
    }
    // Инициализация моделей
    string arcfacePath = "res/arcface.onnx";
    string facesPath = "faces";

    Net arcface = loadArcFaceModel(arcfacePath);
    vector<Mat> knownEncodings;
    vector<string> knownNames;
    loadKnownFaces(facesPath, arcface, knownEncodings, knownNames);

    if (knownEncodings.empty()) {
        cerr << "No known faces loaded!" << endl;
        return -1;
    }

    // Инициализация детектора лиц
    Net faceDetector = readNetFromCaffe("res/deploy.prototxt", "res/res10_300x300_ssd_iter_140000.caffemodel");
    if (faceDetector.empty()) {
        cerr << "Error loading face detector!" << endl;
        return -1;
    }

    // Включение GPU-ускорения (если доступно)
    faceDetector.setPreferableBackend(DNN_BACKEND_CUDA);
    faceDetector.setPreferableTarget(DNN_TARGET_CUDA);
    arcface.setPreferableBackend(DNN_BACKEND_CUDA);
    arcface.setPreferableTarget(DNN_TARGET_CUDA);

    // Инициализация видеопотока
    VideoCapture video(0);
    if (!video.isOpened()) {
        cerr << "Error opening camera!" << endl;
        return -1;
    }

    Mat frame;
    while (video.read(frame)) {
        // Детекция лиц
        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123));
        faceDetector.setInput(blob);
        Mat detections = faceDetector.forward();

        vector<Rect> faceLocations;
        const float* data = detections.ptr<float>();
        const int num_detections = detections.size[2];

        for (int i = 0; i < num_detections; ++i) {
            float confidence = data[i * 7 + 2];
            if (confidence > 0.7) {
                const float* det = data + i * 7;
                int x1 = static_cast<int>(det[3] * frame.cols);
                int y1 = static_cast<int>(det[4] * frame.rows);
                int x2 = static_cast<int>(det[5] * frame.cols);
                int y2 = static_cast<int>(det[6] * frame.rows);

                // Корректировка координат
                x1 = max(0, x1); y1 = max(0, y1);
                x2 = min(frame.cols - 1, x2);
                y2 = min(frame.rows - 1, y2);

                if (x2 > x1 && y2 > y1 && (x2 - x1) > 50 && (y2 - y1) > 50) {
                    faceLocations.emplace_back(x1, y1, x2 - x1, y2 - y1);
                }
            }
        }

        // Обработка каждого лица
        for (const auto& faceLoc : faceLocations) {
            Mat face = frame(faceLoc).clone();
            vector<Mat> embeddings = getFaceEmbeddings(arcface, { face });

            if (!embeddings.empty()) {
                Mat unknownEncoding = embeddings[0];
                string name = "Unknown";
                double bestSimilarity = 0.6; // Порог распознавания

                for (size_t i = 0; i < knownEncodings.size(); ++i) {
                    double similarity = cosineSimilarity(unknownEncoding, knownEncodings[i]);
                    if (similarity > bestSimilarity) {
                        bestSimilarity = similarity;
                        name = knownNames[i];
                    }
                }

                // Визуализация результатов
                auto [name, confidence] = faceDB.identifyFace(embeddings[0]);
                rectangle(frame, faceLoc, Scalar(0, 255, 0), 2);
                putText(frame, name, Point(faceLoc.x, faceLoc.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

                putText(frame, format("%.2f", bestSimilarity),
                    Point(faceLoc.x, faceLoc.y + faceLoc.height + 20),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);
            }
        }

        // Отображение количества лиц
        putText(frame, format("Faces: %d", (int)faceLocations.size()),
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

        imshow("Face Recognition", frame);
        if (waitKey(1) == 'q') break;
    }

    return 0;
}