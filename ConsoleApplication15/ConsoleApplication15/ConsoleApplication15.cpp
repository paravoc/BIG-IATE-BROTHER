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

// Настройки производительности
const float FACE_CONFIDENCE_THRESHOLD = 0.7f;
const float RECOGNITION_THRESHOLD = 0.6f;
const int FACE_MIN_SIZE = 50;
const Size PROCESSING_SIZE(640, 480);  // Разрешение для обработки

// Загрузка модели ArcFace
Net loadArcFaceModel(const string& modelPath) {
    Net net = readNetFromONNX(modelPath);
    if (net.empty()) {
        cerr << "Ошибка загрузки модели ArcFace!" << endl;
        exit(-1);
    }
    return net;
}

// Получение эмбеддингов лиц
vector<Mat> getFaceEmbeddings(Net& arcface, const vector<Mat>& faces) {
    vector<Mat> embeddings;
    embeddings.reserve(faces.size());

    for (const auto& face : faces) {
        Mat rgb;
        cvtColor(face, rgb, COLOR_BGR2RGB);
        resize(rgb, rgb, Size(112, 112));

        Mat blob;
        blobFromImage(rgb, blob, 1.0 / 128.0, Size(112, 112), Scalar(127.5, 127.5, 127.5), true, false);

        arcface.setInput(blob);
        embeddings.emplace_back(arcface.forward().clone());
    }
    return embeddings;
}

// Загрузка базы известных лиц
void loadKnownFaces(const string& facesPath, Net& arcface, vector<Mat>& knownEncodings, vector<string>& knownNames) {
    Net detector = readNetFromCaffe("res/deploy.prototxt", "res/res10_300x300_ssd_iter_140000.caffemodel");
    detector.setPreferableBackend(DNN_BACKEND_CUDA);
    detector.setPreferableTarget(DNN_TARGET_CUDA);

    for (const auto& entry : fs::directory_iterator(facesPath)) {
        if (entry.is_regular_file()) {
            Mat img = imread(entry.path().string());
            if (img.empty()) continue;

            Mat resizedImg;
            float scale = 600.f / max(img.cols, img.rows);
            resize(img, resizedImg, Size(), scale, scale);

            Mat blob = blobFromImage(resizedImg, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
            detector.setInput(blob);
            Mat detections = detector.forward();

            const float* data = detections.ptr<float>();
            const int num_detections = detections.size[2];

            for (int i = 0; i < num_detections; ++i) {
                float confidence = data[i * 7 + 2];
                if (confidence > FACE_CONFIDENCE_THRESHOLD) {
                    const float* det = data + i * 7;
                    int x1 = static_cast<int>(det[3] * resizedImg.cols);
                    int y1 = static_cast<int>(det[4] * resizedImg.rows);
                    int x2 = static_cast<int>(det[5] * resizedImg.cols);
                    int y2 = static_cast<int>(det[6] * resizedImg.rows);

                    x1 = max(0, x1); y1 = max(0, y1);
                    x2 = min(resizedImg.cols - 1, x2);
                    y2 = min(resizedImg.rows - 1, y2);

                    if (x2 > x1 && y2 > y1) {
                        Rect faceRect(x1, y1, x2 - x1, y2 - y1);
                        Mat face = resizedImg(faceRect).clone();
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

// Косинусное сходство
float cosineSimilarity(const Mat& a, const Mat& b) {
    return static_cast<float>(a.dot(b) / (norm(a) * norm(b)));
}

int main() {
    // Инициализация моделей
    string arcfacePath = "res/arcface.onnx";
    string facesPath = "faces";

    Net arcface = loadArcFaceModel(arcfacePath);
    arcface.setPreferableBackend(DNN_BACKEND_CUDA);
    arcface.setPreferableTarget(DNN_TARGET_CUDA);

    vector<Mat> knownEncodings;
    vector<string> knownNames;
    loadKnownFaces(facesPath, arcface, knownEncodings, knownNames);

    if (knownEncodings.empty()) {
        cerr << "Не загружены известные лица!" << endl;
        return -1;
    }

    Net faceDetector = readNetFromCaffe("res/deploy.prototxt", "res/res10_300x300_ssd_iter_140000.caffemodel");
    faceDetector.setPreferableBackend(DNN_BACKEND_CUDA);
    faceDetector.setPreferableTarget(DNN_TARGET_CUDA);

    VideoCapture video(0);
    if (!video.isOpened()) {
        cerr << "Ошибка открытия камеры!" << endl;
        return -1;
    }

    // Установка разрешения камеры
    video.set(CAP_PROP_FRAME_WIDTH, PROCESSING_SIZE.width);
    video.set(CAP_PROP_FRAME_HEIGHT, PROCESSING_SIZE.height);

    Mat frame, resizedFrame;
    int frameCount = 0;
    double totalTime = 0;
    double fps = 0;

    while (true) {
        double loopTime = (double)getTickCount();

        if (!video.read(frame)) break;

        // Изменение размера кадра для обработки
        resize(frame, resizedFrame, PROCESSING_SIZE);

        // Детекция лиц
        Mat blob = blobFromImage(resizedFrame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
        faceDetector.setInput(blob);
        Mat detections = faceDetector.forward();

        vector<Rect> faceLocations;
        vector<Mat> currentFaces;
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

                int width = x2 - x1;
                int height = y2 - y1;

                if (width > FACE_MIN_SIZE && height > FACE_MIN_SIZE) {
                    x1 = max(0, x1); y1 = max(0, y1);
                    x2 = min(resizedFrame.cols - 1, x2);
                    y2 = min(resizedFrame.rows - 1, y2);

                    if (x2 > x1 && y2 > y1) {
                        Rect faceRect(x1, y1, width, height);
                        faceLocations.push_back(faceRect);
                        currentFaces.push_back(resizedFrame(faceRect).clone());
                    }
                }
            }
        }

        // Распознавание лиц
        if (!currentFaces.empty()) {
            vector<Mat> embeddings = getFaceEmbeddings(arcface, currentFaces);

            for (size_t i = 0; i < embeddings.size(); ++i) {
                string name = "Unknown";
                float bestSimilarity = RECOGNITION_THRESHOLD;

                for (size_t j = 0; j < knownEncodings.size(); ++j) {
                    float similarity = cosineSimilarity(embeddings[i], knownEncodings[j]);
                    if (similarity > bestSimilarity) {
                        bestSimilarity = similarity;
                        name = knownNames[j];
                    }
                }

                // Масштабирование координат обратно к исходному кадру
                Rect scaledRect(
                    static_cast<int>(faceLocations[i].x * frame.cols / resizedFrame.cols),
                    static_cast<int>(faceLocations[i].y * frame.rows / resizedFrame.rows),
                    static_cast<int>(faceLocations[i].width * frame.cols / resizedFrame.cols),
                    static_cast<int>(faceLocations[i].height * frame.rows / resizedFrame.rows)
                );

                // Отрисовка результатов
                rectangle(frame, scaledRect, Scalar(0, 255, 0), 2);
                putText(frame, name, Point(scaledRect.x, scaledRect.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
                putText(frame, format("%.2f", bestSimilarity),
                    Point(scaledRect.x, scaledRect.y + scaledRect.height + 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
            }
        }

        // Расчет FPS
        frameCount++;
        double currentTime = ((double)getTickCount() - loopTime) / getTickFrequency();
        totalTime += currentTime;

        if (frameCount % 10 == 0) {
            fps = 10 / totalTime;
            totalTime = 0;
        }

        // Отображение информации
        putText(frame, format("FPS: %.1f", fps), Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        putText(frame, format("Faces: %d", (int)faceLocations.size()),
            Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);

        imshow("Face Recognition", frame);
        if (waitKey(1) == 'q') break;
    }

    video.release();
    return 0;
}