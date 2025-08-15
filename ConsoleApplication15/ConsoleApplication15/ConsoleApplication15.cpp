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

/*
 * Загрузка модели ArcFace для извлечения признаков лиц
 * modelPath - путь к ONNX-файлу модели
 * Возвращает: инициализированную нейросеть
 * Если загрузка не удалась - программа завершается с ошибкой
 */
Net loadArcFaceModel(const string& modelPath) {
    Net net = readNetFromONNX(modelPath);
    if (net.empty()) {
        cerr << "Failed to load ArcFace model!" << endl;
        exit(-1);
    }
    return net;
}

/*
 * Получение векторных представлений (эмбеддингов) лиц
 * arcface - загруженная модель нейросети
 * faces - вектор изображений лиц (Mat)
 * Возвращает: вектор эмбеддингов (по одному для каждого лица)
 *
 * Процесс преобразования:
 * 1. Конвертация BGR -> RGB
 * 2. Ресайз до 112x112 пикселей
 * 3. Нормализация значений пикселей
 * 4. Преобразование в blob
 * 5. Пропуск через нейросеть
 */
vector<Mat> getFaceEmbeddings(Net& arcface, const vector<Mat>& faces) {
    vector<Mat> embeddings;
    for (const auto& face : faces) {
        // Конвертация цветового пространства BGR->RGB
        Mat rgb;
        cvtColor(face, rgb, COLOR_BGR2RGB);

        // Изменение размера до требуемого моделью
        resize(rgb, rgb, Size(112, 112));

        // Подготовка входных данных для нейросети:
        // - Масштабирование пикселей 1/128
        // - Вычитание среднего значения 127.5
        // - Инвертирование каналов (true)
        Mat blob = blobFromImage(rgb, 1.0 / 128.0, Size(112, 112),
            Scalar(127.5, 127.5, 127.5), true, false);

        // Подача blob в нейросеть и получение результата
        arcface.setInput(blob);
        embeddings.push_back(arcface.forward().clone());
    }
    return embeddings;
}

/*
 * Загрузка базы известных лиц из указанной директории
 * facesPath - путь к папке с изображениями
 * arcface - модель для получения эмбеддингов
 * knownEncodings - выходной вектор эмбеддингов
 * knownNames - выходной вектор имен (берутся из имен файлов)
 *
 * Структура папок должна быть:
 * faces/
 *   ├── person1.jpg
 *   ├── person2.jpg
 *   └── ...
 */
void loadKnownFaces(const string& facesPath, Net& arcface,
    vector<Mat>& knownEncodings, vector<string>& knownNames) {
    for (const auto& entry : fs::directory_iterator(facesPath)) {
        if (entry.is_regular_file()) {
            // Загрузка изображения
            Mat img = imread(entry.path().string());
            if (!img.empty()) {
                // Для простоты считаем, что все изображение - это лицо
                Rect faceRect(0, 0, img.cols, img.rows);
                Mat face = img(faceRect).clone();

                // Получение векторного представления лица
                vector<Mat> embeddings = getFaceEmbeddings(arcface, { face });

                if (!embeddings.empty()) {
                    // Сохранение эмбеддинга и имени (без расширения файла)
                    knownEncodings.push_back(embeddings[0]);
                    knownNames.push_back(entry.path().stem().string());
                }
            }
        }
    }
}

/*
 * Вычисление косинусного сходства между двумя векторами
 * a, b - входные векторы (эмбеддинги)
 * Возвращает: значение сходства от -1 до 1
 *
 * Для нормализованных векторов (как в ArcFace):
 * 1.0 - идентичные векторы
 * 0.0 - ортогональные векторы
 * -1.0 - противоположные векторы
 */
double cosineSimilarity(const Mat& a, const Mat& b) {
    return a.dot(b) / (norm(a) * norm(b));
}

int main() {
    // ========== ИНИЦИАЛИЗАЦИЯ ==========
    // Пути к файлам модели и базы лиц
    string arcfacePath = "arcface.onnx";
    string facesPath = "faces";

    // Загрузка модели ArcFace
    Net arcface = loadArcFaceModel(arcfacePath);

    // Загрузка базы известных лиц
    vector<Mat> knownEncodings;
    vector<string> knownNames;
    loadKnownFaces(facesPath, arcface, knownEncodings, knownNames);

    if (knownEncodings.empty()) {
        cerr << "No known faces loaded! Check faces directory." << endl;
        return -1;
    }

    // Инициализация видеопотока с камеры (0 - индекс камеры)
    VideoCapture video(0);
    if (!video.isOpened()) {
        cerr << "Error opening camera! Check camera connection." << endl;
        return -1;
    }

    // Коэффициент уменьшения кадра для повышения производительности
    float scale = 2.0;

    // ========== ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ==========
    Mat frame;
    while (video.read(frame)) {
        // Уменьшение размера кадра для ускорения обработки
        Mat resizedFrame;
        resize(frame, resizedFrame, Size(frame.cols / scale, frame.rows / scale));

        // ===== ДЕТЕКЦИЯ ЛИЦ =====
        CascadeClassifier faceCascade;
        if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
            cerr << "Error loading Haar Cascade classifier!" << endl;
            return -1;
        }

        // Конвертация в градации серого (требуется для Haar Cascade)
        Mat gray;
        cvtColor(resizedFrame, gray, COLOR_BGR2GRAY);

        // Параметры детекции:
        // 1.1 - коэффициент масштабирования
        // 4 - минимальное количество соседей
        vector<Rect> faceLocations;
        faceCascade.detectMultiScale(gray, faceLocations, 1.1, 4);

        // ===== ОБРАБОТКА КАЖДОГО ОБНАРУЖЕННОГО ЛИЦА =====
        for (const auto& faceLoc : faceLocations) {
            // Выделение области лица
            Mat face = resizedFrame(faceLoc).clone();

            // Получение эмбеддинга лица
            vector<Mat> embeddings = getFaceEmbeddings(arcface, { face });

            if (!embeddings.empty()) {
                Mat unknownEncoding = embeddings[0];
                string name = "Unknown";
                double bestSimilarity = 0.4; // Порог распознавания

                // Поиск наиболее похожего лица в базе
                for (size_t i = 0; i < knownEncodings.size(); ++i) {
                    double similarity = cosineSimilarity(unknownEncoding, knownEncodings[i]);
                    if (similarity > bestSimilarity) {
                        bestSimilarity = similarity;
                        name = knownNames[i];
                    }
                }

                // ===== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ =====
                // Масштабирование координат обратно к исходному размеру кадра
                Rect scaledFaceLoc(
                    faceLoc.x * scale,
                    faceLoc.y * scale,
                    faceLoc.width * scale,
                    faceLoc.height * scale
                );

                // Отрисовка прямоугольника вокруг лица
                rectangle(frame, scaledFaceLoc, Scalar(0, 0, 255), 2);

                // Подпись с именем под прямоугольником
                putText(frame, name,
                    Point(scaledFaceLoc.x, scaledFaceLoc.y + scaledFaceLoc.height + 20),
                    FONT_HERSHEY_DUPLEX, 0.8, Scalar(255, 255, 255), 1);
            }
        }

        // Отображение количества обнаруженных лиц
        string facesCount = "Faces: " + to_string(faceLocations.size());
        putText(frame, facesCount, Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        // Показ обработанного кадра
        imshow("Face Recognition", frame);

        // Выход по нажатию 'q'
        if (waitKey(1) == 'q') break;
    }

    return 0;
}