#include <pqxx/pqxx>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Функция для получения эмбединга лица (упрощенная версия)
vector<float> get_face_embedding(Net& net, Mat face) {
    // Предобработка изображения лица
    Mat blob;
    resize(face, face, Size(112, 112));
    cvtColor(face, face, COLOR_BGR2RGB);
    blobFromImage(face, blob, 1.0 / 127.5, Size(112, 112), Scalar(127.5, 127.5, 127.5), true, false);

    // Получение эмбединга
    net.setInput(blob);
    Mat embedding = net.forward();

    // Конвертация в vector<float>
    return vector<float>(embedding.ptr<float>(), embedding.ptr<float>() + embedding.total());
}

int main() {
    // Загрузка модели
    Net net = readNetFromONNX("res/arcface.onnx");

    // Подключение к БД
    pqxx::connection conn("dbname=face_recognition user=postgres password=1234 host=localhost");

    // Вариант 1: Захват с камеры
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        imshow("Camera", frame);

        int key = waitKey(30);
        if (key == 's') { // Нажмите 's' чтобы сохранить текущий кадр
            // Получаем эмбединг
            vector<float> embedding = get_face_embedding(net, frame);

            // Конвертируем в строку для pgvector
            string embedding_str = "[";
            for (size_t i = 0; i < embedding.size(); ++i) {
                if (i != 0) embedding_str += ",";
                embedding_str += to_string(embedding[i]);
            }
            embedding_str += "]";

            // Запрашиваем имя
            cout << "Enter person name: ";
            string name;
            getline(cin, name);

            // Сохраняем в БД
            pqxx::work txn(conn);
            txn.exec_params(
                "INSERT INTO face_embeddings (name, embedding) VALUES ($1, $2)",
                name,
                embedding_str
            );
            txn.commit();

            cout << "Saved to database!" << endl;
        }
        else if (key == 'q') break;
    }

    // Вариант 2: Загрузка из файла
    /*
    Mat img = imread("person.jpg");
    if (img.empty()) {
        cerr << "Cannot open image!" << endl;
        return -1;
    }

    vector<float> embedding = get_face_embedding(net, img);

    // ... аналогично сохраняем в БД ...
    */

    return 0;
}