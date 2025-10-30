#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING  
#define _SILENCE_CXX20_CISO646_REMOVED_WARNING
#pragma warning(disable:4996)

#include <pqxx/pqxx>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include "database.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

// ---------------------------
// Настройки программы
// ---------------------------
const float FACE_CONFIDENCE_THRESHOLD = 0.7f;
const Size PROCESSING_SIZE(640, 480);
const Size GUI_SIZE(800, 600);

// ---------------------------
// Состояния интерфейса
// ---------------------------
enum AppState {
    MAIN_SCREEN,
    SCANNING,
    ATTENDANCE_RESULT
};

AppState current_state = MAIN_SCREEN;
string current_person_name = "";
string attendance_status = "";
string next_departure_time = "";

// Переменные для добавления сотрудника
bool adding_new_person = false;
string new_person_name = "";
string start_time = "09:00";
string end_time = "18:00";
vector<float> new_person_embedding;
Mat captured_frame; // Кадр, снятый при нажатии 'a'

// ---------------------------
// Функция получения эмбеддинга лица
// ---------------------------
vector<float> get_face_embedding(Net& net, Mat face) {
    Mat blob;
    resize(face, face, Size(112, 112));
    cvtColor(face, face, COLOR_BGR2RGB);

    blobFromImage(face, blob, 1.0 / 127.5, Size(112, 112),
        Scalar(127.5, 127.5, 127.5), true, false);

    net.setInput(blob);
    Mat embedding = net.forward();

    vector<float> vec(embedding.ptr<float>(), embedding.ptr<float>() + embedding.total());

    float norm = 0;
    for (float v : vec) norm += v * v;
    norm = sqrt(norm);
    for (float& v : vec) v /= norm;

    return vec;
}

// ---------------------------
// Функция для отметки посещения
// ---------------------------
void mark_attendance(pqxx::connection& conn, const string& name, const string& check_type) {
    pqxx::work txn(conn);

    auto result = txn.exec_params("SELECT id FROM face_embeddings WHERE name = $1", name);
    if (!result.empty()) {
        int employee_id = result[0]["id"].as<int>();

        txn.exec_params(
            "INSERT INTO attendance (employee_id, check_type) VALUES ($1, $2)",
            employee_id,
            check_type
        );

        // Получаем время ухода
        auto schedule_result = txn.exec_params(
            "SELECT end_time FROM work_schedule WHERE employee_id = $1 AND work_date = CURRENT_DATE",
            employee_id
        );

        if (!schedule_result.empty()) {
            next_departure_time = schedule_result[0]["end_time"].as<string>();
        }
        else {
            next_departure_time = "18:00";
        }

        txn.commit();
    }
}

// ---------------------------
// Функция добавления сотрудника с графиком
// ---------------------------
void add_person_with_schedule(pqxx::connection& conn, const string& name,
    const vector<float>& embedding,
    const string& start, const string& end) {
    pqxx::work txn(conn);

    string embedding_str = embedding_to_string(embedding);
    txn.exec_params(
        "INSERT INTO face_embeddings (name, embedding) VALUES ($1, $2::vector)",
        name,
        embedding_str
    );

    auto result = txn.exec_params("SELECT id FROM face_embeddings WHERE name = $1", name);
    if (!result.empty()) {
        int employee_id = result[0]["id"].as<int>();

        txn.exec_params(
            "INSERT INTO work_schedule (employee_id, work_date, start_time, end_time) "
            "VALUES ($1, CURRENT_DATE, $2, $3)",
            employee_id,
            start,
            end
        );
    }

    txn.commit();
}

// ---------------------------
// Функция создания главного экрана
// ---------------------------
Mat create_main_screen(Mat& camera_frame) {
    Mat gui = Mat::zeros(GUI_SIZE, CV_8UC3);
    gui = Scalar(50, 50, 50); // Темно-серый фон

    // Заголовок
    putText(gui, "ACCESS CONTROL SYSTEM", Point(150, 50),
        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 200, 255), 2);

    // Область видео
    Rect video_rect(50, 80, 320, 240);
    rectangle(gui, video_rect, Scalar(255, 255, 255), 2);

    // Вставляем видео
    if (!camera_frame.empty()) {
        Mat resized_frame;
        resize(camera_frame, resized_frame, Size(320, 240));
        resized_frame.copyTo(gui(video_rect));
    }

    putText(gui, "CAMERA VIEW", Point(120, 340),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);

    // Кнопка "Check In"
    Rect scan_button(400, 150, 300, 60);
    rectangle(gui, scan_button, Scalar(0, 150, 0), -1);
    rectangle(gui, scan_button, Scalar(255, 255, 255), 2);
    putText(gui, "CHECK IN", Point(470, 190),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

    // Инструкция
    putText(gui, "Instructions:", Point(400, 350),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 1);
    putText(gui, "- Press 'CHECK IN' for face scan", Point(400, 380),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
    putText(gui, "- Press 'a' to add new employee", Point(400, 400),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
    putText(gui, "- Press 'q' to exit", Point(400, 420),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

    return gui;
}

// ---------------------------
// Функция создания экрана сканирования
// ---------------------------
Mat create_scanning_screen(Mat& camera_frame) {
    Mat gui = Mat::zeros(GUI_SIZE, CV_8UC3);
    gui = Scalar(30, 30, 60); // Синий фон

    putText(gui, "FACE SCANNING", Point(280, 50),
        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 255, 255), 2);

    // Центральная область видео
    if (!camera_frame.empty()) {
        Mat resized_frame;
        resize(camera_frame, resized_frame, Size(400, 300));
        Rect video_rect(200, 100, 400, 300);
        rectangle(gui, video_rect, Scalar(255, 255, 255), 3);
        resized_frame.copyTo(gui(video_rect));
    }

    // Анимированная надпись
    static int anim_counter = 0;
    anim_counter++;
    string scanning_text = "Scanning" + string(anim_counter % 4, '.');
    putText(gui, scanning_text, Point(340, 450),
        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    putText(gui, "Look straight at the camera", Point(280, 480),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 255), 1);

    // Кнопка отмены
    Rect cancel_button(350, 520, 100, 40);
    rectangle(gui, cancel_button, Scalar(50, 50, 150), -1);
    rectangle(gui, cancel_button, Scalar(255, 255, 255), 2);
    putText(gui, "CANCEL", Point(360, 545),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    return gui;
}

// ---------------------------
// Функция создания экрана результата
// ---------------------------
Mat create_result_screen() {
    Mat gui = Mat::zeros(GUI_SIZE, CV_8UC3);
    gui = Scalar(30, 60, 30); // Зеленый фон

    putText(gui, "ATTENDANCE RESULT", Point(250, 80),
        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 2);

    // Иконка статуса
    if (attendance_status.find("Success") != string::npos) {
        circle(gui, Point(400, 200), 60, Scalar(0, 255, 0), -1);
        putText(gui, "✓", Point(380, 230), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 255, 255), 3);
        putText(gui, "SUCCESS!", Point(320, 300),
            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
    }
    else {
        circle(gui, Point(400, 200), 60, Scalar(0, 0, 255), -1);
        putText(gui, "✗", Point(380, 230), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 255, 255), 3);
        putText(gui, "ERROR!", Point(340, 300),
            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
    }

    // Информация
    putText(gui, "Employee: " + current_person_name, Point(300, 350),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);

    if (!next_departure_time.empty() && attendance_status.find("Success") != string::npos) {
        putText(gui, "Departure time: " + next_departure_time, Point(300, 380),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 255, 200), 1);
    }

    putText(gui, "Status: " + attendance_status, Point(300, 410),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);

    // Кнопка продолжения
    Rect continue_button(350, 480, 100, 40);
    rectangle(gui, continue_button, Scalar(0, 100, 0), -1);
    rectangle(gui, continue_button, Scalar(255, 255, 255), 2);
    putText(gui, "OK", Point(375, 505),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    return gui;
}

// ---------------------------
// Функция для окна добавления сотрудника
// ---------------------------
void show_add_person_window(pqxx::connection& conn, Net& net, Net& faceDetector) {
    namedWindow("Add Employee", WINDOW_NORMAL);
    resizeWindow("Add Employee", 600, 400);

    Mat add_gui = Mat::zeros(Size(600, 400), CV_8UC3);
    add_gui = Scalar(60, 30, 60); // Фиолетовый фон

    // Используем сохраненный кадр для извлечения эмбеддинга
    if (captured_frame.empty()) {
        putText(add_gui, "ERROR: No captured frame!", Point(150, 80),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        imshow("Add Employee", add_gui);
        return;
    }

    Mat resizedFrame;
    resize(captured_frame, resizedFrame, PROCESSING_SIZE);

    Mat blob = blobFromImage(resizedFrame, 1.0, Size(300, 300),
        Scalar(104, 177, 123), false, false);
    faceDetector.setInput(blob);
    Mat detections = faceDetector.forward();

    const float* data = detections.ptr<float>();
    const int num_detections = detections.size[2];

    bool face_detected = false;

    for (int i = 0; i < num_detections; ++i) {
        float confidence = data[i * 7 + 2];
        if (confidence > FACE_CONFIDENCE_THRESHOLD) {
            const float* det = data + i * 7;
            int x1 = static_cast<int>(det[3] * resizedFrame.cols);
            int y1 = static_cast<int>(det[4] * resizedFrame.rows);
            int x2 = static_cast<int>(det[5] * resizedFrame.cols);
            int y2 = static_cast<int>(det[6] * resizedFrame.rows);

            Rect faceRect(x1, y1, x2 - x1, y2 - y1);

            if (faceRect.x >= 0 && faceRect.y >= 0 &&
                faceRect.x + faceRect.width <= resizedFrame.cols &&
                faceRect.y + faceRect.height <= resizedFrame.rows) {

                Mat face = resizedFrame(faceRect).clone();
                new_person_embedding = get_face_embedding(net, face);
                face_detected = true;
                break;
            }
        }
    }

    // Интерфейс окна добавления
    putText(add_gui, "ADD NEW EMPLOYEE", Point(150, 40),
        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 200, 255), 2);

    // Статус обнаружения лица
    if (face_detected) {
        putText(add_gui, "Face detected: READY", Point(50, 80),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
    }
    else {
        putText(add_gui, "Face detected: NOT FOUND", Point(50, 80),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
    }

    // Поля ввода
    putText(add_gui, "Full Name:", Point(50, 120),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    rectangle(add_gui, Rect(200, 100, 300, 30), Scalar(255, 255, 255), 1);
    putText(add_gui, new_person_name, Point(210, 120),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    putText(add_gui, "Start Time:", Point(50, 160),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    rectangle(add_gui, Rect(200, 140, 100, 30), Scalar(255, 255, 255), 1);
    putText(add_gui, start_time, Point(210, 160),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    putText(add_gui, "End Time:", Point(50, 200),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    rectangle(add_gui, Rect(200, 180, 100, 30), Scalar(255, 255, 255), 1);
    putText(add_gui, end_time, Point(210, 200),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    // Кнопки
    Rect save_button(150, 250, 120, 40);
    rectangle(add_gui, save_button, Scalar(0, 150, 0), -1);
    rectangle(add_gui, save_button, Scalar(255, 255, 255), 2);
    putText(add_gui, "SAVE", Point(180, 275),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    Rect cancel_button(330, 250, 120, 40);
    rectangle(add_gui, cancel_button, Scalar(150, 0, 0), -1);
    rectangle(add_gui, cancel_button, Scalar(255, 255, 255), 2);
    putText(add_gui, "CANCEL", Point(350, 275),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    // Инструкция
    putText(add_gui, "Press ENTER to save, ESC to cancel", Point(150, 350),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

    imshow("Add Employee", add_gui);
}

// ---------------------------
// Обработка кликов мыши
// ---------------------------
void handle_mouse_click(int x, int y) {
    switch (current_state) {
    case MAIN_SCREEN:
        if (x >= 400 && x <= 700 && y >= 150 && y <= 210) {
            // Кнопка "Check In"
            current_state = SCANNING;
        }
        break;

    case SCANNING:
        if (x >= 350 && x <= 450 && y >= 520 && y <= 560) {
            // Кнопка "Cancel"
            current_state = MAIN_SCREEN;
        }
        break;

    case ATTENDANCE_RESULT:
        if (x >= 350 && x <= 450 && y >= 480 && y <= 520) {
            // Кнопка "OK"
            current_state = MAIN_SCREEN;
        }
        break;
    }
}

// ---------------------------
// Основная функция
// ---------------------------
int main() {
    // Загружаем модели
    Net net = readNetFromONNX("res/arcface.onnx");
    Net faceDetector = readNetFromCaffe("res/deploy.prototxt",
        "res/res10_300x300_ssd_iter_140000.caffemodel");

    // Подключаемся к PostgreSQL
    pqxx::connection conn("dbname=face_recognition user=postgres password=1234 host=localhost");

    // Захват видео с камеры
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, PROCESSING_SIZE.width);
    cap.set(CAP_PROP_FRAME_HEIGHT, PROCESSING_SIZE.height);

    Mat frame;
    vector<float> current_embedding;

    // Создаем основное окно
    namedWindow("Face Recognition System", WINDOW_NORMAL);
    resizeWindow("Face Recognition System", GUI_SIZE.width, GUI_SIZE.height);

    // Обработчик мыши
    setMouseCallback("Face Recognition System", [](int event, int x, int y, int flags, void* userdata) {
        if (event == EVENT_LBUTTONDOWN) {
            handle_mouse_click(x, y);
        }
        }, nullptr);

    // Главный цикл
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Отображение основного интерфейса
        if (!adding_new_person) {
            Mat gui;
            switch (current_state) {
            case MAIN_SCREEN:
                gui = create_main_screen(frame);
                break;
            case SCANNING:
                gui = create_scanning_screen(frame);
                break;
            case ATTENDANCE_RESULT:
                gui = create_result_screen();
                break;
            }
            imshow("Face Recognition System", gui);
        }

        // Обработка распознавания лиц только в режиме сканирования
        if (current_state == SCANNING && !adding_new_person) {
            Mat resizedFrame;
            resize(frame, resizedFrame, PROCESSING_SIZE);

            Mat blob = blobFromImage(resizedFrame, 1.0, Size(300, 300),
                Scalar(104, 177, 123), false, false);
            faceDetector.setInput(blob);
            Mat detections = faceDetector.forward();

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

                    if (faceRect.x >= 0 && faceRect.y >= 0 &&
                        faceRect.x + faceRect.width <= resizedFrame.cols &&
                        faceRect.y + faceRect.height <= resizedFrame.rows) {

                        Mat face = resizedFrame(faceRect).clone();
                        current_embedding = get_face_embedding(net, face);

                        // Автоматическое распознавание в режиме сканирования
                        auto [name, similarity] = check_face_in_db(conn, current_embedding);
                        if (name != "Unknown" && similarity >= 95.0f) {
                            current_person_name = name;
                            mark_attendance(conn, name, "in");
                            attendance_status = "Successfully checked in!";
                            current_state = ATTENDANCE_RESULT;
                        }
                        else if (name == "Unknown") {
                            current_person_name = "Unknown";
                            attendance_status = "Employee not found in database!";
                            current_state = ATTENDANCE_RESULT;
                        }
                        break;
                    }
                }
            }
        }

        // Отображение окна добавления сотрудника
        if (adding_new_person) {
            show_add_person_window(conn, net, faceDetector);
        }

        // Обработка клавиш
        int key = waitKey(30);

        if (key == 'q') break;
        else if (key == 27) { // ESC
            if (adding_new_person) {
                adding_new_person = false;
                new_person_embedding.clear();
                captured_frame.release();
                destroyWindow("Add Employee");
            }
            else if (current_state != MAIN_SCREEN) {
                current_state = MAIN_SCREEN;
            }
        }
        else if (key == 'a' && !adding_new_person) {
            // Начало добавления нового сотрудника - делаем снимок
            frame.copyTo(captured_frame);
            adding_new_person = true;
            new_person_name = "";
            start_time = "09:00";
            end_time = "18:00";
            new_person_embedding.clear();
            cout << "Frame captured for new employee!" << endl;
        }
        else if (key == 13 && adding_new_person) { // ENTER
            // Сохранение нового сотрудника
            if (!new_person_name.empty() && !new_person_embedding.empty()) {
                add_person_with_schedule(conn, new_person_name, new_person_embedding, start_time, end_time);
                adding_new_person = false;
                new_person_embedding.clear();
                captured_frame.release();
                destroyWindow("Add Employee");
                cout << "Employee " << new_person_name << " added successfully!" << endl;
            }
        }
        // Обработка ввода текста для окна добавления
        else if (adding_new_person) {
            if (key >= 32 && key <= 126) { // Печатные символы
                if (new_person_name.length() < 50) {
                    new_person_name += (char)key;
                }
            }
            else if (key == 8 && !new_person_name.empty()) { // Backspace
                new_person_name.pop_back();
            }
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}