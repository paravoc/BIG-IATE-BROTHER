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
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
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
const string ADMIN_PASSWORD = "123qwe"; // Пароль для административных действий

// ---------------------------
// Состояния интерфейса
// ---------------------------
enum AppState {
    MAIN_SCREEN,
    SCANNING,
    ATTENDANCE_RESULT,
    ARRIVAL_STATUS_MESSAGE,
    DELETE_EMPLOYEE
};

AppState current_state = MAIN_SCREEN;
string current_person_name = "";
string attendance_status = "";
string next_departure_time = "";
string check_in_time = "";
string arrival_status_message = "";
string scheduled_start_time = "";
chrono::steady_clock::time_point message_start_time;

// Переменные для добавления сотрудника
bool adding_new_person = false;
bool embedding_extracted = false; // Флаг что эмбеддинг уже извлечен
string new_person_name = "";
string start_time = "09:00";
string end_time = "18:00";
vector<float> new_person_embedding;
Mat captured_frame;

// Переменные для редактирования времени
enum EditField {
    NONE,
    START_TIME,
    END_TIME
};
EditField current_edit_field = NONE;
string time_edit_buffer = "";

// Переменные для удаления сотрудников
bool deleting_employee = false;
vector<string> all_employees;
string search_query = "";
int scroll_offset = 0;
const int EMPLOYEES_PER_PAGE = 8;
int selected_employee_index = 0;

// Переменные для ввода пароля
bool entering_password = false;
string password_input = "";
bool password_for_add = false; // true - для добавления, false - для удаления
string password_employee_name = ""; // Имя сотрудника для удаления

// Глобальные указатели
pqxx::connection* global_conn = nullptr;
Net* global_net = nullptr;
Net* global_faceDetector = nullptr;

// ---------------------------
// Вспомогательная функция для преобразования в нижний регистр
// ---------------------------
string toLower(const string& str) {
    string result = str;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

// ---------------------------
// Функция для отображения окна ввода пароля
// ---------------------------
void show_password_window() {
    Mat password_gui = Mat::zeros(Size(400, 200), CV_8UC3);
    password_gui = Scalar(30, 30, 60);

    string title = password_for_add ? "ADD EMPLOYEE - PASSWORD" : "DELETE EMPLOYEE - PASSWORD";
    putText(password_gui, title, Point(80, 40),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 200, 255), 2);

    if (!password_for_add && !password_employee_name.empty()) {
        putText(password_gui, "Delete: " + password_employee_name, Point(100, 70),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }

    putText(password_gui, "Enter password:", Point(50, 100),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    // Поле ввода пароля
    Rect password_rect(50, 110, 300, 30);
    rectangle(password_gui, password_rect, Scalar(255, 255, 255), 2);

    // Отображаем звездочки вместо пароля
    string display_password = string(password_input.length(), '*');
    putText(password_gui, display_password, Point(60, 130),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    // Кнопки
    Rect ok_button(100, 150, 80, 30);
    rectangle(password_gui, ok_button, Scalar(0, 150, 0), -1);
    rectangle(password_gui, ok_button, Scalar(255, 255, 255), 2);
    putText(password_gui, "OK", Point(120, 170),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    Rect cancel_button(220, 150, 80, 30);
    rectangle(password_gui, cancel_button, Scalar(150, 0, 0), -1);
    rectangle(password_gui, cancel_button, Scalar(255, 255, 255), 2);
    putText(password_gui, "CANCEL", Point(225, 170),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

    imshow("Password Required", password_gui);
}

// ---------------------------
// Функция обработки кликов мыши
// ---------------------------
void handle_mouse_click(int event, int x, int y, int flags, void* userdata) {
    if (event != EVENT_LBUTTONDOWN) return;

    // ОКНО ВВОДА ПАРОЛЯ
    if (entering_password) {
        // Кнопка OK
        if (x >= 100 && x <= 180 && y >= 150 && y <= 180) {
            if (password_input == ADMIN_PASSWORD) {
                entering_password = false;
                destroyWindow("Password Required");

                if (password_for_add) {
                    // Пароль верный - начинаем добавление сотрудника
                    adding_new_person = true;
                    embedding_extracted = false; // Сбрасываем флаг извлечения
                    new_person_name = "";
                    start_time = "09:00";
                    end_time = "18:00";
                    new_person_embedding.clear();
                    current_edit_field = NONE;
                    cout << "Password correct. Starting employee addition..." << endl;
                }
                else {
                    // Пароль верный - удаляем сотрудника
                    if (!password_employee_name.empty()) {
                        try {
                            delete_face_from_db(*global_conn, password_employee_name);
                            cout << "Employee " << password_employee_name << " deleted successfully!" << endl;
                            all_employees = get_all_employees(*global_conn);
                            selected_employee_index = 0;
                            scroll_offset = 0;
                        }
                        catch (const exception& e) {
                            cout << "Error deleting employee: " << e.what() << endl;
                        }
                    }
                    password_employee_name = "";
                }
            }
            else {
                // Неверный пароль
                cout << "Incorrect password!" << endl;
                password_input = "";
            }
        }
        // Кнопка CANCEL
        else if (x >= 220 && x <= 300 && y >= 150 && y <= 180) {
            entering_password = false;
            password_input = "";
            password_employee_name = "";
            destroyWindow("Password Required");
            cout << "Password input cancelled" << endl;
        }
        return;
    }

    // ГЛАВНЫЙ ЭКРАН
    if (current_state == MAIN_SCREEN && !adding_new_person && !deleting_employee) {
        if (x >= 400 && x <= 700 && y >= 150 && y <= 210) {
            current_state = SCANNING;
            cout << "Starting face scan..." << endl;
        }
    }

    // ЭКРАН СКАНИРОВАНИЯ
    else if (current_state == SCANNING) {
        if (x >= 350 && x <= 450 && y >= 520 && y <= 560) {
            current_state = MAIN_SCREEN;
            cout << "Scan cancelled" << endl;
        }
    }

    // ЭКРАН РЕЗУЛЬТАТА
    else if (current_state == ATTENDANCE_RESULT) {
        if (x >= 350 && x <= 450 && y >= 480 && y <= 520) {
            current_state = MAIN_SCREEN;
            cout << "Returning to main screen" << endl;
        }
    }

    // ОКНО ДОБАВЛЕНИЯ СОТРУДНИКА
    else if (adding_new_person) {
        // Кнопка SAVE
        if (x >= 150 && x <= 270 && y >= 250 && y <= 290) {
            if (!new_person_name.empty() && !new_person_embedding.empty()) {
                try {
                    add_person_with_schedule(*global_conn, new_person_name, new_person_embedding, start_time, end_time);
                    adding_new_person = false;
                    embedding_extracted = false;
                    new_person_embedding.clear();
                    captured_frame.release();
                    current_edit_field = NONE;
                    destroyWindow("Add Employee");
                    cout << "Employee " << new_person_name << " added successfully!" << endl;
                }
                catch (const exception& e) {
                    cout << "Error adding employee: " << e.what() << endl;
                }
            }
        }
        // Кнопка CANCEL
        else if (x >= 330 && x <= 450 && y >= 250 && y <= 290) {
            adding_new_person = false;
            embedding_extracted = false;
            new_person_embedding.clear();
            captured_frame.release();
            current_edit_field = NONE;
            destroyWindow("Add Employee");
            cout << "Add employee cancelled" << endl;
        }
        // Поле Start Time
        else if (x >= 200 && x <= 300 && y >= 140 && y <= 170) {
            current_edit_field = START_TIME;
            time_edit_buffer = start_time;
            cout << "Editing start time" << endl;
        }
        // Поле End Time
        else if (x >= 200 && x <= 300 && y >= 180 && y <= 210) {
            current_edit_field = END_TIME;
            time_edit_buffer = end_time;
            cout << "Editing end time" << endl;
        }
    }

    // ОКНО УДАЛЕНИЯ СОТРУДНИКОВ
    else if (deleting_employee) {
        vector<string> filtered_employees;
        for (const auto& employee : all_employees) {
            if (search_query.empty() ||
                employee.find(search_query) != string::npos ||
                toLower(employee).find(toLower(search_query)) != string::npos) {
                filtered_employees.push_back(employee);
            }
        }

        int y_offset = 120;
        bool employee_clicked = false;

        for (int i = scroll_offset; i < filtered_employees.size() && (i - scroll_offset) < EMPLOYEES_PER_PAGE; i++) {
            // Кнопка DELETE для сотрудника
            if (x >= 450 && x <= 550 && y >= (y_offset - 20) && y <= (y_offset + 5)) {
                string employee = filtered_employees[i];

                // Запрашиваем пароль для удаления
                entering_password = true;
                password_for_add = false;
                password_employee_name = employee;
                password_input = "";
                cout << "Password required to delete employee: " << employee << endl;

                employee_clicked = true;
                break;
            }

            // Клик по имени сотрудника - выделяем его
            if (x >= 45 && x <= 445 && y >= (y_offset - 25) && y <= (y_offset + 10)) {
                selected_employee_index = i;
                employee_clicked = true;
                break;
            }

            y_offset += 40;
        }

        // Кнопка UP (прокрутка вверх)
        if (!employee_clicked && scroll_offset > 0 &&
            x >= 250 && x <= 350 && y >= 450 && y <= 480) {
            scroll_offset = max(0, scroll_offset - EMPLOYEES_PER_PAGE);
            selected_employee_index = max(0, selected_employee_index - EMPLOYEES_PER_PAGE);
        }

        // Кнопка DOWN (прокрутка вниз)
        if (!employee_clicked && (scroll_offset + EMPLOYEES_PER_PAGE) < filtered_employees.size() &&
            x >= 360 && x <= 460 && y >= 450 && y <= 480) {
            scroll_offset = min((int)filtered_employees.size() - EMPLOYEES_PER_PAGE,
                scroll_offset + EMPLOYEES_PER_PAGE);
            selected_employee_index = min((int)filtered_employees.size() - 1,
                selected_employee_index + EMPLOYEES_PER_PAGE);
        }

        // Кнопка CANCEL
        if (!employee_clicked && x >= 250 && x <= 350 && y >= 400 && y <= 430) {
            deleting_employee = false;
            search_query = "";
            scroll_offset = 0;
            selected_employee_index = 0;
            destroyWindow("Delete Employee");
            cout << "Delete employee cancelled" << endl;
        }
    }

    // ЭКРАН СТАТУСА ПРИБЫТИЯ
    else if (current_state == ARRIVAL_STATUS_MESSAGE) {
        current_state = MAIN_SCREEN;
        cout << "Status screen closed" << endl;
    }
}

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
// Функция для получения текущего времени в формате строки
// ---------------------------
string get_current_time_string() {
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm local_tm;
    localtime_s(&local_tm, &now_time);

    stringstream ss;
    ss << put_time(&local_tm, "%H:%M:%S");
    return ss.str();
}

// ---------------------------
// Функция для проверки опоздания
// ---------------------------
string check_arrival_status(const string& check_time_str, const string& scheduled_time) {
    try {
        // Парсим время отметки
        int check_hour = stoi(check_time_str.substr(0, 2));
        int check_minute = stoi(check_time_str.substr(3, 2));

        // Парсим время начала работы
        int scheduled_hour = stoi(scheduled_time.substr(0, 2));
        int scheduled_minute = stoi(scheduled_time.substr(3, 2));

        // Сравниваем время
        if (check_hour < scheduled_hour ||
            (check_hour == scheduled_hour && check_minute <= scheduled_minute)) {
            return "On Time";
        }
        else {
            return "Late";
        }
    }
    catch (...) {
        return "Unknown";
    }
}
// ---------------------------
// Функция для отметки посещения с проверкой опоздания
// ---------------------------
void mark_attendance_with_schedule(pqxx::connection& conn, const string& name, const string& check_type) {
    pqxx::work txn(conn);

    auto result = txn.exec_params("SELECT id FROM face_embeddings WHERE name = $1", name);
    if (!result.empty()) {
        int employee_id = result[0]["id"].as<int>();

        // Check if there was already a check today
        auto today_check = txn.exec_params(
            "SELECT check_type, check_time FROM attendance WHERE employee_id = $1 AND DATE(check_time) = CURRENT_DATE ORDER BY check_time DESC LIMIT 1",
            employee_id
        );

        string current_check_type = check_type;
        string action_message = "";

        // If there was already an "in" check today, the next will be "out"
        if (!today_check.empty()) {
            string last_check_type = today_check[0]["check_type"].as<string>();
            if (last_check_type == "in") {
                current_check_type = "out";
                action_message = "checked OUT";
            }
            else {
                current_check_type = "in";
                action_message = "checked IN";
            }
        }
        else {
            // First check today - check in
            current_check_type = "in";
            action_message = "checked IN";
        }

        // Get employee schedule
        auto schedule_result = txn.exec_params(
            "SELECT start_time, end_time FROM work_schedule WHERE employee_id = $1 AND work_date = CURRENT_DATE",
            employee_id
        );

        scheduled_start_time = "09:00";
        string scheduled_end_time = "18:00";

        if (!schedule_result.empty()) {
            scheduled_start_time = schedule_result[0]["start_time"].as<string>();
            scheduled_end_time = schedule_result[0]["end_time"].as<string>();
        }

        check_in_time = get_current_time_string();

        // Check lateness only for check-in
        string status = "Checked";
        if (current_check_type == "in") {
            status = check_arrival_status(check_in_time.substr(0, 5), scheduled_start_time);
        }

        // Form message
        if (current_check_type == "in") {
            if (status == "On Time") {
                arrival_status_message = name + " " + action_message + " on time!\\nScheduled: " + scheduled_start_time + "\\nActual: " + check_in_time.substr(0, 5);
                attendance_status = "Successfully checked IN!";
            }
            else {
                arrival_status_message = name + " " + action_message + " LATE!\\nScheduled: " + scheduled_start_time + "\\nActual: " + check_in_time.substr(0, 5);
                attendance_status = "Successfully checked IN (LATE)!";
            }
        }
        else {
            arrival_status_message = name + " " + action_message + "\\nTime: " + check_in_time.substr(0, 5);
            attendance_status = "Successfully checked OUT!";
        }

        // Insert attendance record
        txn.exec_params(
            "INSERT INTO attendance (employee_id, check_type) VALUES ($1, $2)",
            employee_id,
            current_check_type
        );

        txn.commit();

        // Start timer for auto-close
        message_start_time = chrono::steady_clock::now();
        current_state = ATTENDANCE_RESULT;
    }
}// ---------------------------
// Функция для валидации времени
// ---------------------------
bool validate_time_format(const string& time_str) {
    if (time_str.length() != 5) return false;
    if (time_str[2] != ':') return false;

    try {
        int hour = stoi(time_str.substr(0, 2));
        int minute = stoi(time_str.substr(3, 2));
        return (hour >= 0 && hour <= 23 && minute >= 0 && minute <= 59);
    }
    catch (...) {
        return false;
    }
}

// ---------------------------
// Функция создания главного экрана
// ---------------------------
Mat create_main_screen(Mat& camera_frame) {
    Mat gui = Mat::zeros(GUI_SIZE, CV_8UC3);
    gui = Scalar(50, 50, 50);

    putText(gui, "ACCESS CONTROL SYSTEM", Point(150, 50),
        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 200, 255), 2);

    Rect video_rect(50, 80, 320, 240);
    rectangle(gui, video_rect, Scalar(255, 255, 255), 2);

    if (!camera_frame.empty()) {
        Mat resized_frame;
        resize(camera_frame, resized_frame, Size(320, 240));
        resized_frame.copyTo(gui(video_rect));
    }

    putText(gui, "CAMERA VIEW", Point(120, 340),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);

    Rect scan_button(400, 150, 300, 60);
    rectangle(gui, scan_button, Scalar(0, 150, 0), -1);
    rectangle(gui, scan_button, Scalar(255, 255, 255), 2);
    putText(gui, "CHECK IN", Point(470, 190),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

    putText(gui, "Instructions:", Point(400, 350),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 1);
    putText(gui, "- Press 'CHECK IN' for face scan", Point(400, 380),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
    putText(gui, "- Press 'a' to add new employee", Point(400, 400),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
    putText(gui, "- Press 'd' to delete employee", Point(400, 420),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
    putText(gui, "- Press ESC to exit", Point(400, 440),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

    return gui;
}

// ---------------------------
// Функция создания экрана сканирования
// ---------------------------
Mat create_scanning_screen(Mat& camera_frame) {
    Mat gui = Mat::zeros(GUI_SIZE, CV_8UC3);
    gui = Scalar(30, 30, 60);

    putText(gui, "FACE SCANNING", Point(280, 50),
        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 255, 255), 2);

    if (!camera_frame.empty()) {
        Mat resized_frame;
        resize(camera_frame, resized_frame, Size(400, 300));
        Rect video_rect(200, 100, 400, 300);
        rectangle(gui, video_rect, Scalar(255, 255, 255), 3);
        resized_frame.copyTo(gui(video_rect));
    }

    static int anim_counter = 0;
    anim_counter++;
    string scanning_text = "Scanning" + string(anim_counter % 4, '.');
    putText(gui, scanning_text, Point(340, 450),
        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    putText(gui, "Look straight at the camera", Point(280, 480),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 255), 1);

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

    // Choose background color based on status
    if (attendance_status.find("Success") != string::npos) {
        if (arrival_status_message.find("LATE") != string::npos || attendance_status.find("LATE") != string::npos) {
            gui = Scalar(30, 30, 60); // Blue background for late
        }
        else if (arrival_status_message.find("OUT") != string::npos || attendance_status.find("OUT") != string::npos) {
            gui = Scalar(60, 30, 60); // Purple background for checkout
        }
        else {
            gui = Scalar(30, 60, 30); // Green background for on time
        }
    }
    else {
        gui = Scalar(60, 30, 30); // Red background for errors
    }

    putText(gui, "ATTENDANCE RESULT", Point(250, 80),
        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 2);

    if (attendance_status.find("Success") != string::npos) {
        if (arrival_status_message.find("LATE") != string::npos || attendance_status.find("LATE") != string::npos) {
            // Icon for late
            circle(gui, Point(400, 200), 60, Scalar(0, 100, 255), -1);
            putText(gui, "!", Point(390, 230), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 255, 255), 3);
            putText(gui, "LATE", Point(340, 300),
                FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 100, 255), 2);
        }
        else if (arrival_status_message.find("OUT") != string::npos || attendance_status.find("OUT") != string::npos) {
            // Icon for checkout
            circle(gui, Point(400, 200), 60, Scalar(200, 100, 255), -1);
            putText(gui, "→", Point(380, 230), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 255, 255), 3);
            putText(gui, "CHECKED OUT", Point(300, 300),
                FONT_HERSHEY_SIMPLEX, 1.2, Scalar(200, 100, 255), 2);
        }
        else {
            // Icon for on time
            circle(gui, Point(400, 200), 60, Scalar(0, 255, 0), -1);
            putText(gui, "✓", Point(380, 230), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 255, 255), 3);
            putText(gui, "ON TIME", Point(330, 300),
                FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 255, 0), 2);
        }
    }
    else {
        circle(gui, Point(400, 200), 60, Scalar(0, 0, 255), -1);
        putText(gui, "✗", Point(380, 230), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 255, 255), 3);
        putText(gui, "ERROR!", Point(340, 300),
            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
    }

    putText(gui, "Employee: " + current_person_name, Point(300, 350),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);

    // Display status
    if (attendance_status.find("Success") != string::npos) {
        if (arrival_status_message.find("LATE") != string::npos) {
            putText(gui, "Status: LATE", Point(300, 380),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 100, 255), 1);
        }
        else if (arrival_status_message.find("OUT") != string::npos) {
            putText(gui, "Status: CHECKED OUT", Point(300, 380),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 100, 255), 1);
        }
        else {
            putText(gui, "Status: ON TIME", Point(300, 380),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 1);
        }

        // Time of check
        putText(gui, "Time: " + check_in_time.substr(0, 5), Point(300, 410),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 255, 200), 1);

        // Additional information
        if (arrival_status_message.find("Scheduled:") != string::npos) {
            putText(gui, "Scheduled: " + scheduled_start_time, Point(300, 440),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 255), 1);
        }
    }
    else {
        putText(gui, "Status: " + attendance_status, Point(300, 380),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);
    }

    // Countdown timer
    auto current_time = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::seconds>(current_time - message_start_time);
    int remaining = 3 - elapsed.count();

    if (remaining > 0) {
        putText(gui, "Closing in " + to_string(remaining) + " sec...", Point(300, 480),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);
    }

    return gui;
}// ---------------------------
// Функция создания экрана статуса прибытия
// ---------------------------
Mat create_arrival_status_screen() {
    Mat gui = Mat::zeros(GUI_SIZE, CV_8UC3);

    if (arrival_status_message.find("ON TIME") != string::npos) {
        gui = Scalar(30, 60, 30);
    }
    else {
        gui = Scalar(30, 30, 60);
    }

    putText(gui, "ARRIVAL STATUS", Point(280, 80),
        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 2);

    if (arrival_status_message.find("ON TIME") != string::npos) {
        circle(gui, Point(400, 200), 60, Scalar(0, 255, 0), -1);
        putText(gui, "✓", Point(380, 230), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 255, 255), 3);
        putText(gui, "ON TIME!", Point(330, 300),
            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
    }
    else {
        circle(gui, Point(400, 200), 60, Scalar(0, 100, 255), -1);
        putText(gui, "!", Point(390, 230), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 255, 255), 3);
        putText(gui, "LATE!", Point(350, 300),
            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 100, 255), 2);
    }

    vector<string> lines;
    size_t pos = 0;
    string message = arrival_status_message;
    while ((pos = message.find("\\n")) != string::npos) {
        lines.push_back(message.substr(0, pos));
        message.erase(0, pos + 2);
    }
    lines.push_back(message);

    int y_offset = 350;
    for (const auto& line : lines) {
        putText(gui, line, Point(150, y_offset),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);
        y_offset += 30;
    }

    auto current_time = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::seconds>(current_time - message_start_time);
    int remaining = 3 - elapsed.count();

    if (remaining > 0) {
        putText(gui, "Closing in " + to_string(remaining) + " seconds...", Point(300, 450),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);
    }

    return gui;
}

// ---------------------------
// Функция для извлечения эмбеддинга из захваченного кадра
// ---------------------------
bool extract_face_embedding() {
    if (!global_conn || !global_net || !global_faceDetector || captured_frame.empty()) {
        return false;
    }

    Mat resizedFrame;
    resize(captured_frame, resizedFrame, PROCESSING_SIZE);

    Mat blob = blobFromImage(resizedFrame, 1.0, Size(300, 300),
        Scalar(104, 177, 123), false, false);
    global_faceDetector->setInput(blob);
    Mat detections = global_faceDetector->forward();

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
                new_person_embedding = get_face_embedding(*global_net, face);
                embedding_extracted = true;
                return true;
            }
        }
    }
    return false;
}

// ---------------------------
// Функция для окна добавления сотрудника
// ---------------------------
void show_add_person_window() {
    if (!global_conn || !global_net || !global_faceDetector) return;

    Mat add_gui = Mat::zeros(Size(600, 400), CV_8UC3);
    add_gui = Scalar(60, 30, 60);

    if (captured_frame.empty()) {
        putText(add_gui, "ERROR: No captured frame!", Point(150, 80),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        imshow("Add Employee", add_gui);
        return;
    }

    // Извлекаем эмбеддинг только один раз
    if (!embedding_extracted) {
        bool face_found = extract_face_embedding();
        if (!face_found) {
            putText(add_gui, "Face not found in captured frame!", Point(100, 80),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        }
    }

    putText(add_gui, "ADD NEW EMPLOYEE", Point(150, 40),
        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 200, 255), 2);

    // Статус обнаружения лица
    if (embedding_extracted && !new_person_embedding.empty()) {
        putText(add_gui, "Face detected: READY", Point(50, 80),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
    }
    else {
        putText(add_gui, "Face detected: NOT FOUND", Point(50, 80),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
    }

    putText(add_gui, "Full Name:", Point(50, 120),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    rectangle(add_gui, Rect(200, 100, 300, 30), Scalar(255, 255, 255), 1);
    putText(add_gui, new_person_name, Point(210, 120),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    putText(add_gui, "Start Time:", Point(50, 160),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    Rect start_time_rect(200, 140, 100, 30);
    Scalar start_time_color = (current_edit_field == START_TIME) ? Scalar(0, 255, 255) : Scalar(255, 255, 255);
    rectangle(add_gui, start_time_rect, start_time_color, 2);

    string start_display_text = (current_edit_field == START_TIME && !time_edit_buffer.empty()) ? time_edit_buffer : start_time;
    putText(add_gui, start_display_text, Point(210, 160),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    putText(add_gui, "End Time:", Point(50, 200),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    Rect end_time_rect(200, 180, 100, 30);
    Scalar end_time_color = (current_edit_field == END_TIME) ? Scalar(0, 255, 255) : Scalar(255, 255, 255);
    rectangle(add_gui, end_time_rect, end_time_color, 2);

    string end_display_text = (current_edit_field == END_TIME && !time_edit_buffer.empty()) ? time_edit_buffer : end_time;
    putText(add_gui, end_display_text, Point(210, 200),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

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

    putText(add_gui, "Use TAB to switch fields, ENTER to save, ESC to cancel", Point(100, 350),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

    imshow("Add Employee", add_gui);
}

// ---------------------------
// Функция для окна удаления сотрудников
// ---------------------------
// ---------------------------
// Function for employee deletion window
// ---------------------------
void show_delete_employee_window() {
    if (!global_conn) return;

    Mat delete_gui = Mat::zeros(Size(600, 500), CV_8UC3);
    delete_gui = Scalar(60, 30, 30);

    putText(delete_gui, "DELETE EMPLOYEE", Point(200, 40),
        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 100, 100), 2);

    putText(delete_gui, "Search:", Point(50, 80),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    rectangle(delete_gui, Rect(120, 60, 300, 30), Scalar(255, 255, 255), 1);
    putText(delete_gui, search_query, Point(130, 80),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    vector<string> filtered_employees;
    vector<bool> employee_status; // true = at work, false = not at work

    for (const auto& employee : all_employees) {
        if (search_query.empty() ||
            employee.find(search_query) != string::npos ||
            toLower(employee).find(toLower(search_query)) != string::npos) {
            filtered_employees.push_back(employee);

            // Check if employee is currently at work
            bool is_at_work = false;
            try {
                pqxx::work txn(*global_conn);
                auto result = txn.exec_params(
                    "SELECT check_type FROM attendance WHERE employee_id = (SELECT id FROM face_embeddings WHERE name = $1) AND DATE(check_time) = CURRENT_DATE ORDER BY check_time DESC LIMIT 1",
                    employee
                );

                if (!result.empty()) {
                    string last_check_type = result[0]["check_type"].as<string>();
                    // If last check was "in", employee is at work
                    is_at_work = (last_check_type == "in");
                }
                txn.commit();
            }
            catch (const exception& e) {
                cout << "Error checking employee status: " << e.what() << endl;
            }

            employee_status.push_back(is_at_work);
        }
    }

    int y_offset = 120;
    int count = 0;

    for (int i = scroll_offset; i < filtered_employees.size() && count < EMPLOYEES_PER_PAGE; i++) {
        string employee = filtered_employees[i];
        bool is_at_work = employee_status[i];

        // Different colors based on selection and work status
        Scalar name_color;
        Scalar bg_color;
        Scalar status_color;

        if (i == selected_employee_index) {
            // Selected employee - highlighted background
            bg_color = Scalar(80, 80, 80);
            if (is_at_work) {
                name_color = Scalar(100, 255, 100); // Bright green for selected + at work
                status_color = Scalar(50, 200, 50); // Darker green for status
            }
            else {
                name_color = Scalar(255, 150, 100); // Orange for selected + not at work
                status_color = Scalar(200, 100, 50); // Darker orange for status
            }
        }
        else {
            // Not selected
            bg_color = Scalar(60, 30, 30);
            if (is_at_work) {
                name_color = Scalar(150, 255, 150); // Light green for at work
                status_color = Scalar(0, 200, 0);   // Green for status
            }
            else {
                name_color = Scalar(255, 200, 150); // Light orange for not at work
                status_color = Scalar(200, 120, 0); // Orange for status
            }
        }

        // Background for selected employee
        if (i == selected_employee_index) {
            rectangle(delete_gui, Rect(45, y_offset - 25, 510, 35), bg_color, -1);
        }

        // Employee name
        putText(delete_gui, employee, Point(50, y_offset),
            FONT_HERSHEY_SIMPLEX, 0.6, name_color, 1);

        // Status indicator (small circle)
        int status_circle_x = 430;
        int status_circle_y = y_offset - 8;
        Scalar circle_color = is_at_work ? Scalar(0, 255, 0) : Scalar(0, 165, 255); // Green or orange
        circle(delete_gui, Point(status_circle_x, status_circle_y), 4, circle_color, -1);

        // Status text
        string status_text = is_at_work ? "AT WORK" : "NOT AT WORK";
        Scalar text_color = is_at_work ? Scalar(150, 255, 150) : Scalar(255, 200, 150);
        putText(delete_gui, status_text, Point(440, y_offset),
            FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);

        
        y_offset += 40;
        count++;
    }

    // Display summary
    int at_work_count = 0;
    for (bool status : employee_status) {
        if (status) at_work_count++;
    }

    string summary_text = "Total: " + to_string(filtered_employees.size()) +
        "  |  At work: " + to_string(at_work_count) +
        "  |  Absent: " + to_string(filtered_employees.size() - at_work_count);
    putText(delete_gui, summary_text, Point(150, 100),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 255), 1);

    // Navigation buttons
    if (filtered_employees.size() > EMPLOYEES_PER_PAGE) {
        if (scroll_offset > 0) {
            Rect up_btn(250, 450, 100, 30);
            rectangle(delete_gui, up_btn, Scalar(50, 50, 150), -1);
            rectangle(delete_gui, up_btn, Scalar(255, 255, 255), 1);
            putText(delete_gui, "UP", Point(285, 470),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        }

        if (scroll_offset + EMPLOYEES_PER_PAGE < filtered_employees.size()) {
            Rect down_btn(360, 450, 100, 30);
            rectangle(delete_gui, down_btn, Scalar(50, 50, 150), -1);
            rectangle(delete_gui, down_btn, Scalar(255, 255, 255), 1);
            putText(delete_gui, "DOWN", Point(375, 470),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        }
    }

    // Cancel button
    Rect cancel_button(250, 400, 100, 30);
    rectangle(delete_gui, cancel_button, Scalar(150, 0, 0), -1);
    rectangle(delete_gui, cancel_button, Scalar(255, 255, 255), 2);
    putText(delete_gui, "CANCEL", Point(265, 420),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    // Instructions
    putText(delete_gui, "Use ARROWS to navigate, ENTER to delete, ESC to cancel", Point(100, 490),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

    // Legend
    putText(delete_gui, "Legend: ", Point(50, 470),
        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    circle(delete_gui, Point(100, 468), 4, Scalar(0, 255, 0), -1);
    putText(delete_gui, "At work", Point(108, 472),
        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(150, 255, 150), 1);
    circle(delete_gui, Point(170, 468), 4, Scalar(0, 165, 255), -1);
    putText(delete_gui, "Not at work", Point(178, 472),
        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 200, 150), 1);

    imshow("Delete Employee", delete_gui);
}
// ---------------------------
// Обработка клавиш для окна удаления сотрудников
// ---------------------------
void handle_delete_employee_keys(int key) {
    if (!deleting_employee) return;

    vector<string> filtered_employees;
    for (const auto& employee : all_employees) {
        if (search_query.empty() ||
            employee.find(search_query) != string::npos ||
            toLower(employee).find(toLower(search_query)) != string::npos) {
            filtered_employees.push_back(employee);
        }
    }

    if (filtered_employees.empty()) return;

    switch (key) {
    case 82: // Стрелка вверх (Windows)
        selected_employee_index = max(0, selected_employee_index - 1);
        if (selected_employee_index < scroll_offset) {
            scroll_offset = max(0, scroll_offset - 1);
        }
        break;

    case 9: // Стрелка вниз (Windows)
        selected_employee_index = min((int)filtered_employees.size() - 1, selected_employee_index + 1);
        if (selected_employee_index >= scroll_offset + EMPLOYEES_PER_PAGE) {
            scroll_offset = min((int)filtered_employees.size() - EMPLOYEES_PER_PAGE, scroll_offset + 1);
        }
        break;

    case 13: // ENTER - удаление выбранного сотрудника
        if (selected_employee_index >= 0 && selected_employee_index < filtered_employees.size()) {
            string employee = filtered_employees[selected_employee_index];

            // Запрашиваем пароль для удаления
            entering_password = true;
            password_for_add = false;
            password_employee_name = employee;
            password_input = "";
            cout << "Password required to delete employee: " << employee << endl;
        }
        break;
    }
}

// ---------------------------
// Обработка клавиш для окна добавления сотрудника
// ---------------------------
void handle_add_person_keys(int key) {
    if (!adding_new_person) return;

    switch (key) {
    case 9: // TAB - переключение между полями
        if (current_edit_field == NONE) {
            current_edit_field = START_TIME;
            time_edit_buffer = start_time;
        }
        else if (current_edit_field == START_TIME) {
            current_edit_field = END_TIME;
            time_edit_buffer = end_time;
        }
        else {
            current_edit_field = NONE;
            time_edit_buffer = "";
        }
        break;

    case 13: // ENTER
        if (current_edit_field != NONE) {
            if (validate_time_format(time_edit_buffer)) {
                if (current_edit_field == START_TIME) {
                    start_time = time_edit_buffer;
                }
                else if (current_edit_field == END_TIME) {
                    end_time = time_edit_buffer;
                }
            }
            current_edit_field = NONE;
            time_edit_buffer = "";
        }
        else {
            if (!new_person_name.empty() && !new_person_embedding.empty()) {
                try {
                    add_person_with_schedule(*global_conn, new_person_name, new_person_embedding, start_time, end_time);
                    adding_new_person = false;
                    embedding_extracted = false;
                    new_person_embedding.clear();
                    captured_frame.release();
                    current_edit_field = NONE;
                    destroyWindow("Add Employee");
                    cout << "Employee " << new_person_name << " added successfully!" << endl;
                }
                catch (const exception& e) {
                    cout << "Error adding employee: " << e.what() << endl;
                }
            }
        }
        break;
    }
}

// ---------------------------
// Основная функция
// ---------------------------
int main() {
    Net net = readNetFromONNX("res/arcface.onnx");
    Net faceDetector = readNetFromCaffe("res/deploy.prototxt",
        "res/res10_300x300_ssd_iter_140000.caffemodel");

    pqxx::connection conn("dbname=face_recognition user=postgres password=1234 host=localhost");
    global_conn = &conn;
    global_net = &net;
    global_faceDetector = &faceDetector;

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, PROCESSING_SIZE.width);
    cap.set(CAP_PROP_FRAME_HEIGHT, PROCESSING_SIZE.height);

    Mat frame;
    vector<float> current_embedding;

    namedWindow("Face Recognition System", WINDOW_NORMAL);
    resizeWindow("Face Recognition System", GUI_SIZE.width, GUI_SIZE.height);
    setMouseCallback("Face Recognition System", handle_mouse_click);

    cout << "System started. Press ESC to exit." << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (current_state == ARRIVAL_STATUS_MESSAGE) {
            auto current_time = chrono::steady_clock::now();
            auto elapsed = chrono::duration_cast<chrono::seconds>(current_time - message_start_time);
            if (elapsed.count() >= 3) {
                current_state = MAIN_SCREEN;
            }
        }

        // Отображение окна ввода пароля
        if (entering_password) {
            show_password_window();
        }
        // Отображение основного интерфейса
        else if (!adding_new_person && !deleting_employee) {
            Mat gui;
            switch (current_state) {
            case MAIN_SCREEN:
                gui = create_main_screen(frame);
                break;

            case SCANNING:
                gui = create_scanning_screen(frame);

                if (!adding_new_person && !deleting_employee) {
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

                                auto [name, similarity] = check_face_in_db(conn, current_embedding);
                                if (name != "Unknown" && similarity >= 95.0f) {
                                    current_person_name = name;
                                    mark_attendance_with_schedule(conn, name, "in");
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
                break;

            case ATTENDANCE_RESULT:
            {
                gui = create_result_screen();

                // Автоматическое закрытие через 3 секунды
                auto current_time = chrono::steady_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(current_time - message_start_time);
                if (elapsed.count() >= 3) {
                    current_state = MAIN_SCREEN;
                }
            }
            break;

            case ARRIVAL_STATUS_MESSAGE:
            {
                gui = create_arrival_status_screen();

                // Автоматическое закрытие через 3 секунды
                auto current_time = chrono::steady_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(current_time - message_start_time);
                if (elapsed.count() >= 3) {
                    current_state = MAIN_SCREEN;
                }
            }
            break;
            }

            if (!gui.empty()) {
                imshow("Face Recognition System", gui);
            }
        }

        // Отображение окна добавления сотрудника
        if (adding_new_person && !entering_password) {
            show_add_person_window();
        }

        // Отображение окна удаления сотрудников
        if (deleting_employee && !entering_password) {
            show_delete_employee_window();
        }

        int key = waitKey(30);

        // Убрано закрытие по 'q', теперь только по ESC
        if (key == 27) { // ESC
            if (entering_password) {
                entering_password = false;
                password_input = "";
                password_employee_name = "";
                destroyWindow("Password Required");
            }
            else if (adding_new_person) {
                adding_new_person = false;
                embedding_extracted = false;
                new_person_embedding.clear();
                captured_frame.release();
                current_edit_field = NONE;
                destroyWindow("Add Employee");
            }
            else if (deleting_employee) {
                deleting_employee = false;
                search_query = "";
                scroll_offset = 0;
                selected_employee_index = 0;
                destroyWindow("Delete Employee");
            }
            else if (current_state == SCANNING) {
                current_state = MAIN_SCREEN;
            }
            else if (current_state != MAIN_SCREEN) {
                current_state = MAIN_SCREEN;
            }
            else {
                // ESC в главном экране - выход из программы
                cout << "Exiting program..." << endl;
                break;
            }
        }
        else if (key == 'a' && !adding_new_person && !deleting_employee && current_state == MAIN_SCREEN && !entering_password) {
            // Запрашиваем пароль для добавления сотрудника
            entering_password = true;
            password_for_add = true;
            password_input = "";
            cout << "Password required to add new employee" << endl;
        }
        else if (key == 'd' && !adding_new_person && !deleting_employee && current_state == MAIN_SCREEN && !entering_password) {
            // Начало удаления сотрудника
            deleting_employee = true;
            all_employees = get_all_employees(conn);
            search_query = "";
            scroll_offset = 0;
            selected_employee_index = 0;
        }
        else if (key == 'c' && current_state == MAIN_SCREEN && !entering_password) {
            current_state = SCANNING;
            cout << "Starting face scan (keyboard shortcut)..." << endl;
        }
        else if (entering_password) {
            // Обработка ввода пароля
            if (key >= 32 && key <= 126) { // Печатные символы
                if (password_input.length() < 20) {
                    password_input += (char)key;
                }
            }
            else if (key == 8 && !password_input.empty()) { // Backspace
                password_input.pop_back();
            }
            else if (key == 13) { // Enter - подтверждение пароля
                if (password_input == ADMIN_PASSWORD) {
                    entering_password = false;
                    destroyWindow("Password Required");

                    if (password_for_add) {
                        // Пароль верный - начинаем добавление сотрудника
                        frame.copyTo(captured_frame);
                        adding_new_person = true;
                        embedding_extracted = false; // Сбрасываем флаг
                        new_person_name = "";
                        start_time = "09:00";
                        end_time = "18:00";
                        new_person_embedding.clear();
                        current_edit_field = NONE;
                        cout << "Password correct. Starting employee addition..." << endl;
                    }
                    else {
                        // Пароль верный - удаляем сотрудника
                        if (!password_employee_name.empty()) {
                            try {
                                delete_face_from_db(*global_conn, password_employee_name);
                                cout << "Employee " << password_employee_name << " deleted successfully!" << endl;
                                all_employees = get_all_employees(*global_conn);
                                selected_employee_index = 0;
                                scroll_offset = 0;
                            }
                            catch (const exception& e) {
                                cout << "Error deleting employee: " << e.what() << endl;
                            }
                        }
                        password_employee_name = "";
                    }
                }
                else {
                    // Неверный пароль
                    cout << "Incorrect password!" << endl;
                    password_input = "";
                }
            }
        }
        else if (adding_new_person) {
            handle_add_person_keys(key);

            if (current_edit_field != NONE) {
                if (key >= 48 && key <= 57) {
                    if (time_edit_buffer.length() < 5) {
                        time_edit_buffer += (char)key;
                        if (time_edit_buffer.length() == 2 && time_edit_buffer.find(':') == string::npos) {
                            time_edit_buffer += ':';
                        }
                    }
                }
                else if (key == 8 && !time_edit_buffer.empty()) {
                    time_edit_buffer.pop_back();
                    if (time_edit_buffer.length() == 2 && time_edit_buffer[1] == ':') {
                        time_edit_buffer.pop_back();
                    }
                }
            }
            else {
                if (key >= 32 && key <= 126) {
                    if (new_person_name.length() < 50) {
                        new_person_name += (char)key;
                    }
                }
                else if (key == 8 && !new_person_name.empty()) {
                    new_person_name.pop_back();
                }
            }
        }
        else if (deleting_employee) {
            handle_delete_employee_keys(key);

            if (key >= 32 && key <= 126) {
                if (search_query.length() < 30) {
                    search_query += (char)key;
                    selected_employee_index = 0;
                    scroll_offset = 0;
                }
            }
            else if (key == 8 && !search_query.empty()) {
                search_query.pop_back();
                selected_employee_index = 0;
                scroll_offset = 0;
            }
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}