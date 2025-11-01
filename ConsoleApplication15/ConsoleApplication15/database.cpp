#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING  
#define _SILENCE_CXX20_CISO646_REMOVED_WARNING
#pragma warning(disable:4996)

#include "database.h"
#include <iostream>
#include <cmath>

using namespace std;

// ---------------------------
// Преобразование вектора эмбеддинга в строку для PostgreSQL
// ---------------------------
string embedding_to_string(const vector<float>& embedding) {
    string embedding_str = "[";
    for (size_t i = 0; i < embedding.size(); ++i) {
        if (i != 0) embedding_str += ",";
        embedding_str += to_string(embedding[i]);
    }
    embedding_str += "]";
    return embedding_str;
}

// ---------------------------
// Проверка лица в базе данных
// ---------------------------
pair<string, float> check_face_in_db(pqxx::connection& conn, const vector<float>& embedding) {
    string embedding_str = embedding_to_string(embedding);
    pqxx::work txn(conn);

    // Запрос: находим двух лучших совпадений по cosine similarity
    auto result = txn.exec_params(
        "SELECT name, 1 - (embedding <#> $1::vector) as similarity "
        "FROM face_embeddings "
        "ORDER BY similarity DESC LIMIT 2",
        embedding_str
    );

    if (result.empty()) return { "Unknown", 0.0f };

    // Получаем лучшего кандидата
    float sim1 = result[0]["similarity"].as<float>() * 100.0f;
    string name1 = result[0]["name"].as<string>();

    // Если есть второй кандидат
    float sim2 = 0;
    string name2 = "";
    if (result.size() > 1) {
        sim2 = result[1]["similarity"].as<float>() * 100.0f;
        name2 = result[1]["name"].as<string>();
    }

    // Если разница между первым и вторым меньше 5%, спрашиваем пользователя
    if (sim1 - sim2 < 25.0f && !name2.empty()) {
        cout << "Multiple similar faces detected: '" << name1 << "' and '" << name2
            << "'. Enter correct name: ";
        string user_input;
        getline(cin, user_input);
        return { user_input.empty() ? "Unknown" : user_input, sim1 };
    }

    return sim1 >= 145.0f ? make_pair(name1, sim1) : make_pair("Unknown", sim1);
}

// ---------------------------
// Проверка существования сотрудника
// ---------------------------
bool employee_exists(pqxx::connection& conn, const string& name) {
    pqxx::work txn(conn);
    auto result = txn.exec_params("SELECT id FROM face_embeddings WHERE name = $1", name);
    return !result.empty();
}

// ---------------------------
// Добавление лица в базу данных
// ---------------------------
void add_face_to_db(pqxx::connection& conn, const string& name, const vector<float>& embedding) {
    // Проверяем, существует ли уже сотрудник с таким именем
    if (employee_exists(conn, name)) {
        throw runtime_error("Employee with name '" + name + "' already exists!");
    }

    string embedding_str = embedding_to_string(embedding);

    pqxx::work txn(conn);
    txn.exec_params(
        "INSERT INTO face_embeddings (name, embedding) VALUES ($1, $2::vector)",
        name,
        embedding_str
    );
    txn.commit();
}

// ---------------------------
// Удаление лица из базы данных
// ---------------------------
void delete_face_from_db(pqxx::connection& conn, const string& name) {
    pqxx::work txn(conn);

    // Сначала удаляем связанные записи из work_schedule и attendance
    txn.exec_params("DELETE FROM work_schedule WHERE employee_id IN (SELECT id FROM face_embeddings WHERE name = $1)", name);
    txn.exec_params("DELETE FROM attendance WHERE employee_id IN (SELECT id FROM face_embeddings WHERE name = $1)", name);

    // Затем удаляем саму запись из face_embeddings
    txn.exec_params("DELETE FROM face_embeddings WHERE name = $1", name);
    txn.commit();
}

// ---------------------------
// Добавление сотрудника с графиком
// ---------------------------
void add_person_with_schedule(pqxx::connection& conn, const string& name,
    const vector<float>& embedding,
    const string& start, const string& end) {

    // Проверяем, существует ли уже сотрудник с таким именем
    if (employee_exists(conn, name)) {
        throw runtime_error("Employee with name '" + name + "' already exists!");
    }

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
// Отметка посещения
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
    }

    txn.commit();
}

// ---------------------------
// Получение списка всех сотрудников
// ---------------------------
vector<string> get_all_employees(pqxx::connection& conn) {
    vector<string> employees;
    pqxx::work txn(conn);

    auto result = txn.exec("SELECT name FROM face_embeddings ORDER BY name");
    for (const auto& row : result) {
        employees.push_back(row["name"].as<string>());
    }

    return employees;
}