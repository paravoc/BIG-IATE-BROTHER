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
    pqxx::work txn(conn); // Начало транзакции

    // Запрос: находим двух лучших совпадений по cosine similarity
    auto result = txn.exec_params(
        "SELECT name, 1 - (embedding <#> $1::vector) as similarity "
        "FROM face_embeddings "
        "ORDER BY similarity DESC LIMIT 2",   // берём двух лучших
        embedding_str
    );

    if (result.empty()) return { "Unknown", 0.0f }; // Если совпадений нет

    // Получаем лучшего кандидата
    float sim1 = result[0]["similarity"].as<float>() * 100.0f; // В процентах
    string name1 = result[0]["name"].as<string>();

    // Если есть второй кандидат
    float sim2 = 0;
    string name2 = "";
    if (result.size() > 1) {
        sim2 = result[1]["similarity"].as<float>() * 100.0f;
        name2 = result[1]["name"].as<string>();
    }

    // Если разница между первым и вторым меньше 5%, спрашиваем пользователя
    if (sim1 < 5.0f && !name2.empty()) {
        cout << "Multiple similar faces detected: '" << name1 << "' and '" << name2
            << "'. Enter correct name: ";
        string user_input;
        getline(cin, user_input);
        return { user_input.empty() ? "Unknown" : user_input, sim1 };
    }

    return sim1 >= 140.0f ? make_pair(name1, sim1) : make_pair("Unknown", sim1);
}

// ---------------------------
// Добавление лица в базу данных
// ---------------------------
void add_face_to_db(pqxx::connection& conn, const string& name, const vector<float>& embedding) {
    string embedding_str = embedding_to_string(embedding);

    pqxx::work txn(conn); // Начало транзакции
    txn.exec_params(
        "INSERT INTO face_embeddings (name, embedding) VALUES ($1, $2::vector)",
        name,
        embedding_str
    );
    txn.commit(); // Сохраняем изменения
}

// ---------------------------
// Удаление лица из базы данных
// ---------------------------
void delete_face_from_db(pqxx::connection& conn, const string& name) {
    pqxx::work txn(conn); // Начало транзакции
    txn.exec_params("DELETE FROM face_embeddings WHERE name = $1", name);
    txn.commit(); // Сохраняем изменения
}
