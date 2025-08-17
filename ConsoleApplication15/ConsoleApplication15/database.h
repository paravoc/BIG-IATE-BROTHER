#pragma once
#include <pqxx/pqxx>
#include <vector>
#include <string>

using namespace std;

// Получение эмбеддинга из базы
pair<string, float> check_face_in_db(pqxx::connection& conn, const vector<float>& embedding);

// Добавление лица
void add_face_to_db(pqxx::connection& conn, const string& name, const vector<float>& embedding);

// Удаление лица
void delete_face_from_db(pqxx::connection& conn, const string& name);

// Преобразование эмбеддинга в строку для PostgreSQL
string embedding_to_string(const vector<float>& embedding);
