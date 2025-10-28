#pragma once
#include <pqxx/pqxx>
#include <vector>
#include <string>
#include <utility> // для std::pair

// Получение эмбеддинга из базы
std::pair<std::string, float> check_face_in_db(pqxx::connection& conn, const std::vector<float>& embedding);

// Добавление лица
void add_face_to_db(pqxx::connection& conn, const std::string& name, const std::vector<float>& embedding);

// Удаление лица
void delete_face_from_db(pqxx::connection& conn, const std::string& name);

// Преобразование эмбеддинга в строку для PostgreSQL
std::string embedding_to_string(const std::vector<float>& embedding);