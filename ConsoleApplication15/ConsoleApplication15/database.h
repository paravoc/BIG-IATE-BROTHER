#pragma once
#include <pqxx/pqxx>
#include <vector>
#include <string>
#include <utility> // для std::pair

// Получение эмбеддинга из базы
std::pair<std::string, float> check_face_in_db(pqxx::connection& conn, const std::vector<float>& embedding);

// Проверка существования сотрудника
bool employee_exists(pqxx::connection& conn, const std::string& name);

// Добавление лица
void add_face_to_db(pqxx::connection& conn, const std::string& name, const std::vector<float>& embedding);

// Удаление лица
void delete_face_from_db(pqxx::connection& conn, const std::string& name);

// Преобразование эмбеддинга в строку для PostgreSQL
std::string embedding_to_string(const std::vector<float>& embedding);

// Добавление сотрудника с графиком
void add_person_with_schedule(pqxx::connection& conn, const std::string& name,
    const std::vector<float>& embedding,
    const std::string& start, const std::string& end);

// Отметка посещения
void mark_attendance(pqxx::connection& conn, const std::string& name, const std::string& check_type);

// Получение списка всех сотрудников
std::vector<std::string> get_all_employees(pqxx::connection& conn);