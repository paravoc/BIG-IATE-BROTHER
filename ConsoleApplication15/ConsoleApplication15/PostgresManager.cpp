#include "PostgresFaceDB.h"
#include <iostream>
#include <stdexcept>

PostgresFaceDB::PostgresFaceDB(const std::string& connectionString)
    : conn_(connectionString) {
    if (!conn_.is_open()) {
        throw std::runtime_error("Cannot connect to PostgreSQL database");
    }
}

PostgresFaceDB::~PostgresFaceDB() {
    if (conn_.is_open()) {
        conn_.close();
    }
}

bool PostgresFaceDB::initialize() {
    try {
        createTables();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void PostgresFaceDB::createTables() {
    pqxx::work txn(conn_);

    txn.exec(R"(
        CREATE TABLE IF NOT EXISTS persons (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id SERIAL PRIMARY KEY,
            person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
            embedding BYTEA NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_person_name ON persons(name);
    )");

    txn.commit();
}

std::pair<std::string, float> PostgresFaceDB::identifyFace(const cv::Mat& faceEmbedding) {
    try {
        pqxx::work txn(conn_);
        auto embeddingVec = convertMatToVector(faceEmbedding);
        std::string serialized = serializeEmbedding(embeddingVec);

        // Простейший поиск (для реального проекта используйте pgvector)
        auto res = txn.exec_params(
            "SELECT p.name FROM persons p "
            "JOIN face_embeddings fe ON p.id = fe.person_id "
            "LIMIT 1",
            pqxx::binary_cast(serialized));

        if (!res.empty()) {
            return { res[0][0].as<std::string>(), 1.0f }; // Заглушка для similarity
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Identification failed: " << e.what() << std::endl;
    }

    return { "Unknown", 0.0f };
}

bool PostgresFaceDB::registerFace(const cv::Mat& faceEmbedding, const std::string& name) {
    try {
        pqxx::work txn(conn_);
        auto embeddingVec = convertMatToVector(faceEmbedding);

        // Вставляем запись о человеке
        auto [id] = txn.query1<int>(
            "INSERT INTO persons (name) VALUES ($1) RETURNING id",
            name);

        // Сохраняем эмбеддинг
        txn.exec_params(
            "INSERT INTO face_embeddings (person_id, embedding) VALUES ($1, $2)",
            id, pqxx::binary_cast(serializeEmbedding(embeddingVec)));

        txn.commit();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Registration failed: " << e.what() << std::endl;
        return false;
    }
}

bool PostgresFaceDB::updateName(const std::string& oldName, const std::string& newName) {
    try {
        pqxx::work txn(conn_);
        txn.exec_params(
            "UPDATE persons SET name = $1 WHERE name = $2",
            newName, oldName);
        txn.commit();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Update failed: " << e.what() << std::endl;
        return false;
    }
}

bool PostgresFaceDB::deletePerson(const std::string& name) {
    try {
        pqxx::work txn(conn_);
        txn.exec_params("DELETE FROM persons WHERE name = $1", name);
        txn.commit();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Deletion failed: " << e.what() << std::endl;
        return false;
    }
}

// Вспомогательные методы
std::vector<float> PostgresFaceDB::convertMatToVector(const cv::Mat& mat) {
    CV_Assert(mat.isContinuous() && mat.type() == CV_32F);
    return std::vector<float>(mat.ptr<float>(), mat.ptr<float>() + mat.total());
}

std::string PostgresFaceDB::serializeEmbedding(const std::vector<float>& embedding) {
    std::string blob;
    blob.resize(embedding.size() * sizeof(float));
    std::memcpy(blob.data(), embedding.data(), blob.size());
    return blob;
}