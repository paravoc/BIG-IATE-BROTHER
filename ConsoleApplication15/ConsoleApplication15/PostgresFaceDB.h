#ifndef POSTGRESFACEDB_H
#define POSTGRESFACEDB_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pqxx/pqxx>

class PostgresFaceDB {
public:
    // Конструктор/деструктор
    PostgresFaceDB(const std::string& connectionString);
    ~PostgresFaceDB();

    // Основные методы
    bool initialize();
    std::pair<std::string, float> identifyFace(const cv::Mat& faceEmbedding);
    bool registerFace(const cv::Mat& faceEmbedding, const std::string& name);
    bool updateName(const std::string& oldName, const std::string& newName);
    bool deletePerson(const std::string& name);

    // Утилиты
    static std::vector<float> convertMatToVector(const cv::Mat& mat);

private:
    pqxx::connection conn_;

    void createTables();
    std::string serializeEmbedding(const std::vector<float>& embedding);
};

#endif // POSTGRESFACEDB_H