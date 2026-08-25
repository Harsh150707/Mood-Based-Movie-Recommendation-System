#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

class FeatureVector {
private:
    std::vector<double> values;
public:
    FeatureVector() = default;
    FeatureVector(const std::vector<double>& vals) : values(vals) {}

    double distanceTo(const FeatureVector& other) const {
        double sum = 0.0;
        for (size_t i = 0; i < values.size() && i < other.values.size(); ++i) {
            double diff = values[i] - other.values[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

class Movie {
private:
    std::string title;
    std::string genre;
    double rating;
    FeatureVector features;

public:
    Movie(std::string t, std::string g, double r, FeatureVector f)
        : title(t), genre(g), rating(r), features(f) {}

    std::string getTitle() const { return title; }
    std::string getGenre() const { return genre; }
    double getRating() const { return rating; }
    const FeatureVector& getFeatures() const { return features; }
};

class KNNClassifier {
private:
    std::vector<Movie> trainingData;
    int k;

public:
    explicit KNNClassifier(int kNeighbors = 3) : k(kNeighbors) {}

    void fit(const std::vector<Movie>& data) { trainingData = data; }

    std::vector<Movie> predict(const FeatureVector& inputFeatures) {
        std::vector<std::pair<double, size_t>> distances;
        for (size_t i = 0; i < trainingData.size(); ++i) {
            double dist = inputFeatures.distanceTo(trainingData[i].getFeatures());
            distances.push_back({dist, i});
        }

        std::sort(distances.begin(), distances.end());

        std::vector<Movie> results;
        int count = std::min(k, static_cast<int>(distances.size()));
        for (int i = 0; i < count; ++i) {
            results.push_back(trainingData[distances[i].second]);
        }
        return results;
    }
};

class KaggleCSVLoader {
public:
    static std::vector<Movie> load(const std::string& filepath) {
        std::vector<Movie> movies;
        std::ifstream file(filepath);
        if (!file.is_open()) return movies;

        std::string line;
        std::getline(file, line); // Skip header

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string title, genre, strRating;

            std::getline(ss, title, ',');
            std::getline(ss, genre, ',');
            std::getline(ss, strRating, ',');

            if (!title.empty() && !strRating.empty()) {
                double rating = std::stod(strRating);
                double energy = (genre.find("Action") != std::string::npos || genre.find("Thriller") != std::string::npos) ? 0.9 : 0.3;
                double valence = (genre.find("Comedy") != std::string::npos || genre.find("Animation") != std::string::npos) ? 0.9 : 0.2;
                double suspense = (genre.find("Horror") != std::string::npos || genre.find("Mystery") != std::string::npos) ? 0.9 : 0.1;

                movies.emplace_back(title, genre, rating, FeatureVector({energy, valence, suspense}));
            }
        }
        return movies;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;

    double energy = std::stod(argv[1]);
    double valence = std::stod(argv[2]);
    double suspense = std::stod(argv[3]);

    FeatureVector userMood({energy, valence, suspense});

    // 1. Load Inventory File
    std::vector<Movie> dataset = KaggleCSVLoader::load("movies.csv");

    // 2. Train and Predict using OOP Model
    KNNClassifier model(3);
    model.fit(dataset);
    std::vector<Movie> recommendations = model.predict(userMood);

    // 3. Output Delimited Format for UI Parsing (Title|Genre|Rating)
    for (const auto& movie : recommendations) {
        std::cout << movie.getTitle() << "|" << movie.getGenre() << "|" << movie.getRating() << "\n";
    }

    return 0;
}