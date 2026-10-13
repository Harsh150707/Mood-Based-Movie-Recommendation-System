#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace std;

class FeatureVector {
private:
    vector<double> values;
public:
    FeatureVector(){};
    FeatureVector(const vector<double>& vals) : values(vals) {}

    double distanceTo(const FeatureVector& other) const {
        double sum = 0.0;
        for (size_t i = 0; i < values.size() && i < other.values.size(); ++i) {
            double diff = values[i] - other.values[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
};

class Movie {
private:
    string title;
    string genre;
    double rating;
    FeatureVector features;

public:
    Movie(string t, string g, double r, FeatureVector f)
        : title(t), genre(g), rating(r), features(f) {}

    string getTitle() const { return title; }
    string getGenre() const { return genre; }
    double getRating() const { return rating; }
    const FeatureVector& getFeatures() const { return features; }
};

class KNNClassifier {
private:
    vector<Movie> trainingData;
    int k;

public:
    explicit KNNClassifier(int kNeighbors = 3) : k(kNeighbors) {}

    void fit(const vector<Movie>& data) { trainingData = data; }

    vector<Movie> predict(const FeatureVector& inputFeatures) {
        vector<pair<double, size_t>> distances;
        for (size_t i = 0; i < trainingData.size(); ++i) {
            double dist = inputFeatures.distanceTo(trainingData[i].getFeatures());
            distances.push_back({dist, i});
        }

        sort(distances.begin(), distances.end());

        vector<Movie> results;
        int count = min(k, static_cast<int>(distances.size()));
        for (int i = 0; i < count; ++i) {
            results.push_back(trainingData[distances[i].second]);
        }
        return results;
    }
};

class TMDBDatasetLoader {
private:
    static vector<string> parseCSVRow(const string& line) {
        vector<string> tokens;
        string token;
        bool in_quotes = false;
        
        for (size_t i = 0; i < line.length(); ++i) {
            char c = line[i];
            if (c == '"') {
                in_quotes = !in_quotes;
            } else if (c == ',' && !in_quotes) {
                tokens.push_back(token);
                token.clear();
            } else {
                token += c;
            }
        }
        tokens.push_back(token);
        return tokens;
    }

    static string extractFirstGenre(const string& genres_json) {
        size_t name_pos = genres_json.find("\"name\":");
        if (name_pos == string::npos) return "Drama";
        
        size_t start_quote = genres_json.find('"', name_pos + 7);
        if (start_quote == string::npos) return "Drama";
        
        size_t end_quote = genres_json.find('"', start_quote + 1);
        if (end_quote == string::npos) return "Drama";
        
        return genres_json.substr(start_quote + 1, end_quote - start_quote - 1);
    }

public:
    static vector<Movie> load(const string& filepath) {
        vector<Movie> movies;
        ifstream file(filepath);
        if (!file.is_open()) {
            cerr << "Error: Could not open " << filepath << "\n";
            return movies;
        }

        string line;
        getline(file, line); // Skip header row

        while (getline(file, line)) {
            vector<string> tokens = parseCSVRow(line);
            
            if (tokens.size() > 18) {
                string genres_json = tokens[1];
                string title = tokens[17];
                string strRating = tokens[18];

                if (!title.empty() && !strRating.empty()) {
                    try {
                        double rating = stod(strRating);
                        string genre = extractFirstGenre(genres_json);

                        double energy = (genre.find("Action") != string::npos || genre.find("Thriller") != string::npos || genre.find("Adventure") != string::npos) ? 0.9 : 0.3;
                        double valence = (genre.find("Comedy") != string::npos || genre.find("Animation") != string::npos || genre.find("Romance") != string::npos) ? 0.9 : 0.2;
                        double suspense = (genre.find("Horror") != string::npos || genre.find("Mystery") != string::npos || genre.find("Crime") != string::npos) ? 0.9 : 0.1;

                        movies.emplace_back(title, genre, rating, FeatureVector({energy, valence, suspense}));
                    } catch (...) {
                        continue;
                    }
                }
            }
        }
        return movies;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: ./ml_engine <energy> <valence> <suspense>\n";
        return 1;
    }

    double energy = stod(argv[1]);
    double valence = stod(argv[2]);
    double suspense = stod(argv[3]);

    FeatureVector userMood({energy, valence, suspense});

    vector<Movie> dataset = TMDBDatasetLoader::load("tmdb_5000_movies.csv");

    if (dataset.empty()) {
        cerr << "Warning: Loaded 0 records from tmdb_5000_movies.csv. Using fallback data.\n";
        dataset = {
            {"The Pursuit of Happyness", "Drama", 8.0, FeatureVector({0.3, 0.8, 0.2})},
            {"Mad Max: Fury Road", "Action", 8.1, FeatureVector({0.9, 0.4, 0.7})},
            {"Inception", "Sci-Fi", 8.8, FeatureVector({0.7, 0.3, 0.9})},
            {"Comedy Night", "Comedy", 7.5, FeatureVector({0.6, 0.9, 0.1})}
        };
    }

    KNNClassifier model(3);
    model.fit(dataset);
    vector<Movie> recommendations = model.predict(userMood);

    for (const auto& movie : recommendations) {
        cout << movie.getTitle() << "|" << movie.getGenre() << "|" << movie.getRating() << "\n";
    }

    return 0;
}
