/*
  svd.hpp
  SVD class.
*/

#ifndef SVD_HPP
#define SVD_HPP

#include <string>
#include <time.h>
#include <vector>
using namespace std;

typedef map<unsigned int, unsigned int> IdMap;
typedef IdMap::iterator IdMapItr;

/* Benchmark and timing datastructures. */
typedef pair<clock_t, std::string> Timing;

struct BenchInfo {
  unsigned int    count;
  float           sum;
  float           max;
  float           min;
};
typedef map< std::string, BenchInfo > BenchMap;

/* Main weight data datastructures */
struct Weight {
  unsigned int    user_id;
  unsigned int    item_id;
  float           weight;
  float           cache;
};

struct Item {
  unsigned int        sparse_id;
  unsigned int        weights_count;
  float               weights_sum;
  float               weights_avg;
  float               regularized_avg;
  float               offset_sum;
  float               deviation_sum;
  float               deviation;
  float               offset;
};

struct User : Item {
  // originally this was unique but in order to support SVDs for generic item type pairs, it's been folded into Item.
};

typedef map<unsigned int, float> WeightMap;
typedef WeightMap::iterator WeightMapItr;

class SVD {
  // Class to perform Simon Funk-style singular value decomposition on a table of weights.
  private:

    unsigned int                       total_weights;
    unsigned int                       baseline_count;
    unsigned int                       weights_count;
    unsigned int                       test_count;
    float                              weight_sum;
    float                              global_avg;       // mean of all weights.

    float                              min_weight;       // minimal value of weight
    float                              max_weight;       // maximum value of weight
    unsigned int                       features_count;   // number of features to train
    unsigned int                       min_epochs;       // minimum number of epochs to train a feature for
    unsigned int                       max_epochs;       // maximum number of epochs to train a feature for
    float                              min_improvement;  // minimal improvement required to keep training
    float                              learn_rate;       // learning rate
    float                              tikhonov;         // "Tikhonov" regularization coefficient; penalizes features by magnitude
    float                              f_init;           // Value that features should be initialized to
    unsigned int                       prior_weight;     // Weight of global average in determining regularized averages
    unsigned int                       min_weights;      // Minimal number of weights in order to calculate features for a user or item.

    vector<Weight>                     weights;
    vector<Weight>                     test_weights;

    IdMap                              user_ids;         // Map: sparse user_id => compact index
    vector<User>                       users;
    vector<WeightMap>                  user_weights;
    vector< vector<float> >            features_users;

    IdMap                              item_ids;          // Map: sparse item_id => compact index
    vector<unsigned int>               selected_items;    // compact itemIDs to output.
    vector<Item>                       items;
    vector<WeightMap>                  item_weights;
    vector< vector<float> >            features_items;

    vector<Timing>                     timings;           // Times at which certain points in the code are reached.
    BenchMap                           benchmarks;        // Durations for blocks of code.
    std::map<string, clock_t>          benchmark_starts;

    vector<float>                      feature_improvements;  // Amount by which each feature decreased the test RMSE in training.

    void                               LoadRow(unsigned int user_id, unsigned int item_id, float weight, bool baseline = false, bool test = false);
    void                               LoadCSV(std::string filename, bool baseline = false, bool test = false);

    inline float                       ClipWeight(float weight);
    inline float                       PredictWeight(unsigned int item_id, unsigned int user_id, unsigned int feature, float cache, bool trailing);

  public:
    SVD(unsigned int features, unsigned int min_epochs, unsigned int max_epochs, float min_improvement, float l_rate, float tikhonov, float f_init, unsigned int num_priors, unsigned int min_weights, float min_weight, float max_weight);
    ~SVD(void) { };
    inline void                        AddTiming(std::string description);
    void                               PrintTimings();

    inline void                        StartBenchmark(std::string name);
    inline void                        EndBenchmark(std::string name);
    void                               PrintBenchmarks();

    // convenience alias methods for LoadCSV.
    void                               LoadCSVBaseline(std::string filename);
    void                               LoadCSVWeights(std::string filename);
    void                               LoadCSVTest(std::string filename);

    // convenience alias methods for LoadRow.
    void                               LoadBaselineRow(unsigned int user_id, unsigned int item_id, float weight);
    void                               LoadWeightRow(unsigned int user_id, unsigned int item_id, float weight);
    void                               LoadTestRow(unsigned int user_id, unsigned int item_id, float weight);

    void                               DeleteWeight(unsigned int user_id, unsigned int item_id);

    // splits currently-loaded weights into train and test sets of size proportionate to test_percent.
    void                               PartitionWeights(unsigned int test_percent);

    // finds the weight corresponding to the given user-item pair.
    float                              FindWeight(unsigned int user_id, unsigned int item_id);
    float                              FindTestWeight(unsigned int user_id, unsigned int item_id);

    WeightMap                          UserWeights(unsigned int user_id);
    WeightMap                          ItemWeights(unsigned int item_id);

    // calculates global, user-, and item- wide statistics e.g. means, variances, and offsets from global mean.
    void                               CalcMetrics();

    // transforms all weights so that mean is zero, and, if deviation=true, variance is 1.
    void                               NormalizeWeights(bool deviation = true);

    // runs SVD and computes features.
    void                               Train(bool calculate_metrics = true);

    // calculates test RMSE with SVD's current features.
    float                              TestRMSE();

    // returns predicted score for given item and user, using computed features.
    inline float                       PredictWeight(unsigned int item_id, unsigned int user_id);

    // getters for class attributes.
    float                              MinWeight();
    float                              MaxWeight();
    unsigned int                       FeaturesCount();

    IdMap                              UserIDs();
    vector<User>                       Users();
    vector<float>                      UserFeatures(unsigned int user_id);    

    IdMap                              ItemIDs();
    vector<unsigned int>               SelectedItems();
    vector<Item>                       Items();
    vector<float>                      ItemFeatures(unsigned int item_id);

    vector<float>                      FeatureImprovements();
};

#endif