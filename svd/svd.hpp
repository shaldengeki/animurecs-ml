/*
  svd.hpp
  SVD class.
*/

#ifndef SVD_HPP
#define SVD_HPP

#include <string>
#include <mysql++.h>
using namespace mysqlpp;
using namespace std;
#include <time.h>
#include <vector>

typedef map<unsigned int, unsigned int> IdMap;
typedef IdMap::iterator idMapItr;

/* Benchmark and timing datastructures. */
typedef pair<clock_t, std::string> Timing;

struct BenchInfo {
  unsigned int    Count;
  float           Sum;
  float           Max;
  float           Min;
};
typedef map< std::string, BenchInfo > BenchMap;

/* Main weight data datastructures */
struct Data {
  unsigned int    UserID;
  unsigned int    ItemID;
  float           Weight;
  float           Cache;
};

struct Item {
  unsigned int        SparseID;
  std::string         Source;
  unsigned int        TypeID;
  unsigned int        WeightsCount;
  float               WeightSum;
  float               WeightAvg;
  float               RegularizedAvg;
  float               DeviationSum;
  float               Deviation;
  float               Offset;
};

struct User : Item {
  // originally this was unique but in order to support SVDs for generic entity type pairs, it's been folded into Item.
};


class SVD {
  // Class to perform Simon Funk-style singular value decomposition on a table of weights.
  private:
    Connection                         DbConn;
    std::string                        EntityType;

    unsigned int                       TotalWeights;
    unsigned int                       BaselineCount;
    unsigned int                       WeightsCount;
    unsigned int                       TestCount;
    float                              WeightSum;
    float                              GlobalAvg;        // mean of all weights.
    float                              ItemAvg;          // mean of item average weights.
    float                              UserAvg;          // average of user average weights.

    float                              MinWeight;        // minimal value of weight
    float                              MaxWeight;        // maximum value of weight
    float                              ScaleFactor;      // Value by which to scale all weights up (or down). Scaling is un-done upon output.
    float                              ScaledMin;        // MinWeight * ScaleFactor
    float                              ScaledMax;        // MaxWeight * ScaleFactor
    unsigned int                       Features;         // number of features to train
    unsigned int                       MinEpochs;        // minimum number of epochs to train a feature for
    unsigned int                       MaxEpochs;        // Not currently used
    float                              MinImprovement;   // minimal improvement required to keep training
    float                              LRate;            // learning rate
    float                              Tikhonov;         // "Tikhonov" regularization coefficient; penalizes features by magnitude
    float                              FInit;            // Value that features should be initialized to
    unsigned int                       NumPriors;        // Weight of global average in determining regularized averages
    unsigned int                       MinWeights;       // Minimal number of weights in order to calculate features for a user or item.

    unsigned int                       PartitionSize;
    vector<Data>                       Weights;
    vector<Data>                       TestWeights;

    IdMap                              BaselineUserIDs;  // Map: sparse userID => compact index
    IdMap                              UserIDs;          // Map: sparse userID => compact index
    vector<User>                       Users;
    vector< vector<float> >            FeatureUsers;

    IdMap                              ItemIDs;           // Map: sparse itemID => compact index
    vector<unsigned int>               SelectedItems;     // compact itemIDs to output.
    vector<Item>                       Items;
    vector< vector<float> >            FeatureItems;

    vector<Timing>                     Timings;           // Times at which certain points in the code are reached.
    BenchMap                           Benchmarks;        // Durations for blocks of code.
    std::map<string, clock_t>          BenchmarkStarts;

    void                               LoadRow(unsigned int user_id, unsigned int item_id, float weight, bool baseline = false, bool test = false);
    inline float                       ClipWeight(float weight);
    inline float                       PredictWeight(unsigned int itemID, unsigned int userID, unsigned int feature, float cache, bool trailing);
    inline float                       PredictWeight(unsigned int itemID, unsigned int userID);

  public:
    SVD(Connection& dbConn, std::string type, unsigned int features, unsigned int min_epochs, unsigned int max_epochs, float min_improvement, float l_rate, float tikhonov, float f_init, unsigned int num_priors, unsigned int min_weights, float min_weight, float max_weight);
    ~SVD(void) { };
    inline void                         AddTiming(std::string description);
    void                                PrintTimings();

    inline void                         StartBenchmark(std::string name);
    inline void                         EndBenchmark(std::string name);
    void                                PrintBenchmarks();

    void                                LoadCSV(std::string filename, bool baseline = false, bool test = false);
    void                                LoadBaseline(std::string baseline_table);
    void                                LoadWeights(std::string weights_table);
    void                                LoadTests(std::string test_table);

    void                                CalcMetrics();
    void                                NormalizeWeights(bool deviation = true);
    void                                CalcFeatures();
    void                                RunTest();

    void                                SaveModel(std::string global_table, std::string group_data_table, std::string feature_table);
};

#endif