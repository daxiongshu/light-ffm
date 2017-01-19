#ifndef _LIBFFM_H
#define _LIBFFM_H
#include <string>
#include <map>
#ifdef __cplusplus
extern "C" 
{

namespace ffm
{
#endif

typedef float ffm_float;
typedef double ffm_double;
typedef int ffm_int;
typedef long long ffm_long;



struct ffm_node
{
    ffm_int f;
    ffm_int j;
};

struct ffm_problem
{
    ffm_int n;
    ffm_int l;
    ffm_int m;
    ffm_long nx; // length of X[]
    ffm_long q; // len(Q[]) == q+1
    ffm_node *X;
    ffm_long *P;
    ffm_long *Q;
    ffm_float *Y;
};

struct ffm_model
{
    ffm_int n;
    ffm_int m;
    ffm_int k;
    ffm_float *W; // m*n*k
    bool normalization;
};

struct ffm_matrix
{
    // m rows, n columns
    ffm_int m;
    ffm_int n;
    ffm_float *W;
	
};

struct ffm_vector
{
    // m rows, n columns
    ffm_int m;
    ffm_float *W;

};


struct ffm_parameter
{
    bool unseen;
    ffm_float eta;
    ffm_float decay;
    bool flip;
    ffm_float lambda;
    ffm_float mu; // v = v*mu - eta * dx; mu=0 is just sgd
    ffm_int nr_iters;
    ffm_int k;
    ffm_int nr_threads;
    float margin;
    bool quiet;
    bool normalization;
    bool random;
    bool auto_stop;
    bool regression;
    int apk;
    bool auc; // default false
    bool mcc; // default false
    bool warm_start; // train from a pre-trained model, doesn't work
    int num_classes; // >2 for multiclassification  
    std::string model_path;
    std::string solver;
    bool useinter;
    int savefield; 
};



ffm_problem* ffm_read_problem_from_bin(char const *path);
void ffm_write_problem_to_bin(const ffm_problem* prob, char const *path);
ffm_problem* ffm_read_problem(char const *path, ffm_int num_class = 2, bool regression=false, bool quiet=false);

ffm_problem* read_batch(char const *path, int num_batch, int total_batches=10, bool is_test=false, ffm_int num_class = 2, bool regression=false, bool quiet=false);
void ffm_read_problem_to_batch(char const *path, int num_batch=10, ffm_int num_class = 2, bool regression=false, bool quiet=false);

void ffm_read_problem_to_batch_test(char const *path, int num_batch=10, ffm_int num_class = 2, bool regression=false, bool quiet=false);


int ffm_read_problem_to_disk(char const *txt_path, char const *bin_path);

void ffm_destroy_problem(struct ffm_problem **prob);

ffm_int ffm_save_model(ffm_model *model, char const *path);

ffm_model* ffm_load_model(char const *path);

void ffm_destroy_model(struct ffm_model **model);

ffm_parameter ffm_get_default_param();


void print_problem(ffm_problem* pr);



struct Option
{
    Option() : param(ffm_get_default_param()), nr_folds(1), do_cv(false), on_disk(false) {}
    std::string tr_path, va_path, model_path, test_path, output_path, fmap_path;
    ffm_parameter param;
    ffm_int nr_folds;
    bool do_cv, on_disk, hasLabel;
    bool trainbatch;
};
void ffm_train_batch(Option opt, ffm_problem *va, int* fieldmap, int num_fea);

#ifdef __cplusplus
} // namespace ffm

} // extern "C"
#endif

#endif // _LIBFFM_H
