#pragma GCC diagnostic ignored "-Wunused-result" 
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <new>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <pmmintrin.h>
#include "FileUtil.h"
#include "StringUtil.h"
#include <ctime>
#include <cassert>
#include "Sort.h"

#if defined USEOMP
#include <omp.h>
#endif

#include "ffm.h"

namespace ffm {

namespace {

using namespace std;

ffm_int const kALIGNByte = 16;
ffm_int const kALIGN = kALIGNByte/sizeof(ffm_float);
ffm_int const kCHUNK_SIZE = 10000000;
ffm_int const kMaxLineSize = 100000;
vector<ffm_float> normalize(ffm_problem &prob);

void pred2order(vector<float>& pred, vector<int>& order){

    map<float, int> mymap;
    assert(pred.size()==order.size());
    for (int i=0;i<pred.size();i++){
	mymap[pred[i]] = i;
    }
    Sort<float>::quickSort(pred);
    for (int i=0;i<pred.size();i++){
	order[pred.size() -1 - i] = mymap[pred[i]];
    }         
}

float apk(vector<int>& act, vector<int>& pred, int k){

    int lact = act.size();
    int lpred = pred.size();

    if (lpred>k){
	lpred = k;
    }

    if (lact==0) return 0;
    float score = 0;
    int num_hits = 0;

    for(int i=0; i<lpred; i++){
	bool find  =  false;
	for(int j=0; j<lact; j++){
	    if (act[j]==pred[i]){
		find = true;
		break;
	    }
	}
        if (find){
	    num_hits++;
	    score += num_hits*1.0/(i+1);
	}
    }
    if (lact>k) lact = k ;

    return score/lact;
 
}

inline ffm_float wTx(
    ffm_node *begin_context,
    ffm_node *end_context,
    ffm_node *begin,
    ffm_node *end,
    ffm_float r_context,
    ffm_float r,
    ffm_model &model,
    int* feamap,
    int num_fea,
    int savefield,
    int turnoff = false,
    bool flip = false, 
    ffm_int class_id=0,
    ffm_int num_class=2,
    ffm_float kappa=0, // kappa is the gradient! 
    ffm_float eta=0, 
    ffm_float lambda=0, 
    bool do_update=false,
    string solver = "adagrad",
    float mu = 0)
{
    //assert(ignore_fields);
    ffm_long align0 = (ffm_long)model.k*2;
    ffm_long align1 = (ffm_long)model.m*align0;
    ffm_long align2 = (ffm_long)model.n*align1;

    __m128 XMMkappa = _mm_set1_ps(kappa);  // set all 4 words with the same value
    __m128 XMMeta = _mm_set1_ps(eta);
    __m128 XMMlambda = _mm_set1_ps(lambda);
    __m128 XMMmu = _mm_set1_ps(mu);
    __m128 XMMmup = _mm_set1_ps(1-mu);
    __m128 XMMt = _mm_setzero_ps(); // Clears all four values

    r = sqrt(r*r_context);

    for(ffm_node *N1 = begin_context; N1 != end_context; N1++)
    {
        ffm_int j1 = N1->j;
        ffm_int f1 = N1->f;// % savefield;
        ffm_float v1 = 1;//N1->v;
        if(j1 >= model.n)// || f1 >= model.m)
            continue;
	for(ffm_node *N2 = begin; N2 != end; N2++)
        {
            ffm_int j2 = N2->j;
            ffm_int f2 = feamap[N1->f*num_fea+N2->f];
	    f1 = feamap[N2->f*num_fea+N1->f];

	    if(turnoff){
	    if(flip==false){
	    	f2 = N2->f;
	        f1 = feamap[N2->f*num_fea+N1->f];
	    }else{
 		f1 = N1->f;
                f2 = feamap[N1->f*num_fea+N2->f];
	    }}

	    if (f1 == N1->f)
                f1 = f1 % savefield;
            ffm_float v2 = 1;//N2->v;
            bool bypass = false;
	    if (f1 == -1 || f2 == -1 || f1 >= model.m || f2 >= model.m)
		bypass = true;
	    //if (bypass){
	    //	printf("bypass %d %d\n",N1->f,N2->f);
	    //}
            if(j2 >= model.n || f2 >= model.m || bypass)
                continue;
 
            ffm_float *w1 = model.W + j1*align1 + f2*align0 + class_id*align2;
            ffm_float *w2 = model.W + j2*align1 + f1*align0 + class_id*align2;

            __m128 XMMv = _mm_set1_ps(v1*v2*r); // Sets all four words with the same value

            if(do_update)
            {
                __m128 XMMkappav = _mm_mul_ps(XMMkappa, XMMv); // Multiplies the four single-precision, floating-point values of a and b.

                ffm_float *wg1 = w1 + model.k;
                ffm_float *wg2 = w2 + model.k;
                for(ffm_int d = 0; d < model.k; d += 4)
                {
                    __m128 XMMw1 = _mm_load_ps(w1+d); // Loads four single-precision, floating-point values.
                    __m128 XMMw2 = _mm_load_ps(w2+d);

		    __m128 XMMg1 = _mm_add_ps(
                                   _mm_mul_ps(XMMlambda, XMMw1),
                                   _mm_mul_ps(XMMkappav, XMMw2));
                    __m128 XMMg2 = _mm_add_ps(
                                   _mm_mul_ps(XMMlambda, XMMw2),
                                   _mm_mul_ps(XMMkappav, XMMw1));

		    if (solver.compare("adagrad") == 0){  
			__m128 XMMwg1 = _mm_load_ps(wg1+d);
                        __m128 XMMwg2 = _mm_load_ps(wg2+d);
 

                        XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
                        XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));
			
                        XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMeta, 
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
                        XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMeta, 
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

			//if (turnoff == 0)
			_mm_store_ps(wg1+d, XMMwg1);
			//if (turnoff == 0)
                        _mm_store_ps(wg2+d, XMMwg2);
		    }else if (solver.compare("rmsprop") == 0){
			__m128 XMMwg1 = _mm_load_ps(wg1+d);
                        __m128 XMMwg2 = _mm_load_ps(wg2+d);


                        //XMMwg1 = _mm_add_ps(_mm_mul_ps(XMMwg1,XMMmu), _mm_mul_ps(XMMmup,_mm_mul_ps(XMMg1, XMMg1)));
                        //XMMwg2 = _mm_add_ps(_mm_mul_ps(XMMwg2,XMMmu), _mm_mul_ps(XMMmup,_mm_mul_ps(XMMg2, XMMg2)));

			XMMwg1 = _mm_add_ps(_mm_mul_ps(XMMwg1,XMMmu), _mm_mul_ps(XMMg1, XMMg1));
			XMMwg2 = _mm_add_ps(_mm_mul_ps(XMMwg2,XMMmu), _mm_mul_ps(XMMg2, XMMg2));
                        XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMeta,
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
                        XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMeta,
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

                        _mm_store_ps(wg1+d, XMMwg1);
                        _mm_store_ps(wg2+d, XMMwg2);
		    }
		    else if (solver.compare("sgd") == 0){
			XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMeta,
                            XMMg1));
                        XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMeta,
                            XMMg2));
		    }else if(solver.compare("momentum") == 0){
    			__m128 XMMwg1 = _mm_load_ps(wg1+d);
                        __m128 XMMwg2 = _mm_load_ps(wg2+d);

                        XMMwg1 = _mm_sub_ps(_mm_mul_ps(XMMwg1, XMMmu), _mm_mul_ps(XMMeta, XMMg1));
                        XMMwg2 = _mm_sub_ps(_mm_mul_ps(XMMwg2, XMMmu), _mm_mul_ps(XMMeta, XMMg2));

                        XMMw1 = _mm_add_ps(XMMw1, XMMwg1);
                        XMMw2 = _mm_add_ps(XMMw2, XMMwg2);

                        _mm_store_ps(wg1+d, XMMwg1);
                        _mm_store_ps(wg2+d, XMMwg2);
		    }
			
                    _mm_store_ps(w1+d, XMMw1);
                    _mm_store_ps(w2+d, XMMw2);

                }
            }
            else
            {
                for(ffm_int d = 0; d < model.k; d += 4)
                {
                    //printf("w1: %f\n", *(w1+d));
                    __m128  XMMw1 = _mm_load_ps(w1+d);
                    __m128  XMMw2 = _mm_load_ps(w2+d);

                    XMMt = _mm_add_ps(XMMt, 
                           _mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv));
                }
            }
        }
    }

    if(do_update)
        return 0;

    XMMt = _mm_hadd_ps(XMMt, XMMt);  // horizontal add
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    ffm_float t;
    _mm_store_ss(&t, XMMt);

    return t;
}

void mypredict_batch(Option opt, ffm_model *model, int* feamap, int num_fea)
{
    //printf("num_classes:%d\n",opt.param.num_classes);
    int num_batch = 10;
    ofstream f_out(opt.output_path);
    for (ffm_int q = 0; q < num_batch; q++){
        ffm_problem *va;
        ffm_float va_loss = 0;
        //printf("predict batch\n");    
        va = read_batch(opt.test_path.c_str(), q, num_batch, true, opt.param.num_classes, opt.param.regression, true);
        if(va == nullptr)
        {
             cerr << "cannot load " << opt.test_path << endl << flush;
            //return nullptr;
        }
        vector<ffm_float> R_va;
        if(opt.param.normalization)
        {
            R_va = normalize(*va);
        }
        else
        {
            R_va = vector<ffm_float>(va->l, 1);
        }

	for(ffm_int ix = 0; ix < va->q; ix++){

	    ffm_node *begin_context = &va->X[va->P[va->Q[ix]]];
            ffm_node *end_context = &va->X[va->P[va->Q[ix]+1]];
	    ffm_float r_context = R_va[va->Q[ix]];
            for(ffm_int i = va->Q[ix]+1; i < va->Q[ix+1]; i++)
            {
            	//printf("predict sample %d \n",i);
            	ffm_float y = va->Y[i];
            	ffm_node *begin = &va->X[va->P[i]];
            	ffm_node *end = &va->X[va->P[i+1]];
            	ffm_float r = R_va[i];
            	ffm_float yp;
                //printf("1\n");
                    //printf("3\n");

		yp = wTx(begin_context, end_context, begin, end,  r_context, r, *model, feamap, num_fea, opt.param.savefield);
		if (opt.param.useinter){
                for(ffm_int  ic = va->Q[ix]+1; ic < va->Q[ix+1]; ic++){
                    if (i==ic)
                         continue;

                    ffm_node *begin_context1 = &va->X[va->P[ic]];

                    ffm_node *end_context1 = &va->X[va->P[ic+1]];

                    ffm_float r_context1 = R_va[ic];

                    yp += wTx(begin_context1, end_context1, begin, end,  r_context1, r, *model, feamap, num_fea, opt.param.savefield, 1, opt.param.flip);
                }
		}

		yp = 1.0/(1+exp(-yp));
                    if (opt.hasLabel)
                  	f_out <<i<<","<< yp << ","<<y<<"\n";
                    else
                  	f_out <<i<<","<< yp << "\n";
            }
	}
        ffm_destroy_problem(&va);
    }
    f_out.close();
}

ffm_float* malloc_aligned_float(ffm_long size)
{
    void *ptr;

#ifdef _WIN32
    ptr = _aligned_malloc(size*sizeof(ffm_float), kALIGNByte);
    if(ptr == nullptr)
        throw bad_alloc();
#else
    int status = posix_memalign(&ptr, kALIGNByte, size*sizeof(ffm_float));
    if(status != 0)
        throw bad_alloc();
#endif
    
    return (ffm_float*)ptr;
}

ffm_model* init_model(ffm_int n, ffm_int m, ffm_parameter param)
{
    ffm_int k_aligned = (ffm_int)ceil((ffm_double)param.k/kALIGN)*kALIGN;

    ffm_model *model = new ffm_model;
    model->n = n;  // maximum number of fields
    model->k = k_aligned;
    model->m = m;  // maximum number of features in a single field
    model->W = nullptr;
    model->normalization = param.normalization;
    try
    {
	if (param.num_classes==2){
            model->W = malloc_aligned_float((ffm_long)n*m*k_aligned*2);
 	}else{
 	    model->W = malloc_aligned_float((ffm_long)n*m*k_aligned*2*param.num_classes);	
	    //printf("init model num class: %d\n ", param.num_classes);	
        }	
    }
    catch(bad_alloc const &e)
    {
        ffm_destroy_model(&model);
        throw;
    }

    ffm_float coef = 1.0f/sqrt(param.k);
    ffm_float *w = model->W;

    default_random_engine generator;
    uniform_real_distribution<ffm_float> distribution(0.0, 1.0);
    ffm_int nc = param.num_classes;
    if (param.num_classes==2) nc=1;
    for(ffm_int q = 0; q < nc; q++){
    	for(ffm_int j = 0; j < model->n; j++)
    	{
            for(ffm_int f = 0; f < model->m; f++)
            {
            	for(ffm_int d = 0; d < param.k; d++, w++)
                    *w = coef*distribution(generator);
            	for(ffm_int d = param.k; d < k_aligned; d++, w++)
                    *w = 0;
            	for(ffm_int d = k_aligned; d < 2*k_aligned; d++, w++)
                    *w = 1;
            }
    	}
    }
    return model;
}

void shrink_model(ffm_model &model, ffm_int k_new, ffm_int num_classes = 1)
{
    for(ffm_int k = 0; k < num_classes; k++)
    	for(ffm_int j = 0; j < model.n; j++)
    	{
            for(ffm_int f = 0; f < model.m; f++)
            {
            	ffm_float *src = model.W + ((ffm_long)k*model.m*model.n+(ffm_long)j*model.m+f)*model.k*2;
            	ffm_float *dst = model.W + ((ffm_long)k*model.m*model.n+(ffm_long)j*model.m+f)*k_new;
            	copy(src, src+k_new, dst);
            }
    	}

    model.k = k_new;
}

vector<ffm_float> normalize(ffm_problem &prob)
{
    vector<ffm_float> R(prob.l);
#if defined USEOMP
#pragma omp parallel for schedule(static)
#endif
    for(ffm_int i = 0; i < prob.l; i++)
    {
        ffm_float norm = 0;
        for(ffm_long p = prob.P[i]; p < prob.P[i+1]; p++)
            norm += 1;
        R[i] = 1/norm;
    }

    return R;
}

float train_b(
    shared_ptr<ffm_model>& model,
    bool start_x,
    bool end_x,
    int iterx,
    ffm_problem *tr, 
    vector<ffm_int> &order, 
    ffm_parameter param, 
    int* feamap, 
    int num_fea,
    ffm_problem *va=nullptr
    )
{
#if defined USEOMP
    ffm_int old_nr_threads = omp_get_num_threads();
    omp_set_num_threads(param.nr_threads);
#endif

    ffm_float eta0 = param.eta;
    float final_loss= 100; 
    vector<ffm_float> R_tr, R_va;
    if(param.normalization)
    {
        R_tr = normalize(*tr);
        if(va != nullptr)
            R_va = normalize(*va);
    }
    else
    {
        R_tr = vector<ffm_float>(tr->l, 1);
        if(va != nullptr)
            R_va = vector<ffm_float>(va->l, 1);
    }
    //printf("initialize done\n");
    bool auto_stop = param.auto_stop && va != nullptr && va->l != 0;

    ffm_int k_aligned = (ffm_int)ceil((ffm_double)param.k/kALIGN)*kALIGN;
    ffm_long w_size = (ffm_long)model->n * model->m * k_aligned * 2;
    vector<ffm_float> prev_W;
    ffm_double best_va_loss = numeric_limits<ffm_double>::max();

    if(!param.quiet && start_x)
    {
        if(param.auto_stop && (va == nullptr || va->l == 0))
            cerr << "warning: ignoring auto-stop because there is no validation set" << endl;

        cout.width(4);
        cout << "iter";
        cout.width(13);
        if (!param.regression){
            if (param.num_classes==2){
                if (!param.auc){
                    if (!param.mcc)
                        cout << "tr_logloss";
                    else
                        cout << "tr_mcc";
                }
                else
                    cout << "tr_auc";
            }else
                cout << "tr_mlogloss";
        }

	else
	    cout << "tr_RMSLE";
        if(va != nullptr && va->l != 0)
        {
            cout.width(13);
            if (!param.regression){
                if (param.num_classes==2){
                    if (!param.auc){
                        if (!param.mcc)
                            cout << "va_logloss";
                        else
                            cout << "va_mcc";
                    }
                    else
                        cout << "va_auc";
                }else
                    cout << "va_mlogloss";
            }
            else
                cout << "va_RMSLE";
        }
	cout.width(13);
        cout << "va_apk";

	cout.width(13);
	cout << "Time";
        cout << endl;
    }
    double startx = omp_get_wtime();

    for(ffm_int iter = 1; iter <= param.nr_iters; iter++)
    {
        ffm_double tr_loss = 0;
        ffm_double tr_loss_unseen = 0;
        if(param.random)
            random_shuffle(order.begin(), order.end());
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+: tr_loss)
#endif
        for(ffm_int ii = 0; ii < (ffm_int)order.size(); ii++)
        {
            ffm_int ix = order[ii];

	    ffm_node *begin_context = &tr->X[tr->P[tr->Q[ix]]];

            ffm_node *end_context = &tr->X[tr->P[tr->Q[ix]+1]];

	    ffm_float r_context = R_tr[tr->Q[ix]];

	    vector<ffm_int> order_big(tr->Q[ix+1]-tr->Q[ix]-1);
    	    for(ffm_int i = 0; i < tr->Q[ix+1]-tr->Q[ix]-1; i++)
                order_big[i] = tr->Q[ix]+1+i;
	    random_shuffle(order_big.begin(), order_big.end()); 

	    vector<ffm_float> preds(tr->Q[ix+1]-tr->Q[ix]-1);
	    for(ffm_int iz =0; iz < tr->Q[ix+1]-tr->Q[ix]-1; iz++){

		ffm_int i = order_big[iz];
		ffm_int ib = i - (tr->Q[ix]+1);
		//if (i%10!=iterx%10) continue;
            	ffm_float y = tr->Y[i];

            	ffm_node *begin = &tr->X[tr->P[i]];

            	ffm_node *end = &tr->X[tr->P[i+1]];

            	ffm_float r = R_tr[i];
		
		ffm_float t = wTx(begin_context, end_context, begin, end,  r_context, r, *model, feamap, num_fea, param.savefield);	
		if (param.useinter){
            	for(ffm_int ic =0; ic < tr->Q[ix+1]-tr->Q[ix]-1; ic++){
		    if (i==order_big[ic])
			continue;

		    ffm_node *begin_context1 = &tr->X[tr->P[order_big[ic]]];

                    ffm_node *end_context1 = &tr->X[tr->P[order_big[ic]+1]];

                    ffm_float r_context1 = R_tr[order_big[ic]];

            	    t += wTx(begin_context1, end_context1, begin, end,  r_context1, r, *model, feamap, num_fea, param.savefield, 1, param.flip);
		}
		}
		preds[ib] = t;
	    }
	    ffm_float ypsum = 0;
	    ffm_float maxp = -1000;
	    for(ffm_int iz =0; iz < tr->Q[ix+1]-tr->Q[ix]-1; iz++){
		//ypsum += exp(preds[iz]);
		if (maxp<preds[iz])
		    maxp = preds[iz];
	    }
	    for(ffm_int iz =0; iz < tr->Q[ix+1]-tr->Q[ix]-1; iz++){
		preds[iz] -= maxp;
	    }
	    for(ffm_int iz =0; iz < tr->Q[ix+1]-tr->Q[ix]-1; iz++){
                ypsum += exp(preds[iz]);
            }
	    for(ffm_int iz =0; iz < tr->Q[ix+1]-tr->Q[ix]-1; iz++){
		ffm_int i = order_big[iz];
                ffm_int ib = i - (tr->Q[ix]+1);
		ffm_float yp = exp(preds[ib])/ypsum;
		ffm_float y = tr->Y[i];
	    	ffm_float kappa = y>0? yp-y:yp;	
		ffm_node *begin = &tr->X[tr->P[i]];

                ffm_node *end = &tr->X[tr->P[i+1]];

                ffm_float r = R_tr[i]; 
		if (y>0)
		    tr_loss += (-log(yp)); 
		if (param.useinter){
		for(ffm_int ic =0; ic < tr->Q[ix+1]-tr->Q[ix]-1; ic++){
		    if (i==order_big[ic])
                        continue;

                    ffm_node *begin_context1 = &tr->X[tr->P[order_big[ic]]];

                    ffm_node *end_context1 = &tr->X[tr->P[order_big[ic]+1]];

                    ffm_float r_context1 = R_tr[order_big[ic]];   
		    wTx(begin_context1, end_context1, begin, end,  r_context1, r, *model, feamap, num_fea, param.savefield, 1, param.flip, 0, 2, kappa, eta0, param.lambda, true, param.solver, param.mu);

		}}
            	wTx(begin_context, end_context, begin, end,  r_context, r, *model, feamap, num_fea, param.savefield, 0, 0, 0, 2, kappa, eta0, param.lambda, true, param.solver, param.mu);
	       
	    }
 	    	
        }

        if(!param.quiet && end_x)
        {
            tr_loss /= (tr->l-tr->q);
            cout.width(4);
            cout << iterx;
            cout.width(13);
            if (!param.regression){
               
                   
                        cout << fixed << setprecision(5) << tr_loss;
                    
            
                
            }else
                cout << fixed << setprecision(5) << sqrt(tr_loss);
            if(va != nullptr && va->l != 0)
            {
                ffm_double va_loss = 0;
		ffm_double va_apk = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+:va_loss,va_apk)
#endif
                for(ffm_int ix = 0; ix < va->q; ix++)
                {

		    ffm_node *begin_context = &va->X[va->P[va->Q[ix]]];

 	            ffm_node *end_context = &va->X[va->P[va->Q[ix]+1]];

		    ffm_float r_context = R_va[va->Q[ix]];

		    vector<int> act;	
		    vector<float> pred;
		    vector<int> orderx;
		    for(ffm_int i = va->Q[ix]+1; i < va->Q[ix+1]; i++){

                        ffm_float y = va->Y[i];
			int idx = i - va->Q[ix] - 1;
			if (y>0) act.push_back(idx);
		
                    	ffm_node *begin = &va->X[va->P[i]];

                    	ffm_node *end = &va->X[va->P[i+1]];

                    	ffm_float r = R_va[i];
			ffm_float t = wTx(begin_context, end_context, begin, end,  r_context, r, *model, feamap, num_fea, param.savefield);
			if (param.useinter){
			for(ffm_int  ic = va->Q[ix]+1; ic < va->Q[ix+1]; ic++){
	                    if (i==ic)
        	                 continue;
	
        	            ffm_node *begin_context1 = &va->X[va->P[ic]];

                	    ffm_node *end_context1 = &va->X[va->P[ic+1]];

                    	    ffm_float r_context1 = R_va[ic];


                            t += wTx(begin_context1, end_context1, begin, end,  r_context1, r, *model, feamap, num_fea, param.savefield, 1, param.flip);
                        }}
			pred.push_back(t);
                        orderx.push_back(idx);
			ffm_float expnyt = exp(-y*t);

                        va_loss += log(1+expnyt);

		    }

		    pred2order(pred,orderx);
		    va_apk += apk(act,orderx,param.apk);		    

                }
                va_loss /= (va->l-va->q);
	 	va_apk /= (va->q);
                cout.width(13);
  		if (param.regression)
                    cout << fixed << setprecision(5) << sqrt(va_loss);
                else{
                    
                   
                            cout << fixed << setprecision(5) << va_loss;
                        
                   
                   
                }
		cout.width(13);
		cout << fixed << setprecision(5) << va_apk;
		final_loss = -va_apk;
            }
	    double duration = omp_get_wtime() - startx;
	    cout.width(13);
	    cout << fixed << setprecision(1) << duration<<"s";
            cout << endl;
        }
	eta0 = param.eta * exp(-param.decay*iter);
    }



#if defined USEOMP
    omp_set_num_threads(old_nr_threads);
#endif

    return final_loss;
    
}


} // unnamed namespace

ffm_problem* read_batch(char const *path, int num_batch, int total_batches, bool is_test, ffm_int num_class, bool regression, bool quiet){



    if(strlen(path) == 0)
        return nullptr;

    std::string fileNameBin = std::string(path)+"_"+std::to_string(num_batch)+".b";
    //printf("-1\n"); 
    if (FileUtil::fileExist(fileNameBin) == 0){
	if (is_test)
	    ffm_read_problem_to_batch_test(path, total_batches, num_class, regression, quiet);
	else
	    ffm_read_problem_to_batch(path, total_batches, num_class, regression, quiet);
    }

    if (!quiet)
	printf("load bin: %s\n", fileNameBin.c_str());

    std::clock_t startTime = std::clock();
    ffm_problem* problem = ffm_read_problem_from_bin(fileNameBin.c_str());	
    double duration = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
    if (!quiet){
        printf("load time %.1fs\n", duration);
        printf("features: %d fields:%d\n",problem->n,problem->m);
    }
    //printf("read batch done\n");
    return problem; 
    

}

void ffm_read_problem_to_batch_test(char const *path, int num_batch, ffm_int num_class, bool regression, bool quiet)
{
    if(strlen(path) == 0)
        return ;
    std::clock_t startTime = std::clock();
    
    printf("load raw: %s\n", path);
    FILE *f = fopen(path, "r");
    if(f == nullptr)
        return ;

    

    char line[kMaxLineSize];
    int lines = 0;
    ffm_long* nnz = new ffm_long[num_batch];
    ffm_int* line_ = new ffm_int[num_batch];
    ffm_long* nnq = new ffm_long[num_batch];

    for(ffm_int i = 0; i < num_batch; i++){
        nnz[i] = 0;
	line_[i] = 0;
	nnq[i] = 0;
    }
    
    int group = 0;

    for(; fgets(line, kMaxLineSize, f) != nullptr; lines++)
    {
        char *mark = strtok(line, " \t");
	if (*mark == 'x'){
	    group++;
	    
	}
    }
    rewind(f);
    printf("total groups: %d\n", group);
    for(ffm_int i = 0; i < num_batch-1; i++){

        nnq[i] = group/num_batch;
    }
    nnq[num_batch-1] = group - (num_batch-1)*nnq[0];
 
    ffm_int* linesum = new ffm_int[num_batch];
    linesum[0] = nnq[0];
    for(ffm_int i = 1; i < num_batch; i++){

        linesum[i] = linesum[i-1] + nnq[i];
    }

    int linec = 0;
    int stx = 0;
    rewind(f);

    //printf("3\n");
    group = 0;
    for(; fgets(line, kMaxLineSize, f) != nullptr;)
    {
	char *mark = strtok(line, " \t");
        if (*mark == 'x'){
            group++;

        }

        if (group > linesum[stx]) stx++;
	line_[stx]++;
        for(; ; nnz[stx]++)
        {
            char *ptr = strtok(nullptr," \t");
            if(ptr == nullptr || *ptr == '\n')
                break;
        }
    }


    for (ffm_int q = 0; q < num_batch; q++){
	printf("batch %d: groups:%d items:%d lines:%d linesum:%d \n", q, nnq[q], nnz[q],line_[q], linesum[q]);
    } 
    // start to do here next !!!
    for (ffm_int q = 0; q < num_batch; q++){
        rewind(f);
	ffm_problem *prob = new ffm_problem;
        prob->l = line_[q];
        prob->n = 0;
        prob->m = 0;
        prob->nx = nnz[q];
	prob->q = nnq[q];
        prob->X = new ffm_node[nnz[q]];
        prob->P = new ffm_long[prob->l+1];
        prob->Y = new ffm_float[prob->l];
	prob->Q = new ffm_long[nnq[q]+1];
        ffm_long p = 0;
        prob->P[0] = 0;
	prob->Q[0] = 0;
	int group = -1;
	int line_in_batch = 0;
	int line_in_group = 0;
	int group_count = 0;
	bool start = false;
        for(ffm_int ii = 0; fgets(line, kMaxLineSize, f) != nullptr; ii++)
        {
	    //if (ii%num_batch != q) 
	    //	continue;
	    //ffm_int i = ii / num_batch;

	    
            //printf("i=%d, has super: %d\n",i,hasSuper);
            char *y_char = strtok(line, " \t");
            ffm_float y;
	    if (*y_char=='x'){
		group++;
		    //if (group_count>0)
		if (start){
		     prob->Q[group_count+1] = line_in_group;
		     group_count++;
		   
		}
	  	y = -999;
	    }else{
		
	    	if (!regression && num_class == 2){
		    y = (atoi(y_char)>0)? 1.0f : -1.0f;
	    	}else{
		    y = atof(y_char);		
	    	}
	    }
	    if (q>0 && group<linesum[q-1])
                continue;
	    start = true;
	    if (group>=linesum[q])
		break;	    
	    //printf("line %d batch %d line_in_batch %d line_in_group %d group_count %d group %d\n", ii, q, line_in_batch, line_in_group, group_count, group);
	    int i = line_in_batch;
	    line_in_batch++;
	    line_in_group++;

            prob->Y[i] = y;
            for(; ; p++)
	    {
                char* field_char = strtok(nullptr,":");
                char* idx_char = strtok(nullptr,":");
                char* value_char = strtok(nullptr," \t");
                if(field_char == nullptr || *field_char == '\n')
                    break;
	    	ffm_int field = atoi(field_char);
            	ffm_int idx = atoi(idx_char);
            	ffm_float value = atof(value_char);

            	prob->m = max(prob->m, field+1);
            	prob->n = max(prob->n, idx+1);
            	//printf("p1=%d\n",p); 
            	prob->X[p].f = field;
            	prob->X[p].j = idx;
	    }
	    prob->P[i+1] = p;
	    if (!quiet && i%(prob->l/10) == 0){
	    	double duration = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
            	printf("load %d%% time %.1fs\n", i/(prob->l/10)*10, duration);  
	    }
	    //printf("line %d batch %d line_in_batch %d line_in_group %d group_count %d group %d\n", ii, q, line_in_batch, line_in_group, group_count, group);
	}
	prob->Q[group_count+1] = line_in_group;
	/*for(int xx =0; xx< prob->q+1; xx++){
	    printf("batch: %d Q[%d]=%d\n", q,xx,prob->Q[xx]);
	}
	for(int xx =0; xx< prob->l+1; xx++){
            printf("batch: %d P[%d]=%d\n", q,xx,prob->P[xx]);
        }*/

	if(!quiet){
            double duration = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
            printf("load time %.1fs\n", duration);
  	    printf("features: %d fields:%d\n",prob->n,prob->m);
    	}
	std::string fileNameBin = std::string(path)+"_"+std::to_string(q)+".b";
	//printf("before bin batch: %d\n", q);
	ffm_write_problem_to_bin(prob, fileNameBin.c_str());
	//printf("after bin batch: %d\n", q);
	ffm_destroy_problem(&prob);
	//printf("done batch: %d\n", q);
    }
    fclose(f); 
}      

void ffm_read_problem_to_batch(char const *path, int num_batch, ffm_int num_class, bool regression, bool quiet)
{
    if(strlen(path) == 0)
        return ;
    std::clock_t startTime = std::clock();
    
    printf("load raw: %s\n", path);
    FILE *f = fopen(path, "r");
    if(f == nullptr)
        return ;

    

    char line[kMaxLineSize];
    int lines = 0;
    ffm_long* nnz = new ffm_long[num_batch];
    ffm_int* line_ = new ffm_int[num_batch];
    ffm_long* nnq = new ffm_long[num_batch];

    for(ffm_int i = 0; i < num_batch; i++){
        nnz[i] = 0;
	line_[i] = 0;
	nnq[i] = 0;
    }
    
    int group = 0;

    for(; fgets(line, kMaxLineSize, f) != nullptr; lines++)
    {
        char *mark = strtok(line, " \t");
	if (*mark == 'x'){
	    nnq[group%num_batch]++;
	    group++;
	    
	}else{

	}
	line_[(group-1)%num_batch]++;
        for(; ; nnz[(group-1)%num_batch]++)
        {
            char *ptr = strtok(nullptr," \t");
            if(ptr == nullptr || *ptr == '\n')
                break;
        }
    }

    /*for (ffm_int q = 0; q < num_batch; q++){
	printf("batch %d: groups:%d items:%d \n", q, nnq[q], nnz[q]);
    }*/ 
    // start to do here next !!!
    for (ffm_int q = 0; q < num_batch; q++){
        rewind(f);
	ffm_problem *prob = new ffm_problem;
        prob->l = line_[q];
        prob->n = 0;
        prob->m = 0;
        prob->nx = nnz[q];
	prob->q = nnq[q];
        prob->X = new ffm_node[nnz[q]];
        prob->P = new ffm_long[prob->l+1];
        prob->Y = new ffm_float[prob->l];
	prob->Q = new ffm_long[nnq[q]+1];
        ffm_long p = 0;
        prob->P[0] = 0;
	prob->Q[0] = 0;
	int group = -1;
	int line_in_batch = 0;
	int line_in_group = 0;
	int group_count = 0;
        for(ffm_int ii = 0; fgets(line, kMaxLineSize, f) != nullptr; ii++)
        {
	    //if (ii%num_batch != q) 
	    //	continue;
	    //ffm_int i = ii / num_batch;

	    
            //printf("i=%d, has super: %d\n",i,hasSuper);
            char *y_char = strtok(line, " \t");
            ffm_float y;
	    if (*y_char=='x'){
		group++;
		if (group%num_batch == q){
		    if (group_count>0)
			prob->Q[group_count] = line_in_group;
		    group_count++;
		}
	  	y = -999;
	    }else{
		
	    	if (!regression && num_class == 2){
		    y = (atoi(y_char)>0)? 1.0f : -1.0f;
	    	}else{
		    y = atof(y_char);		
	    	}
	    }

	    if (group%num_batch != q)
		continue;	    
	    //printf("line %d batch %d line_in_batch %d line_in_group %d group_count %d group %d\n", ii, q, line_in_batch, line_in_group, group_count, group);
	    int i = line_in_batch;
	    line_in_batch++;
	    line_in_group++;

            prob->Y[i] = y;
            for(; ; p++)
	    {
                char* field_char = strtok(nullptr,":");
                char* idx_char = strtok(nullptr,":");
                char* value_char = strtok(nullptr," \t");
                if(field_char == nullptr || *field_char == '\n')
                    break;
	    	ffm_int field = atoi(field_char);
            	ffm_int idx = atoi(idx_char);
            	ffm_float value = atof(value_char);

            	prob->m = max(prob->m, field+1);
            	prob->n = max(prob->n, idx+1);
            	//printf("p1=%d\n",p); 
            	prob->X[p].f = field;
            	prob->X[p].j = idx;
	    }
	    prob->P[i+1] = p;
	    if (!quiet && i%(prob->l/10) == 0){
	    	double duration = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
            	printf("load %d%% time %.1fs\n", i/(prob->l/10)*10, duration);  
	    }
	    //printf("line %d batch %d line_in_batch %d line_in_group %d group_count %d group %d\n", ii, q, line_in_batch, line_in_group, group_count, group);
	}
	prob->Q[group_count] = line_in_group;
	/*for(int xx =0; xx< prob->q+1; xx++){
	    printf("batch: %d Q[%d]=%d\n", q,xx,prob->Q[xx]);
	}
	for(int xx =0; xx< prob->l+1; xx++){
            printf("batch: %d P[%d]=%d\n", q,xx,prob->P[xx]);
        }*/

	if(!quiet){
            double duration = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
            printf("load time %.1fs\n", duration);
  	    printf("features: %d fields:%d\n",prob->n,prob->m);
    	}
	std::string fileNameBin = std::string(path)+"_"+std::to_string(q)+".b";
	//printf("before bin batch: %d\n", q);
	ffm_write_problem_to_bin(prob, fileNameBin.c_str());
	//printf("after bin batch: %d\n", q);
	ffm_destroy_problem(&prob);
	//printf("done batch: %d\n", q);
    }
    fclose(f); 
}    

    

	    
       

   
ffm_problem* ffm_read_problem(char const *path, ffm_int num_class, bool regression, bool quiet)
{
    if(strlen(path) == 0)
        return nullptr;
    std::clock_t startTime = std::clock();
    std::string fileNameBin = std::string(path)+".b";
    if (FileUtil::fileExist(fileNameBin)){
	if (!quiet)
	    printf("load bin: %s\n", fileNameBin.c_str());
	ffm_problem* problem = ffm_read_problem_from_bin(fileNameBin.c_str());	
        double duration = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
	if (!quiet){
            printf("load time %.1fs\n", duration);
            printf("features: %d fields:%d\n",problem->n,problem->m);
	}
        return problem; 
    }
    printf("load raw: %s\n", path);
    FILE *f = fopen(path, "r");
    if(f == nullptr)
        return nullptr;

    ffm_problem *prob = new ffm_problem;
    prob->l = 0;
    prob->n = 0;
    prob->m = 0;
    prob->nx = 0;
    prob->X = nullptr;
    prob->P = nullptr;
    prob->Y = nullptr;

    char line[kMaxLineSize];

    ffm_long nnz = 0;
    for(; fgets(line, kMaxLineSize, f) != nullptr; prob->l++)
    {
        strtok(line, " \t");
        for(; ; nnz++)
        {
            char *ptr = strtok(nullptr," \t");
            if(ptr == nullptr || *ptr == '\n')
                break;
        }
    }
    if (!quiet)
	printf("lines: %d X size: %lld\n", prob->l, nnz);
    rewind(f);
    prob->nx = nnz;
    prob->X = new ffm_node[nnz];
    prob->P = new ffm_long[prob->l+1];
    prob->Y = new ffm_float[prob->l];

    ffm_long p = 0;
    prob->P[0] = 0;
    for(ffm_int i = 0; fgets(line, kMaxLineSize, f) != nullptr; i++)
    {
	//vector<string> tokens = StringUtil::split(string(line)," ");
        bool hasSuper = false;//StringUtil::split(tokens[1],":").size() == 4;
        //printf("i=%d, has super: %d\n",i,hasSuper);
        char *y_char = strtok(line, " \t");
        ffm_float y;
		
	if (!regression && num_class == 2){
		y = (atoi(y_char)>0)? 1.0f : -1.0f;
	}else{
		y = atof(y_char);		
	}
        //printf("regression:%d num_class:%d y=%f\n",regression, num_class, y);
        prob->Y[i] = y;
        for(; ; p++)
        {
	    //printf("p0=%d\n",p);
	    ffm_int sfield;
            char* sfield_char;
            if (hasSuper){
	    	sfield_char = strtok(nullptr,":");
            }else{
                sfield = -1;
	    }

	    char* field_char = strtok(nullptr,":");
            char* idx_char = strtok(nullptr,":");
            char* value_char = strtok(nullptr," \t");
            if (!hasSuper){
                if(field_char == nullptr || *field_char == '\n')
                    break;
 	    }else{
		if(sfield_char == nullptr || *sfield_char == '\n')
                    break;
	    }
	    if (hasSuper) sfield = atoi(sfield_char);	
	    ffm_int field = atoi(field_char);
            ffm_int idx = atoi(idx_char);
            ffm_float value = atof(value_char);

            prob->m = max(prob->m, field+1);
            prob->n = max(prob->n, idx+1);
            //printf("p1=%d\n",p); 
            prob->X[p].f = field;
            prob->X[p].j = idx;
            //prob->X[p].v = value;
            //printf("p=%d\n",p);
        }
        prob->P[i+1] = p;
	if (!quiet && i%(prob->l/10) == 0){
	    double duration = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
            printf("load %d%% time %.1fs\n", i/(prob->l/10)*10, duration);  
	}
    }

    fclose(f);
    if(!quiet){
        double duration = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
        printf("load time %.1fs\n", duration);
  	printf("features: %d fields:%d\n",prob->n,prob->m);
    }
    ffm_write_problem_to_bin(prob, fileNameBin.c_str()); 
    return prob;
}

int ffm_read_problem_to_disk(char const *txt_path, char const *bin_path)
{
    FILE *f_txt = fopen(txt_path, "r");
    if(f_txt == nullptr)
        return 1;

    FILE *f_bin = fopen(bin_path, "wb");
    if(f_bin == nullptr)
        return 1;

    vector<char> line(kMaxLineSize);

    ffm_int m = 0;
    ffm_int n = 0;
    ffm_int max_l = 0;
    ffm_long max_nnz = 0;
    ffm_long p = 0;

    vector<ffm_float> Y;
    vector<ffm_float> R;
    vector<ffm_long> P(1, 0);
    vector<ffm_node> X;

    auto write_chunk = [&] ()
    {
        ffm_int l = Y.size();
        ffm_long nnz = P[l];

        max_l = max(max_l, l);
        max_nnz = max(max_nnz, nnz);

        fwrite(&l, sizeof(ffm_int), 1, f_bin);
        fwrite(Y.data(), sizeof(ffm_float), l, f_bin);
        fwrite(R.data(), sizeof(ffm_float), l, f_bin);
        fwrite(P.data(), sizeof(ffm_long), l+1, f_bin);
        fwrite(X.data(), sizeof(ffm_node), nnz, f_bin);

        Y.clear();
        R.clear();
        P.assign(1, 0);
        X.clear();
        p = 0;
    };

    fwrite(&m, sizeof(ffm_int), 1, f_bin);
    fwrite(&n, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_l, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_nnz, sizeof(ffm_long), 1, f_bin);

    while(fgets(line.data(), kMaxLineSize, f_txt))
    {
        char *y_char = strtok(line.data(), " \t");

        ffm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;

        ffm_float scale = 0;
        for(; ; p++)
        {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_node N;
            N.f = atoi(field_char);
            N.j = atoi(idx_char);
            //N.v = atof(value_char);

            X.push_back(N);

            m = max(m, N.f+1);
            n = max(n, N.j+1);

            scale += 1;//N.v*N.v;
        }
        scale = 1/scale;

        Y.push_back(y);
        R.push_back(scale);
        P.push_back(p);

        if(X.size() > (size_t)kCHUNK_SIZE)
            write_chunk(); 
    }
    write_chunk(); 
    write_chunk(); 

    rewind(f_bin);
    fwrite(&m, sizeof(ffm_int), 1, f_bin);
    fwrite(&n, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_l, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_nnz, sizeof(ffm_long), 1, f_bin);

    fclose(f_bin);
    fclose(f_txt);

    return 0;
}

void ffm_destroy_problem(ffm_problem **prob)
{
    if(prob == nullptr || *prob == nullptr)
        return;
    delete[] (*prob)->X;
    //printf("X destroyed\n");
    delete[] (*prob)->P;
    //printf("P destroyed\n");
    delete[] (*prob)->Y;
    //printf("Y destroyed\n");
    delete[] (*prob)->Q;
    //printf("Q destroyed\n");
    delete *prob;
    *prob = nullptr;
}

ffm_int ffm_save_model(ffm_model *model, char const *path)
{
    ofstream f_out(path);
    if(!f_out.is_open())
        return 1;

    f_out << "n " << model->n << "\n";
    f_out << "m " << model->m << "\n";
    f_out << "k " << model->k << "\n";
    f_out << "normalization " << model->normalization << "\n";

    ffm_float *ptr = model->W;
    for(ffm_int j = 0; j < model->n; j++)
    {
        for(ffm_int f = 0; f < model->m; f++)
        {
            f_out << "w" << j << "," << f << " ";
            for(ffm_int d = 0; d < model->k; d++, ptr++)
                f_out << *ptr << " ";
            f_out << "\n";
        }
    }

    return 0;
}

ffm_model* ffm_load_model(char const *path)
{
    ifstream f_in(path);
    if(!f_in.is_open())
        return nullptr;

    string dummy;

    ffm_model *model = new ffm_model;
    model->W = nullptr;

    f_in >> dummy >> model->n >> dummy >> model->m >> dummy >> model->k 
         >> dummy >> model->normalization;

    try
    {
        model->W = malloc_aligned_float((ffm_long)model->m*model->n*model->k);
    }
    catch(bad_alloc const &e)
    {
        ffm_destroy_model(&model);
        return nullptr;
    }

    ffm_float *ptr = model->W;
    for(ffm_int j = 0; j < model->n; j++)
    {
        for(ffm_int f = 0; f < model->m; f++)
        {
            f_in >> dummy;
            for(ffm_int d = 0; d < model->k; d++, ptr++)
                f_in >> *ptr;
        }
    }

    return model;
}

void ffm_destroy_model(ffm_model **model)
{
    if(model == nullptr || *model == nullptr)
        return;
#ifdef _WIN32
    _aligned_free((*model)->W);
#else
    free((*model)->W);
#endif
    delete *model;
    *model = nullptr;
}

ffm_parameter ffm_get_default_param()
{
    ffm_parameter param;
    param.apk = 7;
    param.flip = false;
    param.savefield = 10000;
    param.margin = -100; // -100 means no margin
    param.auc = 0;
    param.useinter = 0;
    param.mcc = 0;
    param.unseen = 0;
    param.eta = 0.2;
    param.decay = 0; // 0 means no decay
    param.lambda = 0.00002;
    param.mu = 0;
    param.nr_iters = 15;
    param.k = 4;
    param.nr_threads = 1;
    param.quiet = false;
    param.normalization = true;
    param.random = true;
    param.auto_stop = false;
    param.regression = false;
    param.warm_start = false;
    param.solver = "adagrad";
    param.num_classes = 2;  // by default, do binary classification
    return param;
}

void ffm_train_batch(Option opt, ffm_problem *va, int* fieldmap, int num_fea)
{
    shared_ptr<ffm_model> model =
            shared_ptr<ffm_model>(init_model(va->n, opt.param.savefield, opt.param),
                [] (ffm_model *ptr) { ffm_destroy_model(&ptr); });
    printf("model initiated, m=%d, n=%d\n", model->m, model->n);
    int num_batch = 10;
    float bestloss = 100;
    vector<ffm_int> order_big(num_batch);
    for(ffm_int i = 0; i < num_batch; i++)
         order_big[i] = i;
    
    int iters = opt.param.nr_iters;
    opt.param.nr_iters = 1;
    float floss;
    for(ffm_int iter = 1; iter <= iters; iter++){

	random_shuffle(order_big.begin(), order_big.end());

    	for (ffm_int q = 0; q < num_batch; q++){
    	    ffm_problem *tr = read_batch(opt.tr_path.c_str(), order_big[q], num_batch, false, opt.param.num_classes, opt.param.regression, true);
	    //printf("load tr \n");
            vector<ffm_int> order(tr->q);
            for(ffm_int i = 0; i < tr->q; i++)
               order[i] = i;
	    //printf("start to train \n");

    	    floss = train_b(model, (q==0)&&(iter == 1), (q==num_batch-1), iter, tr, order, opt.param, fieldmap, num_fea, va);
	    ffm_destroy_problem(&tr);
	}
	opt.param.eta = opt.param.eta * exp(-opt.param.decay*iter);
	if (opt.param.auto_stop && floss>bestloss){
                printf("early stopping, bestloss %f current loss %f\n",bestloss,floss);
                break;
        }
        bestloss = floss;
    }
    ffm_int nc = opt.param.num_classes;
    //if (nc==2) nc=1;
    //shrink_model(*model, opt.param.k, nc);

    ffm_model *model_ret = new ffm_model;

    model_ret->n = model->n;
    model_ret->m = model->m;
    model_ret->k = model->k;
    model_ret->normalization = model->normalization;

    model_ret->W = model->W;
    model->W = nullptr;

    mypredict_batch(opt, model_ret, fieldmap, num_fea);
    ffm_destroy_model(&model_ret); 

}






ffm_problem* ffm_read_problem_from_bin(char const *path){

	ffm_problem *prob = new ffm_problem;
	prob->l = 0;
    	prob->n = 0;
    	prob->m = 0;
   	prob->nx = 0;
    	prob->X = nullptr;
    	prob->P = nullptr;
    	prob->Y = nullptr;

	FILE *fp = fopen(path, "rb");
        if (fp != NULL)
        {
                prob->n = FileUtil::readNumber<ffm_int>(fp);
                prob->l = FileUtil::readNumber<ffm_int>(fp);
                prob->m = FileUtil::readNumber<ffm_int>(fp);
                prob->nx = FileUtil::readNumber<ffm_long>(fp);
	        prob->q = FileUtil::readNumber<ffm_long>(fp);

		prob->X = new ffm_node[prob->nx];
  		prob->P = new ffm_long[prob->l+1];
    		prob->Y = new ffm_float[prob->l];
		prob->Q = new ffm_long[prob->q+1];
		
                fread(prob->X, sizeof(ffm_node), prob->nx, fp);
                fread(prob->P, sizeof(ffm_long), prob->l+1, fp);
                fread(prob->Y, sizeof(ffm_float), prob->l, fp);
		fread(prob->Q, sizeof(ffm_long), prob->q+1, fp);
                fclose(fp);
        }

	return prob;

}
void ffm_write_problem_to_bin(const ffm_problem* prob, char const *path){

	FILE *fp = fopen(path, "wb");
	if (fp != NULL)
        {
                FileUtil::writeNumber<ffm_int>(fp, prob->n);
		FileUtil::writeNumber<ffm_int>(fp, prob->l);
		FileUtil::writeNumber<ffm_int>(fp, prob->m);
		FileUtil::writeNumber<ffm_long>(fp, prob->nx);
	        FileUtil::writeNumber<ffm_long>(fp, prob->q);
                fwrite(prob->X, sizeof(ffm_node), prob->nx, fp);
                fwrite(prob->P, sizeof(ffm_long), prob->l+1, fp);
		fwrite(prob->Y, sizeof(ffm_float), prob->l, fp);
		fwrite(prob->Q, sizeof(ffm_long), prob->q+1, fp);
                fclose(fp);
        }
}

void print_problem(ffm_problem* prob){
    printf("***********************\n");
    printf("n:%d m:%d l:%d nx:%d q:%d \n", prob->n, prob->m, prob->l, prob->nx, prob->q);
    for(int xx =0; xx< prob->q+1; xx++){
            printf("Q[%d]=%d\n", xx,prob->Q[xx]);
    }

    for(int i=0; i<prob->q; i++){
	printf("group %d\n", i);
	for (int j=prob->Q[i]; j<prob->Q[i+1]; j++){
	    printf("y:%f ", prob->Y[j]);
	    for (int k=prob->P[j]; k<prob->P[j+1]; k++)
		printf("%d:%d:1 ", prob->X[k].f, prob->X[k].j);
	    printf("\n");
	}
	    
    } 
    printf("***********************\n\n");

}

} // namespace ffm
