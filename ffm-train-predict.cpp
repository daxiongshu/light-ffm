#pragma GCC diagnostic ignored "-Wunused-result" 
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <cassert>
#include "ffm.h"

#include <fstream>
#include <iomanip>
#include <memory>
#include <cmath>
#include <map>
#include "StringUtil.h"
using namespace std;
using namespace ffm;





   
int tField(string filename){
    int const kMaxLineSize = 1000000;
    char line[kMaxLineSize];
    ffm_int i = 0;
    FILE *f_in = fopen(filename.c_str(), "r");
    for(; fgets(line, kMaxLineSize, f_in) != nullptr; i++){
    }
    fclose(f_in);
    return i;
}

int* read_fieldmap(string filename, int num_field){

    int* result = nullptr;
    FILE *f_in = fopen(filename.c_str(), "r");
    int const kMaxLineSize = 1000000;
    char line[kMaxLineSize];
    ffm_int i = 0;

    result = new int[num_field*num_field];
    for(int i=0;i<num_field;i++)
    	for(int j=0;j<num_field;j++)
            result[i*num_field+j] = j;
    if (filename=="")
	return result;

    for(; fgets(line, kMaxLineSize, f_in) != nullptr; i++)
    {
        char *f_char = strtok(line, ":");
        int field = atoi(f_char);
	int c = 0;
        while(true)
        {
            char *value_char = strtok(nullptr," \t");
            if(value_char == nullptr || *value_char == '\n')
                break;
            int v = atoi(value_char);
            result[field*num_field+c] = v;
	    c += 1;
        }
    }
    fclose(f_in);
    return result;
}



void printInteraction(int* interaction, int num_field){
    for(int j=0;j<num_field;j++){
	for(int k=0;k<num_field;k++){
	    printf("%d ",interaction[j*num_field+k]);
	}
	printf("\n");
    }
}

void printInteraction(map<int, map<int,int> > mymap){

    for (map<int, map<int,int> >::iterator it=mymap.begin(); it!=mymap.end(); ++it){
	printf("%d: ",it->first);
	for (map<int,int>::iterator itx=it->second.begin(); itx!=it->second.end(); ++itx){
	    printf("%d ",itx->first);
	}
        printf("\n");
    }
}
string train_help()
{
    return string(
"usage: ffm-train [options] training_set_file [model_file]\n"
"\n"
"options:\n"
"-l <lambda>: set regularization parameter (default 0.00002)\n"
"-k <factor>: set number of latent factors (default 4)\n"
"-t <iteration>: set number of iterations (default 15)\n"
"-r <eta>: set learning rate (default 0.2)\n"
"-s <nr_threads>: set number of threads (default 1)\n"
"-p <path>: set path to the validation set\n"
"-v <fold>: set the number of folds for cross-validation\n"
"--quiet: quiet model (no output)\n"
"--no-norm: disable instance-wise normalization\n"
"--warm-start: train a pre-trained model\n"
"--no-rand: disable random update\n"
"--on-disk: perform on-disk training (a temporary file <training_set_file>.bin will be generated)\n"
"--regression: do regression instead of binary classification\n"
"--auto-stop: stop at the iteration that achieves the best validation loss (must be used with -p)\n");
}


string basename(string path)
{
    const char *ptr = strrchr(&*path.begin(), '/');
    if(!ptr)
        ptr = path.c_str();
    else
        ptr++;
    return string(ptr);
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option opt;
    opt.fmap_path = "";
    opt.trainbatch = false;
    ffm_int i = 1;
    for(; i < argc; i++)
    {
        if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of iterations after -t");
            i++;
            opt.param.nr_iters = atoi(args[i].c_str());
            if(opt.param.nr_iters <= 0)
                throw invalid_argument("number of iterations should be greater than zero");
        }
        else if(args[i].compare("-k") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of factors after -k");
            i++;
            opt.param.k = atoi(args[i].c_str());
            if(opt.param.k <= 0)
                throw invalid_argument("number of factors should be greater than zero");
        }
        else if(args[i].compare("-r") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify eta after -r");
            i++;
            opt.param.eta = atof(args[i].c_str());
            if(opt.param.eta <= 0)
                throw invalid_argument("learning rate should be greater than zero");
        }
	else if(args[i].compare("-decay") == 0){
	    i++;
	    opt.param.decay = atof(args[i].c_str());
	}
	else if(args[i].compare("-apk") == 0){
            i++;
            opt.param.apk = atoi(args[i].c_str());
        }

	else if(args[i].compare("-margin") == 0){
            i++;
            opt.param.margin = atof(args[i].c_str());
        }

	else if(args[i].compare("-savefield") == 0){
            i++;
            opt.param.savefield = atoi(args[i].c_str());
        }
	else if(args[i].compare("--useinter") == 0){
            opt.param.useinter = true;//atoi(args[i].c_str());
        }
       	else if(args[i].compare("--flip") == 0){
            opt.param.flip = true;//atoi(args[i].c_str());
        }

	else if(args[i].compare("-mu") == 0){
            i++;
            opt.param.mu = atof(args[i].c_str());
        }
  	else if(args[i].compare("-num_classes") == 0){
            i++;
            opt.param.num_classes = atoi(args[i].c_str());
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify lambda after -l");
            i++;
            opt.param.lambda = atof(args[i].c_str());
            if(opt.param.lambda < 0)
                throw invalid_argument("regularization cost should not be smaller than zero");
        }
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of threads after -s");
            i++;
            opt.param.nr_threads = atoi(args[i].c_str());
            if(opt.param.nr_threads <= 0)
                throw invalid_argument("number of threads should be greater than zero");
        }
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of folds after -v");
            i++;
            opt.nr_folds = atoi(args[i].c_str());
            if(opt.nr_folds <= 1)
                throw invalid_argument("number of folds should be greater than one");
            opt.do_cv = true;
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;
            opt.va_path = args[i];
        }

	else if(args[i].compare("-fmap") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;
            opt.fmap_path = args[i];
        }

 
	else if(args[i].compare("-test") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify test path after -test");
            i++;
            opt.test_path = args[i];
        }

	else if(args[i].compare("-out") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify output path after -test");
            i++;
            opt.output_path = args[i];
        }	
	else if(args[i].compare("-solver") == 0){
	    i++;
	    opt.param.solver = args[i];
	}

        else if(args[i].compare("--no-norm") == 0)
        {
            opt.param.normalization = false;
        }
        else if(args[i].compare("--quiet") == 0)
        {
            opt.param.quiet = true;
        }
	else if(args[i].compare("--trainbatch") == 0)
        {
            opt.trainbatch = true;
        }

	else if(args[i].compare("--auc") == 0)
        {
            opt.param.auc = true;
        }
	else if(args[i].compare("--unseen") == 0)
        {
            opt.param.unseen = true;
        }

        else if(args[i].compare("--mcc") == 0)
        {
            opt.param.mcc = true;
        }

   	else if (args[i].compare("--warm-start")==0){
	    opt.param.warm_start = true;
	}	
	else if(args[i].compare("--regression") == 0)
        {
            opt.param.regression = true;
        }
	else if(args[i].compare("--hasLabel") == 0)
        {
            opt.hasLabel = true;
        }
        else if(args[i].compare("--no-rand") == 0)
        {
            opt.param.random = false;
        }
        else if(args[i].compare("--on-disk") == 0)
        {
            opt.on_disk = true;
        }
        else if(args[i].compare("--auto-stop") == 0)
        {
            opt.param.auto_stop = true;
        }
        else
        {
            break;
        }
    }

    if(i != argc-2 && i != argc-1)
        throw invalid_argument("cannot parse command\n");

    opt.tr_path = args[i];
    i++;

    if(i < argc)
    {
        opt.model_path = string(args[i]);
	opt.param.model_path = opt.model_path;
    }
    else if(i == argc)
    {
        opt.model_path = basename(opt.tr_path) + ".model";
	opt.param.model_path = opt.model_path;
    }
    else
    {
        throw invalid_argument("cannot parse argument");
    }
    if (opt.param.solver=="momentum") assert(opt.param.mu>0);
    return opt;
}
void train_batch(Option opt, int* feamap, int num_fea)
{
    printf("num_classes:%d\n",opt.param.num_classes);
 
    ffm_problem *va = nullptr;
    if(!opt.va_path.empty())
    {
        //va = ffm_read_problem(opt.va_path.c_str(), opt.param.num_classes, opt.param.regression, opt.param.quiet);
	va = read_batch(opt.va_path.c_str(), 0, 1, true, opt.param.num_classes, opt.param.regression, true);
        if(va == nullptr)
        {
            cerr << "cannot load " << opt.va_path << endl << flush;
            //return nullptr;
        }
    }

    ffm_train_batch(opt, va, feamap, num_fea);

    ffm_destroy_problem(&va);

}

int main(int argc, char **argv)
{
    Option opt;
    try
    {
        opt = parse_option(argc, argv);
    }
    catch(invalid_argument &e)
    {
        cout << e.what() << endl;
        return 1;
    }
    //printInteraction(result);   
    int num_fea = tField(opt.fmap_path);

    int* feamap = nullptr;
    feamap = read_fieldmap(opt.fmap_path, num_fea);
    
    printInteraction(feamap,num_fea); 
    {
        train_batch(opt, feamap, num_fea);

        return 0;//ffm_save_model(model, opt.model_path.c_str());;
    }
    /*for(int i=0;i<10;i++){
    	ffm_problem *tr = read_batch("sample.tr", i, 10, false, 2, false, true);
	print_problem(tr);	
    }*/
}
