./ffm-train-predict --auto-stop --hasLabel -savefield 19 --trainbatch -s 10 -fmap feamap.txt -t 10 -r 0.5 -p sample.va -test sample.va -out sample.out sample.tr cv-model


#--auto-stop: like xgb's early stop
#--regression: without it, ffm will do binary classification
#--haslabel: include true labels in prediction, only for cv
#-l: lambda*l2norm
#-k: lenght of latent vector, this will affect memory usage!
#-t: number of iterations
#-decay: learning rate decay, exp(-decay*iteration), 0 will be no decay
#-r: learning rate
#-s: number of threads
#-p: path of holdout data, which should have true labels with it
#-test: path of test data
#-out: path of output predictions
# the last two parameter are fixed, which are train_path, dump_model_path
