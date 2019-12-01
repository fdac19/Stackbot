import re
from scipy import sparse
import gensim
from tqdm import tqdm_notebook as tqdm
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import TweetTokenizer
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from p_tqdm import p_map
from pathos.multiprocessing import ProcessingPool as Pool

warnings.filterwarnings("ignore", category=Warning)

class doc2vec:
    
    # tokenizes a string
    # Parameters: document is a string
    # Returns: the parsed string
    def tokenization(self, document):
        return re.findall(self.w, document)

    # initializer for the class
    # Parameters: df the pandas dataframe that will be manipulated
    #             X is the name of the collumn in df that holds the text to be trained on
    #             Y is a list of names of collumns in the df that hold the values of the true ansers
    # Returns: None
    def __init__(self, df, X, Y, build=False, given=None, epoch=50, vector=300, window=15):
        
        for one in Y:
            df[one] = df[one].map(lambda x: str(float(x)))
        
        self.w = re.compile("\w+", re.I)
        if 'basestring' not in globals():
            basestring = str
        self.tknzr = re.compile("\w+").findall
        # Hyperparameters : https://arxiv.org/pdf/1607.05368.pdf
        self.vector_size = vector
        self.window_size = window
        self.min_count = 2
        self.sampling_threshold = 1e-4
        self.negative_size = 5
        self.train_epoch = epoch
        self.dm = 0
        self.worker_count = 7
        self.build = build
        self.given = given
    
        labeled_sentences = []
        df_tags = []

        if isinstance(Y, basestring):
            df_tags.append(Y)
        elif isinstance(Y, list):
            df_tags = Y
        elif not isinstance(Y, list):
            raise TypeError
        self.df = df
        self.x = X
        self.y = Y
        self.df_tags = df_tags
        self.testseries = df[df_tags[0]].unique()
        self.testseries_name = df_tags[0]

        if build == True:
            if given is None:
                for index, datapoint in df.iterrows():
                    tokenized_words = self.tokenization(datapoint[X])
                    labeled_sentences.append(TaggedDocument(words=tokenized_words, tags=[datapoint[i] for i in df_tags]))
                model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vector_size,
                                                      window_size=self.window_size,
                                                      min_count=self.min_count,
                                                      sampling_threshold=self.sampling_threshold,
                                                      negative_size=self.negative_size,
                                                      train_epoch=self.train_epoch,
                                                      dm=self.dm,
                                                      worker_count=self.worker_count)
           
                model.build_vocab(labeled_sentences)
                model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
                self.model = model
            else:
                self.model = given
        
    # iters is the function called to allow multiprocessing with p_map
    # Parameters: X is the name of the collumn in df that holds the text to be trained on
    #             total__label_accuracy is a pandas df that has a column filled with the labels and an empty column for F1 score
    #             col is the column we are training on in the main df
    #             oversample is 0 if you do not want to oversample and is set to a number you want all tags to be oversampled to
    def iters( self, X, total_label_accuracy, labeled_sentences, col, oversample):               
        train, test = train_test_split(self.df, shuffle=True, test_size=0.05)   
            
        if oversample > 0:
            max_size = oversample
            train_over = [train]
            grpy_obj = train.groupby(col)
            for class_index, group in grpy_obj:
                train_over.append(group.sample(max_size-len(group), replace=True))
            train = pd.concat(train_over)
           
        for index, datapoint in train.iterrows():    
            tokenized_words = self.tokenization(datapoint[X])
            labeled_sentences.append(TaggedDocument(words=tokenized_words, tags=[datapoint[i] for i in self.df_tags]))

        model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vector_size,
                                              window_size=self.window_size,
                                              min_count=self.min_count,
                                              sampling_threshold=self.sampling_threshold,
                                              negative_size=self.negative_size,
                                              train_epoch=self.train_epoch,
                                              dm=self.dm,
                                              worker_count=self.worker_count)
       
        model.build_vocab(labeled_sentences)
        model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model                 
        test['results'] = self.predict(test[X])     
        test_iter = []
        test_iter_before = list(test[col])
        for i in test_iter_before:
            i = float(i)
            i = str(i)
            if i.lower() == 'nan':
                continue
            test_iter.append(i)

        train_iter = []
        train_iter_before = list(test['results'])
        for i in train_iter_before:
            i = float(i)
            i = str(i)
            if i.lower() == 'nan':
                continue
            train_iter.append(i)
        y_true = np.array(test_iter)
        y_pred = np.array(train_iter)

        label = self.class_maker(y_true, y_pred, total_label_accuracy['Tag'])
        labelaccuracy = f1_score(test[self.testseries_name], test['results'], labels=label, average=None)
        recall_l = recall_score(test[self.testseries_name], test['results'], labels=label, average=None)
        precision_l = precision_score(test[self.testseries_name], test['results'], labels=label, average=None)
        num = 0
        for i in labelaccuracy:
            total_label_accuracy.at[num, 'F1 Score'] = i
            num += 1
        num = 0
        for i in recall_l:
            total_label_accuracy.at[num, 'Recall Score'] = i
            num += 1
        num = 0
        for i in precision_l:
            total_label_accuracy.at[num, 'Precision Score'] = i
            num += 1
        num = 0
        for i in label: 
            total_label_accuracy.at[num, 'Tag'] = i
            num += 1
        accuracy = accuracy_score(test[self.testseries_name], test['results'])
        
        return accuracy, total_label_accuracy, train, test, model
    
    # score trains, tests and saves the model
    # Parameters: verbose allows you to print accuracies and confusion matrices if desired
    #             iterations is the number of separate models you want to be created to try and find the best one
    #             oversample is set to 0 if you want no oversampling and set to the number you want to oversample each label to
    # Returns: list of the df for total_label_accuracy and accuracy
    def score(self, verbose=True, iterations=10, oversample=0):
        self.best_accuracy = 0
        df = self.df
        X = self.x
        Y = self.y
        self.verbose = verbose
        
        if 'basestring' not in globals():
            basestring = str
        
        labeled_sentences = []
        df_tags = []
        
        if isinstance(Y, basestring):
            df_tags.append(Y)
        elif isinstance(Y, list):
            df_tags = Y
        elif not isinstance(Y, list):
            raise TypeError
        
        first = True
        for col in self.df_tags:
            print(col)
            total_accuracy = 0
            total_label_accuracy = pd.DataFrame( columns=[ 'Tag', 'F1 Score'])
            
            num = 0
            for i in df[col].unique():
                total_label_accuracy.at[num, 'Tag'] = i
                num += 1
             
            oversample_l = [oversample] * iterations
            xs = [X] * iterations
            ls = [labeled_sentences] * iterations
            ts = [total_label_accuracy] * iterations
            l = p_map(self.iters, xs, ts, ls, col, oversample_l)

            best_accuracy = 0
            for i in l:
                if best_accuracy < i[0]:
                    best_accuracy = i[0]
                    accuracy = i[0]
                    total_label_accuracy = i[1]
                    train = i[2]
                    test = i[3]
                    model = i[4]
                    self.model = model

            if first:
                if verbose==True: print('Best Accuracy: ' + str(accuracy))
                for i in total_label_accuracy.index:
                    total_label_accuracy.at[i, 'F1 Score'] = total_label_accuracy.at[i, 'F1 Score']

                if verbose==True: print("Labeled F1 Score: ")
                if verbose==True: print(total_label_accuracy.to_string(index=False))
                if verbose==True: print()
           
            np.set_printoptions(precision=2)
            
            class_list = []
            for i in total_label_accuracy.index:
                class_list.append(str(float(total_label_accuracy.at[i, 'Tag'])))
            class_list = np.array(class_list)            
            class_list.sort()
            
            test_final_v1 = []
            test_final = list(test[col])
            for i in test_final:
                i = float(i)
                i = str(i)
                if i.lower() == 'nan':
                    continue
                test_final_v1.append(i)

            train_final_v1 = []
            train_final = list(test['results'])
            for i in train_final:
                i = float(i)
                i = str(i)
                if i.lower() == 'nan':
                    continue
                train_final_v1.append(i)
            
            test_final_v1 = np.array(test_final_v1)
            train_final_v1 = np.array(train_final_v1)
   
            if first:
                # Plot non-normalized confusion matrix
                if verbose==True: 
                    self.plot_confusion_matrix(test_final_v1, train_final_v1, classes=class_list, total_label_accuracy=total_label_accuracy, title='Confusion matrix, without normalization')

                # Plot normalized confusion matrix
                if verbose==True: 
                    self.plot_confusion_matrix(test_final_v1, train_final_v1, classes=class_list, total_label_accuracy=total_label_accuracy, normalize=True, title='Normalized confusion matrix')

                if verbose==True: plt.show()
                first = False

        return [total_label_accuracy, accuracy]

    # takes in a taged document and infers vector and returns whether it is releveant or not (1 or 0)
    def predict_taggedtext(self, document):  
        inferred_vector = document
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        return sims

    # takes in a string and infers vector and returns vectors and distance
    def predict_text(self, document):  
        tokenized_words = self.tokenization(document)
        inferred_vector = TaggedDocument(words=tokenized_words, tags=["inferred_vector"])[0]
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        tags = []
        for col in self.df_tags:
            tags.append([rec for rec in sims if rec[0] in set(self.df[col].unique())][0][0])
        return tags
    
    # takes in a string and infers vector and returns vectors and distance
    def predict_sims(self, document):  
        tokenized_words = self.tokenization(document)
        inferred_vector = TaggedDocument(words=tokenized_words, tags=["inferred_vector"])[0]
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        return sims
    
    # takes in a string and infers vector and returns vectors and distance
    def get_vector(self, document):  
        tokenized_words = self.tokenization(document)
        inferred_vector = TaggedDocument(words=tokenized_words, tags=["inferred_vector"])[0]
        inferred_vector = self.model.infer_vector(inferred_vector)
        return sparse.csr_matrix(inferred_vector).toarray()

    # takes in a string and infers vector and returns vectors and distance
    def predict_text_main(self, document, col=None):  
        if col == None:
            col = self.df_tags[0]
        tokenized_words = self.tokenization(document)
        inferred_vector = TaggedDocument(words=tokenized_words, tags=["inferred_vector"])[0]
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        return [rec for rec in sims if rec[0] in set(self.df[col].unique())][0][0]

    # creates the labeled sentences list from the pandas df and the columns
    def label_sentences(self, df, X, Y):
        if 'basestring' not in globals():
            basestring = str

        labeled_sentences = []
        df_tags = []

        if isinstance(Y, basestring):
            df_tags.append(Y)
        elif isinstance(Y, list):
            df_tags = Y
        elif not isinstance(Y, list):
            raise TypeError
        self.df = df
        self.x = X
        self.y = Y

        for index, datapoint in df.iterrows():
            tokenized_words = self.tokenization(document)
            labeled_sentences.append(TaggedDocument(words=tokenized_words, tags=[datapoint[i] for i in df_tags]))
        return labeled_sentences

    # makes the new_classes np.array that holds the labels
    def class_maker( self, y_true, y_pred, total_label_accuracy ):
        class_list = []
        for i in total_label_accuracy.iteritems():
            class_list.append(str(float(i[1])))
        class_list.sort()
            
        new_classes = np.array([])
        stuff = unique_labels(y_true, y_pred)
        for one in stuff:
            for two in class_list:
                if one == two:
                    new_classes = np.append(new_classes, one)
                    
        new_classes= np.ndarray.astype(new_classes, dtype=float)
        np.ndarray.sort(new_classes)
        new_classes = np.ndarray.astype(new_classes, dtype=str)
        
        return list(new_classes)        
    
    # Takes a series of text and returns a series of predictions
    def predict(self, X):  
        if self.verbose:
            from tqdm import tqdm
            tqdm.pandas()
            return X.progress_apply(self.predict_text_main)
        else:
            return X.apply(self.predict_text_main)
        
    #This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`
    def plot_confusion_matrix(self, y_true, y_pred, classes, total_label_accuracy, normalize=False, title=None, cmap=plt.cm.Blues): 

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        label = self.class_maker(y_true, y_pred, total_label_accuracy['Tag'])
        cm = confusion_matrix(y_true, y_pred, labels=label)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=label, yticklabels=label,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax
    
    #This function prints the confusion matrix for a test set when given a pandas df
    # takes in a df and infers vector and returns vectors and distance
    def predict_text_from_df_manually_coded(self, df, true_label, pred_label, text_label, verbose=True, return_sims=None):  
        
        if verbose:
            for i in tqdm(df.index):
                document = df.at[i, text_label]
                tokenized_words = self.tokenization(document)
                inferred_vector = TaggedDocument(words=tokenized_words, tags=["inferred_vector"])[0]
                inferred_vector = self.model.infer_vector(inferred_vector)
                sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
                tags = []
                for col in self.df_tags:
                    tags.append([rec for rec in sims if rec[0] in set(self.df[col].unique())][0][0])
                df.at[i, pred_label] = tags[0]
                if return_sims != None:
                    df.at[i, return_sims] = sims[0]
        else:
            for i in df.index:
                document = df.at[i, text_label]
                tokenized_words = self.tokenization(document)
                inferred_vector = TaggedDocument(words=tokenized_words, tags=["inferred_vector"])[0]
                inferred_vector = self.model.infer_vector(inferred_vector)
                sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
                tags = []
                for col in self.df_tags:
                    tags.append([rec for rec in sims if rec[0] in set(self.df[col].unique())][0][0])
                df.at[i, pred_label] = tags[0]
                if return_sims != None:
                    df.at[i, return_sims] = sims[0]
            
        if verbose==True:   
            total_label_accuracy = pd.DataFrame( columns=[ 'Tag', 'F1 Score'])
            
            num = 0
            for i in df[pred_label].unique():
                total_label_accuracy.at[num, 'Tag'] = i
                num += 1
            total_label_accuracy = total_label_accuracy.sort_values(by=['Tag'])
            total_label_accuracy.reset_index(drop=True, inplace=True)
   
            test_iter = []
            test_iter_before = list(df[true_label])
            for i in test_iter_before:
                i = float(i)
                i = str(i)
                if i.lower() == 'nan':
                    continue
                test_iter.append(i)

            train_iter = []
            train_iter_before = list(df[pred_label])
            for i in train_iter_before:
                i = float(i)
                i = str(i)
                if i.lower() == 'nan':
                    continue
                train_iter.append(i)
            
            df[true_label] = pd.Series(test_iter)
            df[pred_label] = pd.Series(train_iter)
                
            y_true = np.array(test_iter)
            y_pred = np.array(train_iter)
            
            label = self.class_maker(y_true, y_pred, total_label_accuracy['Tag'])
          
            num = 0
            for i in total_label_accuracy.index:
                total_label_accuracy.at[i, 'F1 Score'] = total_label_accuracy.at[i, 'F1 Score']
                num += 1
                
            num = 0
            labelaccuracy = f1_score(df[true_label], df[pred_label], labels=label, average=None)
            for i in labelaccuracy:
                total_label_accuracy.at[num, 'F1 Score'] = i
                num += 1
            
            num = 0
            recall_l = recall_score(df[true_label], df[pred_label], labels=label, average=None)
            for i in recall_l:
                total_label_accuracy.at[num, 'Recall Score'] = i
                num += 1
                
            num = 0
            precision_l = precision_score(df[true_label], df[pred_label], labels=label, average=None)
            for i in precision_l:
                total_label_accuracy.at[num, 'Precision Score'] = i
                num += 1
            
            if verbose==True: print("Labeled F1 Score: ")
            if verbose==True: print(total_label_accuracy.to_string(index=False))
            if verbose==True: print()    
            
            
        if verbose:
            cm = self.plot_confusion_matrix_predict(df, true_label, pred_label)
        else:
            cm = self.plot_confusion_matrix_predict(df, true_label, pred_label, verbose=False)
        return df, cm
    
    #This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`
    def plot_confusion_matrix_predict(self, df, true_label, pred_label, verbose=True): 

        import matplotlib.pyplot as plt
        from sklearn import svm, datasets
        from sklearn.metrics import confusion_matrix
        from sklearn.utils.multiclass import unique_labels
        import numpy as np

        y_true = []
        y_pred = []
        classes = []

        for i in df.index:
            y_true.append(int(round(float(df.at[i, true_label]))))
            y_pred.append(int(round(float(df.at[i, pred_label]))))
            if str(int(round(float(df.at[i, true_label])))) not in classes: ### changed from pred to true
                classes.append(str(int(round(float(df.at[i, true_label])))))
        classes.sort()
        classes = np.array(classes, dtype=np.int64)

        normalize=True
        title=None
        cmap=plt.cm.Blues

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        cm = confusion_matrix(y_true, y_pred)
        stuff = unique_labels(y_true, y_pred)
        classes = classes[stuff]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        if verbose:
            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=classes, yticklabels=classes,
                   title=title,
                   ylabel='True label',
                   xlabel='Predicted label')

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="black")
            fig.tight_layout()
        
        return cm 
