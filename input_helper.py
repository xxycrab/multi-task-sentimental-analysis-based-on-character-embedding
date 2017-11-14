import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import data_helpers
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

class InputHelper(object):
    pre_emb = dict()

    def loadW2V(self,emb_path, type="textgz"):
        print("Loading W2V data...")
        num_keys = 0
        if type=="textgz":
            # this seems faster than gensim non-binary load
            for line in gzip.open(emb_path):
                l = line.strip().split()
                self.pre_emb[l[0]]=np.asarray(l[1:])
            num_keys=len(self.pre_emb)
        else:
            self.pre_emb = Word2Vec.load_word2vec_format(emb_path,binary=True)
            self.pre_emb.init_sims(replace=True)
            num_keys=len(self.pre_emb.vocab)
        print("loaded word2vec len ", num_keys)
        gc.collect()

    def deletePreEmb(self):
        self.pre_emb=dict()
        gc.collect()
    
    def getTsvData(self, dataset_name):
        print("Loading training data of "+ str(dataset_name))
        if dataset_name == "mrpolarity":
            dataset = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                                            cfg["datasets"][dataset_name]["negative_data_file"]["path"])
        elif dataset_name == "20newsgroup":
            dataset = data_helpers.get_datasets_20newsgroup(subset="train",
                                                             categories=cfg["datasets"][dataset_name]["categories"],
                                                             shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                             random_state=cfg["datasets"][dataset_name]["random_state"])
        elif dataset_name == "subjectivity":
            dataset = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["subjective"]["path"],
                                                            cfg["datasets"][dataset_name]["objective"]["path"])
        elif dataset_name == "book_review":
            dataset = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                                            cfg["datasets"][dataset_name]["negative_data_file"]["path"])
        elif dataset_name == "localdata":
            dataset = data_helpers.get_datasets_localdata(
                container_path=cfg["datasets"][dataset_name]["container_path"],
                categories=cfg["datasets"][dataset_name]["categories"],
                shuffle=cfg["datasets"][dataset_name]["shuffle"],
                random_state=cfg["datasets"][dataset_name]["random_state"])
        return dataset
   
    def getUnlabelData(self, filepath):
        print("Loading unlabelled data from "+filepath)
        x=[]
        for line in open(filepath):
            l=line.strip()
            if len(l)<1:
                continue
            x.append(l)
        return np.asarray(x)
 
    def dumpValidation(self,x_text,y,shuffled_index,dev_idx,i):
        print("dumping validation "+str(i))
        x_shuffled=x_text[shuffled_index]
        y_shuffled=y[shuffled_index]
        x_dev=x_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
        del x_shuffled
        del y_shuffled
        with open('validation.txt'+str(i),'w') as f:
            for text,label in zip(x_dev,y_dev):
                f.write(str(label)+'\t'+text+'\n')
            f.close()
        del x_dev
        del y_dev
    
    # Data Preparatopn
    # ==================================================

    def get_datasets(self, percent_dev, batch_size, dataset_names):
        x_list=[]
        y_list=[]
        multi_train_size = len(dataset_names)
        for i in range(multi_train_size):
            dataset = self.getTsvData(dataset_names[i])
            x, y = data_helpers.load_data_labels(dataset)
            x_list.append(x)
            y_list.append(y)
        gc.collect()
        # Build vocabulary
        print("Building vocabulary")

        max_document_length=0
        for i in range(multi_train_size):
            max_document_length = max(max(len(x.split(" ")) for x in x_list[i]), max_document_length)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,min_frequency=1)
        vocab_processor.fit_transform(np.concatenate(x_list,axis=0))
        print("Length of loaded vocabulary ={}".format( len(vocab_processor.vocabulary_)))
        i1=0
        train_set=[]
        dev_set=[]
        sum_no_batches = 0
        for x_text,y in zip(x_list, y_list):
            x = np.asarray(list(vocab_processor.transform(x_text)))
            x = np.concatenate((np.zeros((len(x),5)),x),axis=1)
            # Randomly shuffle data
            np.random.seed(10)
            shuffle_indices = np.random.permutation(np.arange(len(y)))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
            dev_idx = -1*int(float(len(y_shuffled))*percent_dev)
            #self.dumpValidation(x,y,shuffle_indices,dev_idx,i1)
            del x
            del x_text
            del y
            # Split train/test set
            # TODO: This is very crude, should use cross-validation
            x_train, x_dev = x_shuffled[:dev_idx], x_shuffled[dev_idx:]
            y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
            print("Train/Dev split for {}: {:d}/{:d}".format(dataset_names[i1], len(y_train), len(y_dev)))
            sum_no_batches = sum_no_batches+(len(y_train)//batch_size)
            train_set.append((x_train,y_train))
            dev_set.append((x_dev,y_dev))
            del x_shuffled
            del y_shuffled
            del x_train
            del x_dev
            i1=i1+1
        del x_list
        del y_list
        gc.collect()
        return train_set,dev_set,vocab_processor,sum_no_batches
    
    def getTestDataSet(self, data_path, vocab_path, max_document_length, filter_h_pad):
        x_temp,y = self.getTsvData(data_path)

        # Build vocabulary
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length-filter_h_pad,min_frequency=1)
        vocab_processor = vocab_processor.restore(vocab_path)

        x = np.asarray(list(vocab_processor.transform(x_temp)))
        x = np.concatenate((np.zeros((len(x),filter_h_pad)),x),axis=1)
        # Randomly shuffle data
        del x_temp
        del vocab_processor
        gc.collect()
        return x, y
