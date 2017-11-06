import random
from code import dataset
from code.transitionparser import TransitionParser
from code.evaluate import DependencyEvaluator
from code.featureextractor import FeatureExtractor
from code.transition import Transition
from code.dependencygraph import DependencyGraph

if __name__ == '__main__':
    data = dataset.get_english_train_corpus().parsed_sents()
    random.seed(1234)
    subdata = random.sample(data, 200)

    tp = TransitionParser(Transition, FeatureExtractor)
    tp.train(subdata)
    tp.save('english.model')
    testdata = dataset.get_english_test_corpus().parsed_sents()

    parsed = tp.parse(testdata)

    with open('test.conll', 'w') as f:
        for p in parsed:
            f.write(p.to_conll(10).encode('utf-8'))
            f.write('\n')

    ev = DependencyEvaluator(testdata, parsed)
    print "UAS: {} \nLAS: {}".format(*ev.eval())

    # parsing arbitrary sentences (english):
    #s = DependencyGraph.from_sentence('I am having a good time!')

    #tp = TransitionParser.load('english.model')
    #parsed = tp.parse([s])
    #print parsed[0].to_conll(10).encode('utf-8')
    #with open('test0.conll', 'w') as f:
        #f.write(parsed[0].to_conll(10).encode('utf-8'))
