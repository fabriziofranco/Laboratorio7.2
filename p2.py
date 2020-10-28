import numpy as np

def score1(q, doc):
    return np.dot(q, doc)

def score2(q, doc):
    return np.dot(q, doc) / (np.linalg.norm(q) * np.linalg.norm(doc))    

def retrieval(collection, query, func_score):
    result = []
    for i in range(len(collection)):
        sim = func_score(query, collection[i])
        result.append( (i+1, sim) )#[ (doc1, sc1), (doc, sc2) ]
    result.sort(key = lambda  tup: tup[1])
    return result

#####################   main ##############################
collection = [
    np.array([1.452,0,2.122,3.564,4.123,0,0,2.342,0,0,0,1.975,4.543,0,6.134,2.234]),
    np.array([0,2.093,0,0,4.245,1.234,0,0,0,0,2.345,0,2.135,0,0,3.456])
]
query = np.array([0,1.345,1.453,1.987,0,2.133,0,0,0,0,0,0,3.452,0,0,4.234])

### aplicar score 
result = retrieval(collection, query, score2)
print(result)