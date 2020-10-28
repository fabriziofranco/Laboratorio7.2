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
    np.array([15,5, 20,25]),
    np.array([30,0,22,0])
]
query = np.array([115, 10, 2, 0])
query = np.log10(1 + query) 

for i in range(len(collection)):
    collection[i] = np.log10(1 + collection[i])

### aplicar score 1

result = retrieval(collection, query, score1)
print(result)