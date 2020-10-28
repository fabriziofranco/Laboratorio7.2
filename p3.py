import numpy as np

def cosine(q, doc):
    return np.dot(q, doc) / (np.linalg.norm(q) * np.linalg.norm(doc))    

def retrievalCosine1(collection, queryText, k): #query : texto
    result = []
    queryTerms = getTerms(queryText)
    query = getTfIdf(queryTerms) #tf-idf del query        
    for i in range(collection.size):        
        doc = collection.getDocument(i, queryTerms); #tf-idf del documento        
        sim = cosine(query, doc)
        result.append( (doc.id, sim) )#[ (doc1, sc1), (doc, sc2) ]    
    result.sort(key = lambda  tup: tup[1])
    return result[0: k]

def retrievalCosine2(index, queryText, k): #query : texto
    #diccionario de los scores por documento
    score = { };
    #preprocesamiento para extraer los terminos
    queryTerms = getTerms(queryText)
    #tf-idf del query
    query = getTfIdf(queryTerms)     

    #recorriendo cada termino del query
    for i in range(len(queryTerms)):
        # term: {idf: 1.6,  pub: [ (doc1, tf1), (doc2, tf2), (doc3, tf) ]}
        listPub = index.get(queryTerms[i])[pub] 
        idf = index.get(queryTerms[i])[idf]
        #producto punto        
        for par in listPub:
            score[par.docId] += (idf * par.tf) * query[i]

    #obtenemos las normas de cada documento previamente guardado
    normas = index.getNorms();
    for docId in score:
        score[docId] = score[docId] / (normas[docId] * norm(query))

    #calculamos el top k
    result = []
    for docId in score:
        result.append( (docId, score[docId]) )
    result.sort(key = lambda  tup: tup[1])
    return result[0: k]
    