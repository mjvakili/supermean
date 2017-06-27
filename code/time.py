import time , sampler , numpy

d = numpy.zeros((625))

s = time.time()

sampler.imatrix(d, 4)

print (time.time()-s)
