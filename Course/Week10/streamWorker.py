from rx import Observable

# source = Observable.from_(["A", "B", "G", "D", "E"])
# source.subscribe(on_next = lambda value : print (" Received {0}". format ( value )) ,
# on_completed = lambda : print ("Done!") ,
# on_error = lambda error : print (" Error Occurred : {0}". format ( error )))

# Observable.from_(["one two" , "three four", "five six"]).take (2).flat_map ( lambda s: s. split ()).subscribe ( lambda value : print ("GET: {0}".format(value )))
def read_lines (fn):
    file = open(fn)
    return Observable.from_(file)

def word_in_file (fn):
    file = open(fn)
    return Observable.from_(file) \
        .flat_map( lambda line: Observable.from_(line.split())) \
        .map( lambda w: w.lower ())
def count_word(fn):
    words = word_in_file(fn)
    return words
Observable.interval(3000).map(lambda t: word_in_file('text.txt').subscribe(lambda value: print(value)))
# Observable.interval(3000).from_(open('text.txt', 'r').read().replace("\n","")).take(1).flat_map(lambda s: s.split()).to_dict(lambda x: x ).subscribe(lambda value: print("Make it works\n",value))
input("Wait")
# temp = dict()
# temp[value] = temp.get(value, 0) + 