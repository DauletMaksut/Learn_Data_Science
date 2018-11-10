from rx import Observable
import signal
def word_in_file (fn):
    file = open(fn)
    return Observable.from_(file) \
        .flat_map(lambda line: Observable.from_(line.split())) \
        .map(lambda w: w.lower()) \
        .group_by(lambda word: word) \
        .map(lambda group: group.count().map(lambda pair: (group.key, pair))) \
        .merge_all() \
        .to_dict(lambda pair: pair[0], lambda pair: pair[1])

Observable.interval(300) \
        .map(lambda t: word_in_file("text.txt")) \
        .merge_all() \
        .distinct_until_changed() \
        .subscribe(lambda value: print(value))
signal.pause()
