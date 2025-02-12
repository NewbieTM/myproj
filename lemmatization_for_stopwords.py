from pymorphy3 import MorphAnalyzer
from functools import lru_cache

morph = MorphAnalyzer()


@lru_cache(maxsize=10000)
def cached_parse(word):
    return morph.parse(word)[0].normal_form

if __name__ == '__main__':
    word = 'калорий'
    res = cached_parse(word)
    print(res)

