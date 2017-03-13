# KoNLPy 분석기 예제

from konlpy.tag import Kkma
from konlpy.utils import pprint
kkma = Kkma()
text = '네 맞습니다. 오웅천 입니다.질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'
# sentences 분석
pprint(kkma.sentences(text))

# nouns 분석
pprint(kkma.nouns(text))

# 형태소 분석
pprint(kkma.pos(text))

