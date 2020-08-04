from pyhanlp import HanLP
from snownlp import SnowNLP
import pkuseg


# Chinese segmentation
def zh_segmentator(line):
    return " ".join(pkuseg.pkuseg().cut(line))


# Chinese simplify -> Chinese traditional standard
def zh_traditional_standard(line):
    return HanLP.convertToTraditionalChinese(line)


# Chinese simplify -> Chinese traditional (HongKong)
def zh_traditional_hk(line):
    return HanLP.s2hk(line)


# Chinese simplify -> Chinese traditional (Taiwan)
def zh_traditional_tw(line):
    return HanLP.s2tw(line)


# Chinese traditional -> Chinese simplify (v1)
def zh_simplify(line):
    return HanLP.convertToSimplifiedChinese(line)


# Chinese traditional -> Chinese simplify (v2)
def zh_simplify_v2(line):
    return SnowNLP(line).han
