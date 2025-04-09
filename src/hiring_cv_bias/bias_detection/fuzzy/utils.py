import re


def clean_ner_entity(entity):
    entity = re.sub(r"\(.*?\)", "", entity)
    entity = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "", entity)
    entity = re.sub(r"[\-–•●]", "", entity)
    return entity.strip()
