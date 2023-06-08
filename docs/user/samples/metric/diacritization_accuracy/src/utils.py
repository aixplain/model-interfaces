__author__='aiXplain'

import re

def remove_non_arabic(text:str):
    """Remove non arabic text

    Args:
        text (str): text

    Returns:
        str: preprocessed text
    """
    tmp = re.sub(r'([a-zA-Z]+)', r' ', text)
    tmp = re.sub(r' +', ' ', tmp).strip()
    return tmp


def remove_superfluous_arabic(text:str):
    """Remove superfluos Arabic expressions

    Args:
        text (str): text to be preprocessed

    Returns:
        str: preprocessed text
    """
    ArabicSuperfluous = re.compile(
        '[\u06DC\u06DF\u06E0\u06E2\u06E3\u06E5\u06E6\u06E8\u06EA\u06EB\u06EC\u06ED\u0653]+')
    tmp = ArabicSuperfluous.sub(r' ', text)
    tmp = re.sub(r' +', ' ', tmp).strip()
    return tmp


def remove_default_diacritics_arabic(text:str):
    return text.replace('\u064e\u0627', '\u0627').replace('\u0650\u064A', '\u064A').replace('\u064F\u0648', '\u0648').replace('\u0625\u0650', '\u0625')


def remove_kashida_dagger_arabic(text:str):
    return text.replace('ـ', '').replace('\u0670', '')

def remove_diacritics(text:str):
    arabic_diacritics = re.compile('[\u0640\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670]+')
    return arabic_diacritics.sub(r'', text)

def split_diacritics(word):
    tmp = word.strip()

    tmp = re.sub(r'([\u0640\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670]+)', r' \1 ', tmp,
                    flags=re.UNICODE)  # tag the diacritics
    tmp = re.sub(r'([^\u0640\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670 ])', r'\1*', tmp,
                    flags=re.UNICODE)  # insert * between after each non diacritic character
    tmp = re.sub(r'\* ', r' ',
                    tmp)  # make sure the * is only between 2 letters not between a letter and a diacritic
    tmp = re.sub(r'\*', ' * ', tmp).strip()
    tmp = re.sub(r' +', ' ', tmp).strip()
    tmp = tmp.split(' ')
    it = iter(tmp)
    tagged_letters_diacs = [*zip(it, it)]  # list(zip(it, it))
    return tagged_letters_diacs

def remove_punctuation(text:str):
    punctuation = re.compile('[\u0000-\u002F\u003A-\u0040\u00B0' +
                                '\u2000-\u206F\u0021-\u002F' +
                                '\u005B-\u0060\u007B-\u007F\u00A1\u00A6\u00AB-\u00AF' +
                                '\u00B7\u00B8\u00BB\u00BF\u2E00-\u2E4E\u3000-\u301F\uFE30-\uFE4F\uFF01-\uFF0F' +
                                '\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF64\uFE50-\uFE6B\uFE10-\uFE19' +
                                '\u0609\u060A\u060C\u060D\u061B\u061E\u061F\u066A' +  # Arabic specific
                                ']+')

    tmp = punctuation.sub(r' ', text)
    tmp = re.sub(r' +', ' ', tmp).strip()
    return tmp