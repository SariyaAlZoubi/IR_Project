import re
import string

from nltk.corpus import wordnet

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag


class DataPreProcessing:
    custom_words = {
        'isnt', 'arent', 'im', 'id', 'ie', 'eg', 'ive', 'whatev', 'wed', 'somehow',
        'going', 'get', 'yes', 'no', 'couldnt', 'didnt', 'dont', 'doesnt', 'would',
        'could', 'should', 'cant', 'wont', 'hasnt', 'hadnt', 'havent', 'mightnt',
        'mustnt', 'neednt', 'shall', 'shant', 'werent', 'wouldnt', 'ought', 'oughtnt',
        'aint', 'gonna', 'wanna', 'whatcha', 'yall', 'ya', 'gotta', 'coulda', 'shoulda',
        'woulda', 'lotta', 'lemme', 'kinda', 'sorta', 'hafta', 'dunno', 'outta', 'alot',
        'yup', 'nope', 'nah', 'yeah', 'uh', 'um', 'uhm', 'okay', 'ok', 'yep', 'hmm',
        'mmm', 'oh', 'hey', 'hi', 'hello', 'bye', 'goodbye', 'please', 'thanks', 'thank',
        'welcome', 'etc', 'alright', 'okay', 'ok', 'gonna', 'gotta', 'wanna', 'kinda',
        'sorta', 'lemme', 'coulda', 'shoulda', 'woulda', 'whereby', 'many', 'much', 'want'
    }
    stop_words = set(stopwords.words('english')).union(custom_words)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    @staticmethod
    def remove_urls(text):
        return re.sub(r'http\S+|www.\S+', '', text)

    @staticmethod
    def remove_non_english_chars(text):
        allowed_chars = string.ascii_letters + string.digits + string.punctuation + " "
        filtered_text = ''.join(char if char in allowed_chars else '' for char in text)
        return filtered_text

    @staticmethod
    def remove_punctuation(text):
        text = re.sub(r'\d+', ' ', text)
        translator = str.maketrans({char: ' ' if char != '&' else '' for char in string.punctuation})
        no_punct = text.translate(translator)
        clean_text = re.sub(r'\s+', ' ', no_punct).strip()
        return clean_text

    # def replace_abbreviation(text):
    #     new_text = []
    #     for i in text.split():
    #         if i.upper() in DataPreProcessing.abbreviations:
    #             new_text.append(DataPreProcessing.abbreviations[i.upper()])
    #         else:
    #             new_text.append(i)
    #     return " ".join(new_text)

    @staticmethod
    def toLowercase(text):
        return text.lower()

    @staticmethod
    def contains_emoji(text):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # other miscellaneous symbols
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return bool(emoji_pattern.search(text))

    @staticmethod
    def any_text_contains_emoji(docs):
        for text in docs['text']:
            if DataPreProcessing.contains_emoji(text):
                return True
        return False

    @staticmethod
    def fix_repeated_chars(text):
        pattern = r'(\w)(\1{2,})'
        fixed_text = re.sub(pattern, r'\1\1', text)
        return fixed_text

    @staticmethod
    def remove_time(text):
        time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\b'
        text_without_time = re.sub(time_pattern, '', text)
        clean_text = re.sub(r'\s+', ' ', text_without_time).strip()
        return clean_text

    @staticmethod
    def remove_null_values(docs):
        return docs.dropna()

    @staticmethod
    def tokenize_text(text):
        return word_tokenize(text)

    @staticmethod
    def remove_stopwords(tokens):
        return [word for word in tokens if word not in DataPreProcessing.stop_words]

    @staticmethod
    def remove_duplicated_chars(tokens):
        def has_duplicated_chars(word):
            return len(set(word)) == 1

        filtered_words = [word for word in tokens if len(word) != 2 and not has_duplicated_chars(word)]
        return filtered_words

    @staticmethod
    def remove_words_start_with_duplicate_chars(text):
        # Define a regular expression pattern to match words starting with duplicate alphabetical characters
        pattern = r'\b([a-zA-Z])\1\w*\b'
        # Use re.sub to replace matched words with an empty string
        result = re.sub(pattern, '', text)
        return result

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    @staticmethod
    def lemmatize_tokens(tokens):
        pos_tagged_tokens = pos_tag(tokens)
        lemmatized_tokens = [
            DataPreProcessing.lemmatizer.lemmatize(token, DataPreProcessing.get_wordnet_pos(pos_tag))
            for token, pos_tag in pos_tagged_tokens
        ]
        return lemmatized_tokens

    @staticmethod
    def stem_tokens(tokens):
        return [DataPreProcessing.stemmer.stem(token) for token in tokens]

    @staticmethod
    def process_text(text):
        text = DataPreProcessing.remove_urls(text)
        text = DataPreProcessing.remove_non_english_chars(text)
        text = DataPreProcessing.remove_time(text)
        text = DataPreProcessing.remove_punctuation(text)
        text = DataPreProcessing.toLowercase(text)
        text = DataPreProcessing.fix_repeated_chars(text)
        text = DataPreProcessing.remove_words_start_with_duplicate_chars(text)
        tokens = DataPreProcessing.tokenize_text(text)
        tokens = DataPreProcessing.remove_stopwords(tokens)
        tokens = DataPreProcessing.remove_duplicated_chars(tokens)
        tokens = DataPreProcessing.stem_tokens(tokens)
        return ' '.join(tokens)
