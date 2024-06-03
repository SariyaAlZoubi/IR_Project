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
    'sorta', 'lemme', 'coulda', 'shoulda', 'woulda', 'whereby', 'many', 'much', 'want',
    'always'
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

    def remove_punctuation_and_numbers(tokens):
      # Create a translation table to remove punctuation and numbers
      translator = str.maketrans({char: ' ' if char not in '&0123456789' else '' for char in string.punctuation + string.digits})

      # Process each token individually
      cleaned_tokens = []
      for token in tokens:
        no_punct_or_nums = token.translate(translator)
        clean_token = re.sub(r'\s+', ' ', no_punct_or_nums).strip()
        if clean_token:  # Avoid adding empty strings
          cleaned_tokens.append(clean_token)
      return cleaned_tokens


    @staticmethod
    def toLowercase(text):
        return text.lower()

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
        text = DataPreProcessing.toLowercase(text)
        tokens = DataPreProcessing.tokenize_text(text)
        tokens = DataPreProcessing.remove_stopwords(tokens)
        tokens = DataPreProcessing.remove_punctuation_and_numbers(tokens)
        tokens = DataPreProcessing.stem_tokens(tokens)
        return ' '.join(tokens)
