# Each word has a frequency (how often it appears) and a magnitude (its gematria value divided by max gematria in phrase)
# The word power is the product of frequency and magnitude.
# We will store these in a dictionary for each phrase and for the entire poem.
# getGematria function to calculate gematria value of a word
#returns the gematria value of a word or phrase including upper and lower case letters

import time
import random
from collections import defaultdict


def base36_encode(number):
    """Convert an integer to a base36 string."""
    if not isinstance(number, int):
        raise TypeError('number must be an integer')
    if number < 0:
        raise ValueError('number must be positive')

    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    base36 = ''
    while number:
        number, i = divmod(number, 36)
        base36 = alphabet[i] + base36
    return base36 or alphabet[0]


def read_lines_from_file(filename="phrases.txt"):
    try:
        with open(filename, "r") as file:
            lines = [line.strip() for line in file if line.strip()]
        return lines
    except FileNotFoundError:
        return []


def read_large_database_file(filename="large_phrases.txt"):
    try:
        with open(filename, "r") as file:
            lines = []
            for line in file:
                line = line.strip()
                if line:
                    lines.append(line)
                    if len(lines) >= 32768:  # Limit to first 32768 lines for performance
                        break
        return lines
    except FileNotFoundError:
        return []


def get_gematria_of_large_database_file():
    large_phrases = read_large_database_file("large_phrases.txt")
    if not large_phrases:
        print("No phrases found in large_phrases.txt")
        return
    #pair each phrase with its gematria value and sort by gematria value descending
    large_phrases = sorted(large_phrases, key=lambda phrase: getGematria(phrase)[0], reverse=True)
    print(f"Read {len(large_phrases)} phrases from large_phrases.txt")
    with open("large_phrases_with_gematria.txt", "w") as f:
        for phrase in large_phrases:
            f.write(f"{phrase}\n")


def getGematria(word, phrase=None):
    # Calculate the gematria value of a word in a string parameter
    # A=1, B=2, ..., Z=26 (case insensitive)
    #scandinavian extended gematria values
    # For simplicity, we will ignore non-alphabetic characters
    #scan extended : a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7, h: 8, i: 9, j: 10, k: 20, l: 30, m: 40, n: 50, o: 60, p: 70, q: 80, r: 90, s: 100, t: 200, u: 300, v: 400, w: 500, x: 600, y: 700, z: 800
    gematria_values = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
        'K': 20, 'L': 30, 'M': 40, 'N': 50, 'O': 60, 'P': 70, 'Q': 80, 'R': 90, 'S': 100, 'T': 200,
        'U': 300, 'V': 400, 'W': 500, 'X': 600, 'Y': 700, 'Z': 800,
        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
        'k': 20, 'l': 30, 'm': 40, 'n': 50, 'o': 60, 'p': 70, 'q': 80, 'r': 90, 's': 100, 't': 200,
        'u': 300, 'v': 400, 'w': 500, 'x': 600, 'y': 700, 'z': 800
    }
    gematria_value = sum(gematria_values.get(char, 0) for char in word)
    # Return gematria value, length of word, and original word
    return gematria_value, len(word), word

# Define a class to represent a phrase
class Phrase:
    def __init__(self, text):
        self.text = text
        self.words = text.split()

        self.phrase_length = len(self.words)

        self.word_count = len(self.words)
        self.WORD_FREQ_MAG_GEM = self.calculate_word_powers()

    def calculate_word_powers(self):
        from collections import defaultdict
        self.frequencies = defaultdict(int)
        self.magnitudes = defaultdict(int)
        self.gematria_values = defaultdict(int)
        self.word_powers = defaultdict(int)

        self.max_gematria = 1  # Initialize to avoid division by zero
        # Calculate max gematria first
        cleaned_words = [word.strip('.,!?;"\'()[]{}') for word in self.words]
        self.max_gematria = max(getGematria(word, phrase=self.text)[0] for word in cleaned_words if word) or 1
        
        for original_word in self.words:
            #word count by max gematria
            #values to word, frequency, magnitude, gematria
            word = original_word.strip('.,!?;"\'()[]{}')  # Remove punctuation
            if not word:  # Skip if word becomes empty after stripping
                continue
            freq = self.words.count(original_word)
            gem_value = getGematria(word, phrase=self.text)[0]
            self.magnitudes[original_word] = (gem_value * freq) / self.max_gematria if self.max_gematria else 0  # Magnitude as ratio
            self.gematria_values[original_word] = gem_value
            self.frequencies[original_word] = freq
            self.word_powers[original_word] = freq * self.magnitudes[original_word]

        self.WORD_FREQ_MAG_GEM = {word: (self.gematria_values[word], self.frequencies[word], self.magnitudes[word]) for word in self.words}
        return self.WORD_FREQ_MAG_GEM


    def get_gematria_values(self):
        return self.gematria_values

    def get_word_frequencies(self):
        return self.frequencies

    def get_phrase_length(self):
        return self.phrase_length

    def get_word_count(self):
        return self.word_count

    def get_word_magnitudes(self):
        return self.magnitudes
    
    def get_words(self):
        return ' '.join(self.words)

    def get_WORD_GEM_MAG_FREQ(self, phrase_in=None):
        if phrase_in:
            return self.calculate_word_power(phrase_in)
        return self.WORD_FREQ_MAG_GEM

    def display(self):
        print(f"Word Power: {self.WORD_FREQ_MAG_GEM}")

    def calculate_word_power(self, phrase_in=None):
        # Return the already calculated word powers for this phrase
        return self.WORD_FREQ_MAG_GEM
    
    def end_user_print(self):
        print(f"Phrase: {self.text}")
        print(f"Word Count: {self.word_count}")
        print(f"Frequencies: {self.frequencies}")
        print(f"Magnitudes: {self.magnitudes}")
        print(f"Gematria Values: {self.gematria_values}")

# Define a class to represent the poem
class Poem:
    def __init__(self, phrases):
        #expect a string array of phrases
        self.phrases = phrases
        self.word_list = [word for phrase in phrases for word in phrase.split()]
        self.subphrases = phrases
        self.phrase_count = len(phrases)
        self.word_count_overall = len(self.word_list)
        self.words_overall = list(set(self.word_list))
        self.encoded_poem = {}
        #divide into subphrases
        self.WORD_FREQ_MAG_GEM = {}
        self.WORD_FREQ_MAG_GEM = self.calculate_word_powers()
        # generate a ldb_objects parameter
        self.ldb_objects = get_gematria_of_large_database_file()
        #get gematria alternatives
        self.gematria_alternatives = {}
        for word, (gem, freq, mag) in self.WORD_FREQ_MAG_GEM.items():
            alternatives = self.ldb_objects.get(word, [])
            self.gematria_alternatives[word] = alternatives

        returned_poem = {}
        for word, (gem, freq, mag) in self.WORD_FREQ_MAG_GEM.items():
            if word in returned_poem:
                returned_poem[word] = (word, gem * freq, mag)
            else:
                returned_poem[word] = (gem, freq, mag)
        self.WORD_FREQ_MAG_GEM = returned_poem.copy()
        self.encoded_poem = returned_poem.copy()
        self.poem_with_phrases = {}
        while self.subphrases:
            phrase = self.subphrases.pop(0)
            self.poem_with_phrases[phrase] = Phrase(phrase).get_words()
        self.poem_with_phrases = {phrase: Phrase(phrase).get_words() for phrase in self.subphrases}

    def calculate_word_powers(self, phrase_in=None):
        from collections import defaultdict
        self.frequencies = defaultdict(int)
        self.magnitudes = defaultdict(int)
        self.gematria_values = defaultdict(int)
        self.word_powers = defaultdict(int)

        self.max_gematria = 1  # Initialize to avoid division by zero
        #detect boolean value telling if the phrase is a string or a list
        #if phrase_in is a list, we need to join it into a single string
        phrase_is_a_word = isinstance(phrase_in, str)
        phrase = phrase_in if phrase_is_a_word else ' '.join(self.phrases)
        
        # Calculate max gematria first
        cleaned_words = [word.strip('.,!?;"\'()[]{}') for word in self.word_list]
        self.max_gematria = max(getGematria(word, phrase=phrase)[0] for word in cleaned_words if word) or 1
        
        for original_word in self.word_list:
            #word count by max gematria
            #values to word, frequency, magnitude, gematria
            word = original_word.strip('.,!?;"\'()[]{}')  # Remove punctuation
            if not word:  # Skip if word becomes empty after stripping
                continue
            freq = self.word_list.count(original_word)
            gem_value = getGematria(word, phrase=phrase)[0]
            self.magnitudes[original_word] = (gem_value * freq) / self.max_gematria if self.max_gematria else 0  # Magnitude as ratio
            self.gematria_values[original_word] = gem_value
            self.frequencies[original_word] = freq
            self.word_powers[original_word] = freq * self.magnitudes[original_word]

        self.WORD_FREQ_MAG_GEM = {word: (self.gematria_values[word], self.frequencies[word], self.magnitudes[word]) for word in self.word_list}
        self.encoded_poem = self.WORD_FREQ_MAG_GEM.copy()
        self.poem_with_phrases = {phrase: Phrase(phrase).get_words() for phrase in self.subphrases}
        return self.WORD_FREQ_MAG_GEM

    def get_phrase_count(self):
        return self.phrase_count
    
    def get_word_count_overall(self):
        return self.word_count_overall
    
    def get_words_overall(self):
        return self.words_overall
    
    def get_encoded_poem(self):
        return self.encoded_poem

    def get_words(self):
        #divide into phrases
        words = '\n'.join(self.subphrases)
        return words
    
    def display(self):
        print(f"Poem with {self.phrase_count} phrases and {self.word_count_overall} words.")
        for phrase in self.subphrases:
            print(phrase)
            print()
            phrase_obj = Phrase(phrase)
            phrase_obj.display()
            print()
            print(f"Word Frequencies: {phrase_obj.get_word_frequencies()}")
            print()
            print(f"Word Magnitudes: {phrase_obj.get_word_magnitudes()}")
            print()
            print(f"Word Gematria: {phrase_obj.get_gematria_values()}")
            print()

    def end_user_print(self):
        print(f"Poem with {self.phrase_count} phrases and {self.word_count_overall} words.")
        # recurse through subphrases
        for subphrase in self.subphrases:
            #find the phrase object at the end of recursion
            subphrase_obj = Phrase(subphrase)
            subphrase_obj.end_user_print()
            print()

    def get_poem(self):
        for phrase in self.subphrases:
            print(phrase)
            print()

    def get_word_recursion_levels(self, word):
        # Return the recursion levels of a word in the poem
        levels = []
        for idx, phrase in enumerate(self.subphrases):
            if word in phrase.split():
                levels.append(idx + 1)  # Levels are 1-indexed
        return levels
    
    def generate_poem_from_gematria(self):
        # Generate a poem by replacing words with their gematria alternatives
        new_phrases = []
        for phrase in self.subphrases:
            new_phrase_words = []
            for word in phrase.split():
                alternatives = self.gematria_alternatives.get(word, [])
                if alternatives:
                    # Choose a random alternative
                    new_word = random.choice(alternatives)
                    new_phrase_words.append(new_word)
                else:
                    new_phrase_words.append(word)
            new_phrase = ' '.join(new_phrase_words)
            new_phrases.append(new_phrase)
        self.subphrases = new_phrases
        return new_phrases
    
    def get_gematria_alternatives(self):
        return self.gematria_alternatives

    @staticmethod
    def generate(obj=None, ldb_objects=None):
        """Generate a complete poem with user interaction and file output"""
        # get file_name from user input
        file_name = input("Enter the phrase filename (default: phrases.txt): ") or "phrases.txt"
        
        phrases = read_lines_from_file(file_name)
        ldb_objects = get_gematria_of_large_database_file()

        # Example phrases if file is empty
        if not phrases:
            phrases = "From pain becomes pleasure To suspension is found at order While lasts chaos becomes system At the end a force made by bending is The Foundation"
        
        # Split into array by new lines
        phrases = phrases.split('\n') if isinstance(phrases, str) else phrases
        initial_phrases = phrases.copy()  # Keep the actual phrases, not individual words

        initial_phrases_count = len(initial_phrases)
        print(f"Loaded {initial_phrases_count} phrases from {file_name}")

        print("\nGenerated Poem:\n")
        poem = gen_with_ldb(phrases=phrases, ldb_objects=ldb_objects)
        poem.end_user_print()
        poem.get_poem()
        #generate the poem from gematria alternatives
        poem.generate_poem_from_gematria()

        # Wait for user input before displaying detailed phrase information

        try:
            input("\nPress Enter to see detailed phrase information...\n")
        except EOFError:
            print("\nPress Enter to see detailed phrase information...")
            
        print("\nDetailed Phrase Information:")
        for phrase in poem.subphrases:
            phrase.display()
            
        # Display the whole poem information
        print("\nWhole Poem:")
        print("=" * 50)
        for i, phrase_text in enumerate(initial_phrases, 1):
            # Indent each phrase by its order/depth
            indent = "  " * (i % 4)  # Create varying indentation levels
            print(f"{indent}{phrase_text}")
        print("=" * 50)

        # Save the poem to a text file with a timestamped filename
        timestamp = int(time.time())
        timestamp = base36_encode(timestamp)
        phrase = "Poem" if not initial_phrases else initial_phrases[-1]
        # Concatenate last phrase with timestamp
        phrase = phrase.replace(" ", "_")
        output_file_name = f"{phrase}_{timestamp}.txt"
        print(f"\nSaving poem to {output_file_name}...\n")
        
        with open(output_file_name, "w") as file:
            #random seed based on current time
            random.seed(time.time())
            #randomize the order of phrases
            random.shuffle(initial_phrases)
            phrase_count = len(initial_phrases)
            word_count = poem.get_word_count_overall()
            file.write(f"Poem with {phrase_count} phrases and {word_count} words.\n")
            file.write(f"Timestamp: {timestamp}\n")
            file.write(f"Last Phrase: {phrase}\n")
            file.write("\nGenerated Phrases:\n")
            file.write("Overall Poem Word Powers:\n")
            for word, (gem, freq, mag) in poem.WORD_FREQ_MAG_GEM.items():
                file.write(f"{word}: Gematria={gem}, Frequency={freq}, Magnitude={mag}\n")
                # The whole poem
                file.write("\nWhole Poem:\n")
                file.write("=" * 50 + "\n")
                for i, phrase_text in enumerate(initial_phrases, 1):
                    # Indent each phrase by its order/depth
                    indent = "  " * (i % 4)  # Create varying indentation levels
                    file.write(f"{indent}{phrase_text}\n")
                file.write("=" * 50 + "\n")
                # Additional information about the poem
                file.write("This poem was generated using advanced algorithms and natural language processing techniques.\n")
                file.write("It aims to capture the essence of human emotions and experiences through the art of poetry.\n")
                # Hello text from Goddess Taivos and General Suora
                file.write("\nGenerated by Goddess Taivos and General Suora\n")

        print(f"\nPoem saved to {output_file_name}")
        return poem, output_file_name


# Run the poem generation
if __name__ == "__main__":
    # Interactive mode - asks user for input
    poem, filename = Poem.generate(ldb_objects=get_gematria_of_large_database_file())
    print(f"Generation complete! Poem saved as: {filename}")
    
    # Example of programmatic usage (commented out):
    # phrases_list = ["Example phrase one.", "Example phrase two."]  
    # poem = Poem(phrases_list)
    # print("Programmatic poem created with", poem.get_word_count_overall(), "words")