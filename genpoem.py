#!/usr/bin/env python3
"""
Poem Generation Algorithm with Gematria Analysis
Author: Goddess Taivos and General Suora
Date: 26. syyskuuta 2025

This module provides classes for generating and analyzing poems using gematria calculations
with Scandinavian extended values. The algorithm calculates word frequencies, magnitudes,
and creates beautiful formatted output with phrase depth indentation.
"""

import time
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class Word:
    """Represents a single word with its gematria and analysis properties."""
    
    # Scandinavian extended gematria values
    GEMATRIA_VALUES = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
        'K': 20, 'L': 30, 'M': 40, 'N': 50, 'O': 60, 'P': 70, 'Q': 80, 'R': 90, 'S': 100, 'T': 200,
        'U': 300, 'V': 400, 'W': 500, 'X': 600, 'Y': 700, 'Z': 800,
        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
        'k': 20, 'l': 30, 'm': 40, 'n': 50, 'o': 60, 'p': 70, 'q': 80, 'r': 90, 's': 100, 't': 200,
        'u': 300, 'v': 400, 'w': 500, 'x': 600, 'y': 700, 'z': 800
    }
    
    def __init__(self, text: str):
        """Initialize a Word with its text."""
        self.original_text = text
        self.cleaned_text = self._clean_text(text)
        self.gematria_value = self._calculate_gematria()
        self.length = len(self.cleaned_text)
        
    def _clean_text(self, text: str) -> str:
        """Remove punctuation from word for gematria calculation."""
        return text.strip('.,!?;"\'()[]{}')
    
    def _calculate_gematria(self) -> int:
        """Calculate the gematria value using Scandinavian extended values."""
        return sum(self.GEMATRIA_VALUES.get(char, 0) for char in self.cleaned_text)
    
    def __str__(self) -> str:
        return self.original_text
    
    def __repr__(self) -> str:
        return f"Word('{self.original_text}', gematria={self.gematria_value})"


class Phrase:
    """Represents a phrase containing multiple words with frequency and magnitude analysis."""
    
    def __init__(self, text: str):
        """Initialize a Phrase with its text and analyze its words."""
        self.text = text
        self.words = [Word(word_text) for word_text in text.split()]
        self.word_count = len(self.words)
        
        # Analysis dictionaries
        self.frequencies = defaultdict(int)
        self.gematria_values = defaultdict(int)
        self.magnitudes = defaultdict(float)
        self.word_powers = defaultdict(float)
        
        # Calculate analysis
        self._analyze_words()
    
    def _analyze_words(self):
        """Analyze word frequencies, gematria values, and magnitudes."""
        # Count frequencies and collect gematria values
        for word in self.words:
            self.frequencies[word.cleaned_text] += 1
            self.gematria_values[word.cleaned_text] = word.gematria_value
        
        # Calculate max gematria for magnitude normalization
        max_gematria = max(self.gematria_values.values()) if self.gematria_values else 1
        
        # Calculate magnitudes and word powers
        for word_text in self.frequencies:
            gematria = self.gematria_values[word_text]
            frequency = self.frequencies[word_text]
            magnitude = (gematria * frequency) / max_gematria if max_gematria else 0
            
            self.magnitudes[word_text] = magnitude
            self.word_powers[word_text] = frequency * magnitude
    
    def get_word_analysis(self) -> Dict[str, Tuple[int, int, float]]:
        """Return word analysis as (gematria, frequency, magnitude) tuples."""
        return {
            word: (self.gematria_values[word], self.frequencies[word], self.magnitudes[word])
            for word in self.frequencies
        }
    
    def display(self):
        """Display detailed phrase information."""
        analysis = self.get_word_analysis()
        word_power_str = ', '.join(f"'{word}': ({gem}, {freq}, {mag})" 
                                 for word, (gem, freq, mag) in analysis.items())
        print(f"Word Power: {{{word_power_str}}}")
    
    def __str__(self) -> str:
        return self.text


class Poem:
    """Represents a complete poem with multiple phrases and overall analysis."""
    
    def __init__(self, phrase_texts: List[str]):
        """Initialize a Poem with a list of phrase texts."""
        self.phrase_texts = phrase_texts.copy()
        self.phrases = [Phrase(text) for text in phrase_texts]
        self.phrase_count = len(self.phrases)
        
        # Overall poem statistics
        self.word_count_overall = sum(phrase.word_count for phrase in self.phrases)
        self.words_overall = []
        for phrase in self.phrases:
            self.words_overall.extend(phrase.words)
        
        # Combined analysis across all phrases
        self.overall_frequencies = defaultdict(int)
        self.overall_gematria_values = defaultdict(int)
        self.overall_magnitudes = defaultdict(float)
        self.WORD_FREQ_MAG_GEM = {}
        
        self._analyze_overall()
    
    def _analyze_overall(self):
        """Analyze word statistics across the entire poem."""
        # Collect frequencies and gematria values from all phrases
        for phrase in self.phrases:
            for word_text, frequency in phrase.frequencies.items():
                self.overall_frequencies[word_text] += frequency
                self.overall_gematria_values[word_text] = phrase.gematria_values[word_text]
        
        # Calculate overall max gematria for magnitude normalization
        max_gematria = max(self.overall_gematria_values.values()) if self.overall_gematria_values else 1
        
        # Calculate overall magnitudes
        for word_text in self.overall_frequencies:
            gematria = self.overall_gematria_values[word_text]
            frequency = self.overall_frequencies[word_text]
            magnitude = (gematria * frequency) / max_gematria if max_gematria else 0
            self.overall_magnitudes[word_text] = magnitude
        
        # Create the combined analysis dictionary
        self.WORD_FREQ_MAG_GEM = {
            word: (self.overall_gematria_values[word], self.overall_frequencies[word], self.overall_magnitudes[word])
            for word in self.overall_frequencies
        }
    
    def get_encoded_poem(self) -> Dict[str, Tuple[int, int, float]]:
        """Return the encoded poem analysis."""
        return self.WORD_FREQ_MAG_GEM.copy()
    
    def display_summary(self):
        """Display a summary of the poem."""
        print(f"Poem with {self.phrase_count} phrases and {self.word_count_overall} words.")
    
    def display_phrases(self):
        """Display all phrases in the poem."""
        for phrase in self.phrases:
            print(phrase.text)
            print()
    
    def display_detailed_analysis(self):
        """Display detailed phrase-by-phrase analysis."""
        print("Detailed Phrase Information:")
        for phrase in self.phrases:
            phrase.display()
    
    def display_whole_poem(self):
        """Display the formatted whole poem with indentation."""
        print("Whole Poem:")
        print("=" * 50)
        for i, phrase_text in enumerate(self.phrase_texts, 1):
            # Create varying indentation levels
            indent = "  " * ((i - 1) % 4)
            print(f"{indent}{phrase_text}")
        print("=" * 50)


class PoemGenerator:
    """Main class for generating poems with file I/O and user interaction."""
    
    @staticmethod
    def base36_encode(number: int) -> str:
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
    
    @staticmethod
    def read_lines_from_file(filename: str = "phrases.txt") -> List[str]:
        """Read lines from a text file, returning empty list if file not found."""
        try:
            with open(filename, "r") as file:
                lines = [line.strip() for line in file if line.strip()]
            return lines
        except FileNotFoundError:
            return []
    
    @staticmethod
    def generate_interactive() -> Tuple[Poem, str]:
        """Generate a poem with interactive user input and file output."""
        # Get filename from user
        file_name = input("Enter the phrase filename (default: phrases.txt): ") or "phrases.txt"
        
        # Read phrases from file
        phrases = PoemGenerator.read_lines_from_file(file_name)
        
        # Fallback if file is empty or not found
        if not phrases:
            phrases = ["From pain becomes pleasure To suspension is found at order While lasts chaos becomes system At the end a force made by bending is The Foundation"]
        
        # Handle single string vs list
        if isinstance(phrases, str):
            phrases = phrases.split('\n')
        
        # Create poem
        poem = Poem(phrases)
        
        # Display initial summary
        print("\nGenerated Poem:\n")
        poem.display_summary()
        
        # Wait for user input
        try:
            input("\nPress Enter to see detailed phrase information...\n")
        except EOFError:
            print("\nPress Enter to see detailed phrase information...")
        
        # Display detailed analysis
        poem.display_detailed_analysis()
        
        # Display formatted whole poem
        print()
        poem.display_whole_poem()
        
        # Save to file
        return PoemGenerator._save_poem_to_file(poem, phrases)
    
    @staticmethod
    def _save_poem_to_file(poem: Poem, original_phrases: List[str]) -> Tuple[Poem, str]:
        """Save the poem analysis to a timestamped file."""
        # Generate timestamp and filename
        timestamp = int(time.time())
        timestamp_str = PoemGenerator.base36_encode(timestamp)
        
        last_phrase = original_phrases[-1] if original_phrases else "No_phrases_generated"
        filename = f"{last_phrase.replace(' ', '_')}_{timestamp_str}.txt"
        
        print(f"\nSaving poem to {filename}...")
        
        with open(filename, "w") as file:
            # Shuffle phrases for output variety
            shuffled_phrases = original_phrases.copy()
            random.seed(time.time())
            random.shuffle(shuffled_phrases)
            
            # Write header information
            file.write(f"Poem with {poem.phrase_count} phrases and {poem.word_count_overall} words.\n")
            file.write(f"Timestamp: {timestamp_str}\n")
            file.write(f"Last Phrase: {last_phrase}\n")
            
            # Write generated phrases
            file.write("\nGenerated Phrases:\n")
            for phrase in shuffled_phrases:
                file.write(f"{phrase}\n")
            
            # Write overall word powers
            file.write("\nOverall Poem Word Powers:\n")
            for word, (gem, freq, mag) in poem.WORD_FREQ_MAG_GEM.items():
                file.write(f"{word}: Gematria={gem}, Frequency={freq}, Magnitude={mag}\n")
            
            # Write formatted whole poem
            file.write("\nWhole Poem:\n")
            file.write("=" * 50 + "\n")
            for i, phrase_text in enumerate(original_phrases, 1):
                indent = "  " * ((i - 1) % 4)
                file.write(f"{indent}{phrase_text}\n")
            file.write("=" * 50 + "\n")
            
            # Write footer
            file.write("\nThis poem was generated using advanced algorithms and natural language processing techniques.\n")
            file.write("It aims to capture the essence of human emotions and experiences through the art of poetry.\n")
            file.write("\nGenerated by Goddess Taivos and General Suora\n")
        
        print(f"Poem saved to {filename}")
        return poem, filename


def main():
    """Main function for interactive poem generation."""
    print("Poem Generator with Gematria Analysis")
    print("=" * 40)
    
    try:
        poem, filename = PoemGenerator.generate_interactive()
        print(f"\nGeneration complete! Poem saved as: {filename}")
        
    except KeyboardInterrupt:
        print("\nPoem generation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")


def example_programmatic_usage():
    """Example of how to use the classes programmatically."""
    print("Programmatic Usage Example:")
    print("-" * 30)
    
    # Create some example phrases
    phrases = [
        "Taivos is chaos incarnate",
        "Suora brings order to all",
        "Balance is found between extremes"
    ]
    
    # Create a poem
    poem = Poem(phrases)
    
    # Display analysis
    poem.display_summary()
    print()
    
    # Show individual words
    word_example = Word("chaos")
    print(f"Word example: {word_example} (gematria: {word_example.gematria_value})")
    
    # Show phrase analysis
    phrase_example = Phrase("Taivos is chaos incarnate")
    print(f"Phrase analysis: {phrase_example.get_word_analysis()}")
    
    # Show whole poem
    poem.display_whole_poem()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        example_programmatic_usage()
    else:
        main()
