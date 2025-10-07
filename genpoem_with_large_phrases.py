#!/usr/bin/env python3
"""
Advanced Poem Generation Algorithm with Large Phrase Database
Author: Goddess Taivos and General Suora
Date: 26. syyskuuta 2025

This module extends the poem generation algorithm by incorporating a large database
of phrases. It calculates word powers within their original phrase contexts and
uses these encoded words as intelligent alternatives for poem generation.
"""

import time
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import re


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
    
    def __init__(self, text: str, phrase_context: str = "", phrase_power: float = 0.0):
        """Initialize a Word with its text and optional phrase context."""
        self.original_text = text
        self.cleaned_text = self._clean_text(text)
        self.gematria_value = self._calculate_gematria()
        self.length = len(self.cleaned_text)
        
        # Context from large phrase database
        self.phrase_context = phrase_context
        self.phrase_power = phrase_power
        
    def _clean_text(self, text: str) -> str:
        """Remove punctuation from word for gematria calculation."""
        import re
        # Define comprehensive punctuation to remove
        punctuation_to_remove = [
            '\'', '"', ':', '¬', '†', '‡', ''', ''', '"', '"', '–', '—',
            '`', '~', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', 
            '/', '\\', '|', '_', '.', ',', '!', '?', ';', '(', ')', 
            '[', ']', '{', '}', '-'
        ]
        
        # Remove the specified punctuation
        cleaned = text
        for punct in punctuation_to_remove:
            cleaned = cleaned.replace(punct, '')
        
        # Remove any remaining non-alphanumeric characters except spaces
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        
        # Clean up multiple spaces and strip
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Return empty string if no valid content (allow single chars and mixed alphanumeric)
        if len(cleaned) < 1 or (not any(c.isalpha() for c in cleaned) and not cleaned.isdigit()):
            return ""
        return cleaned.lower()
    
    def _calculate_gematria(self) -> int:
        """Calculate the gematria value using Scandinavian extended values or numeric value for numbers."""
        # If the cleaned text is a pure number, use its numeric value
        if self.cleaned_text.isdigit():
            return int(self.cleaned_text)
        # Otherwise use traditional gematria
        return sum(self.GEMATRIA_VALUES.get(char, 0) for char in self.cleaned_text)
    
    def __str__(self) -> str:
        return self.original_text
    
    def __repr__(self) -> str:
        return f"Word('{self.original_text}', gematria={self.gematria_value}, power={self.phrase_power:.3f})"


class LargePhraseDatabase:
    """Manages the large phrase database and word power calculations."""
    
    def __init__(self, filename: str = "large_phrases.txt"):
        """Initialize the database by reading and analyzing the large phrases file."""
        self.filename = filename
        self.phrases = []
        self.word_powers = defaultdict(list)  # word -> list of (power, phrase, frequency, magnitude)
        self.word_alternatives = defaultdict(set)  # word -> set of similar power words
        
        self._load_phrases()
        self._analyze_phrases()
        self._calculate_alternatives()
    
    def _load_phrases(self):
        """Load phrases from the large file, filtering for meaningful content."""
        try:
            with open(self.filename, "r", encoding="utf-8", errors="ignore") as file:
                lines = []
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    
                    # Filter for meaningful phrases (sentences with reasonable length)
                    if (line and 
                        len(line) > 15 and 
                        len(line) < 200 and
                        not line.isupper() and  # Skip all caps headers
                        not line.isdigit() and  # Skip page numbers
                        not re.match(r'^[^a-zA-Z]*$', line) and  # Skip non-alphabetic lines
                        ' ' in line):  # Must contain spaces (multiple words)
                        
                        lines.append(line)
                        if len(lines) >= 10000:  # Limit for performance
                            break
                
                self.phrases = lines
                print(f"Loaded {len(self.phrases)} meaningful phrases from {self.filename}")
                
        except FileNotFoundError:
            print(f"Warning: {self.filename} not found. Using empty database.")
            self.phrases = []
    
    def _analyze_phrases(self):
        """Analyze each phrase and calculate word powers within their contexts."""
        print("Analyzing phrase contexts and word powers...")
        
        for phrase_text in self.phrases:
            # Create a temporary Phrase object to get word analysis
            temp_phrase = Phrase(phrase_text)
            analysis = temp_phrase.get_word_analysis()
            
            # Store word powers with their phrase contexts
            for word, (gematria, frequency, magnitude) in analysis.items():
                power = frequency * magnitude
                self.word_powers[word.lower()].append({
                    'power': power,
                    'phrase': phrase_text,
                    'frequency': frequency,
                    'magnitude': magnitude,
                    'gematria': gematria
                })
    
    def _calculate_alternatives(self):
        """Calculate word alternatives based on similar power levels."""
        print("Calculating word alternatives based on power similarity...")
        
        # Group words by power ranges for alternatives
        power_ranges = defaultdict(list)
        
        for word, power_list in self.word_powers.items():
            # Only consider valid words that appear in phrases AND pass validation
            if power_list and word and len(word) >= 1 and self._is_valid_alternative(word):
                avg_power = sum(entry['power'] for entry in power_list) / len(power_list)
                power_range = int(avg_power * 10)  # Group by tenths
                power_ranges[power_range].append((word, avg_power))
        
        # Create alternative mappings
        for power_range, word_list in power_ranges.items():
            if len(word_list) > 1:  # Only create alternatives if multiple words exist
                # Filter the word_list to only include valid alternatives
                valid_words = [(w, p) for w, p in word_list if self._is_valid_alternative(w)]
                
                if len(valid_words) > 1:  # Ensure we still have multiple words after filtering
                    for word, power in valid_words:
                        # Additional validation: clean alternatives to ensure no punctuation gets through
                        alternatives = set()
                        for alt_word, alt_power in valid_words:
                            if alt_word != word and abs(alt_power - power) < 0.5:
                                # Double-check that alternative is clean and valid
                                if self._is_valid_alternative(alt_word) and not any(c in alt_word for c in "':\".,!?;()[]{}"):
                                    alternatives.add(alt_word)
                        
                        if alternatives:  # Only store if we have actual alternatives
                            self.word_alternatives[word] = alternatives
    
    def _is_valid_alternative(self, word: str) -> bool:
        """Check if a word is a valid alternative (no quotes, special chars, etc.)."""
        if not word or len(word) < 1:  # Allow single characters
            return False
        # Comprehensive list of problematic punctuation and special characters
        problematic_chars = [
            '\'', '"', ':', '¬', '†', '‡', ''', ''', '"', '"', '–', '—',
            '`', '~', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', 
            '/', '\\', '|', '_', '.', ',', '!', '?', ';', '(', ')', 
            '[', ']', '{', '}', '-'
        ]
        if any(char in word for char in problematic_chars):
            return False
        # Must contain at least one letter OR be a valid number (allow mixed alphanumeric)
        if not any(c.isalpha() for c in word) and not word.isdigit():
            return False
        return True

    def get_word_power_info(self, word: str) -> List[Dict]:
        """Get power information for a specific word."""
        return self.word_powers.get(word.lower(), [])
    
    def get_alternatives(self, word: str, count: int = 3) -> List[str]:
        """Get alternative words with similar power levels."""
        alternatives = self.word_alternatives.get(word.lower(), set())
        # Filter alternatives one more time to ensure no punctuation gets through
        clean_alternatives = [alt for alt in alternatives if self._is_valid_alternative(alt)]
        return clean_alternatives[:count]
    
    def get_enhanced_word(self, word_text: str) -> Word:
        """Create an enhanced Word object with phrase context information."""
        power_info = self.get_word_power_info(word_text)
        
        if power_info:
            # Use the most powerful context
            best_context = max(power_info, key=lambda x: x['power'])
            return Word(
                word_text,
                phrase_context=best_context['phrase'],
                phrase_power=best_context['power']
            )
        else:
            return Word(word_text)


class Phrase:
    """Represents a phrase containing multiple words with frequency and magnitude analysis."""
    
    def __init__(self, text: str, large_db: Optional[LargePhraseDatabase] = None):
        """Initialize a Phrase with its text and optional large database for enhancement."""
        self.text = text
        self.large_db = large_db
        
        # Create enhanced words if database is available, filtering out invalid words
        raw_words = text.split()
        if large_db:
            self.words = [large_db.get_enhanced_word(word_text) for word_text in raw_words]
        else:
            self.words = [Word(word_text) for word_text in raw_words]
        
        # Filter out words with empty cleaned_text (punctuation-only, too short, etc.)
        self.words = [word for word in self.words if word.cleaned_text]
            
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
        # Count frequencies and collect gematria values (skip empty cleaned_text)
        for word in self.words:
            if word.cleaned_text:  # Only process valid words
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
    
    def get_enhanced_alternatives(self, aggressiveness: float = 0.7) -> str:
        """Generate an alternative version of the phrase using word alternatives.
        
        Args:
            aggressiveness: Probability of using alternatives (0.0 to 1.0)
        """
        if not self.large_db:
            return self.text
        
        alternative_words = []
        changes_made = 0
        
        for word in self.words:
            alternatives = self.large_db.get_alternatives(word.cleaned_text, 3)  # Get more alternatives
            if alternatives and random.random() < aggressiveness:
                # Choose randomly from available alternatives and validate it
                chosen_alt = random.choice(alternatives)
                # Double-check the alternative is valid (safety net)
                if self.large_db._is_valid_alternative(chosen_alt):
                    alternative_words.append(chosen_alt)
                    changes_made += 1
                else:
                    alternative_words.append(word.original_text)
            else:
                alternative_words.append(word.original_text)
        
        # If no changes were made, force at least one change if alternatives exist
        if changes_made == 0 and aggressiveness > 0.5:
            for i, word in enumerate(self.words):
                alternatives = self.large_db.get_alternatives(word.cleaned_text, 3)
                if alternatives:
                    chosen_alt = random.choice(alternatives)
                    # Double-check the alternative is valid (safety net)
                    if self.large_db._is_valid_alternative(chosen_alt):
                        alternative_words[i] = chosen_alt
                        break
        
        return " ".join(alternative_words)
    
    def display(self):
        """Display detailed phrase information including alternatives if available."""
        analysis = self.get_word_analysis()
        word_power_str = ', '.join(f"'{word}': ({gem}, {freq}, {mag:.3f})" 
                                 for word, (gem, freq, mag) in analysis.items())
        print(f"Word Power: {{{word_power_str}}}")
        
        # Show alternatives if database is available
        if self.large_db:
            alternative_phrase = self.get_enhanced_alternatives(0.8)  # High aggressiveness for display
            if alternative_phrase != self.text:
                print(f"Alternative: {alternative_phrase}")
            else:
                # Try with maximum aggressiveness if no changes
                max_alt = self.get_enhanced_alternatives(1.0)
                if max_alt != self.text:
                    print(f"Max Alternative: {max_alt}")
    
    def __str__(self) -> str:
        return self.text


class EnhancedPoem:
    """Enhanced Poem class that integrates with the large phrase database."""
    
    def __init__(self, phrase_texts: List[str], large_db: Optional[LargePhraseDatabase] = None):
        """Initialize an Enhanced Poem with phrases and optional large database."""
        self.phrase_texts = phrase_texts.copy()
        self.large_db = large_db
        self.phrases = [Phrase(text, large_db) for text in phrase_texts]
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
    
    def generate_alternative_version(self, aggressiveness: float = 0.7) -> 'EnhancedPoem':
        """Generate an alternative version using word alternatives from the database.
        
        Args:
            aggressiveness: How often to use alternatives (0.0 to 1.0)
        """
        if not self.large_db:
            return self
        
        alternative_phrases = []
        for phrase in self.phrases:
            alt_phrase = phrase.get_enhanced_alternatives(aggressiveness)
            alternative_phrases.append(alt_phrase)
        
        return EnhancedPoem(alternative_phrases, self.large_db)
    
    def display_summary(self):
        """Display a summary of the poem with database info."""
        print(f"Enhanced Poem with {self.phrase_count} phrases and {self.word_count_overall} words.")
        if self.large_db:
            print(f"Database: {len(self.large_db.phrases)} reference phrases loaded.")
    
    def display_phrases(self):
        """Display all phrases in the poem."""
        for phrase in self.phrases:
            print(phrase.text)
            print()
    
    def display_detailed_analysis(self):
        """Display detailed phrase-by-phrase analysis with alternatives."""
        print("Detailed Phrase Information:")
        for i, phrase in enumerate(self.phrases, 1):
            print(f"Phrase {i}:")
            phrase.display()
            print()
    
    def display_whole_poem(self):
        """Display the formatted whole poem with indentation."""
        print("Whole Poem:")
        print("=" * 50)
        phrase_count = 0
        for phrase_text in self.phrase_texts:
            if phrase_text.strip():  # Only display non-empty phrases
                phrase_count += 1
                # Create varying indentation levels
                indent = "  " * ((phrase_count - 1) % 4)
                print(f"{indent}{phrase_text}")
        print("=" * 50)


class EnhancedPoemGenerator:
    """Enhanced poem generator with large phrase database integration."""
    
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
    def generate_interactive() -> Tuple[EnhancedPoem, str]:
        """Generate an enhanced poem with interactive user input and file output."""
        # Initialize large phrase database
        print("Loading large phrase database...")
        large_db = LargePhraseDatabase()
        
        # Get filename from user
        file_name = input("Enter the phrase filename (default: phrases.txt): ") or "phrases.txt"
        
        # Read phrases from file
        phrases = EnhancedPoemGenerator.read_lines_from_file(file_name)
        
        # Fallback if file is empty or not found
        if not phrases:
            phrases = ["From pain becomes pleasure To suspension is found at order While lasts chaos becomes system At the end a force made by bending is The Foundation"]
        
        # Handle single string vs list
        if isinstance(phrases, str):
            phrases = phrases.split('\n')
        
        # Create enhanced poem
        poem = EnhancedPoem(phrases, large_db)
        
        # Display initial summary
        print("\nGenerated Enhanced Poem:\n")
        poem.display_summary()
        
        # Wait for user input
        try:
            input("\nPress Enter to see detailed phrase information...\n")
        except EOFError:
            print("\nPress Enter to see detailed phrase information...")
        
        # Display detailed analysis
        poem.display_detailed_analysis()
        
        # Display formatted whole poem
        poem.display_whole_poem()
        
        # Ask if user wants to see alternative version
        try:
            choice = input("\nGenerate alternative version using word alternatives? (y/N): ").lower()
        except EOFError:
            choice = 'n'
            
        if choice.startswith('y'):
            print("\nGenerating alternative versions...")
            
            # Moderate alternatives
            alt_poem_moderate = poem.generate_alternative_version(0.5)
            print("\nModerate Alternative (50% substitution):")
            alt_poem_moderate.display_whole_poem()
            
            # Aggressive alternatives
            alt_poem_aggressive = poem.generate_alternative_version(0.9)
            print("\nAggressive Alternative (90% substitution):")
            alt_poem_aggressive.display_whole_poem()
        
        # Save to file
        return EnhancedPoemGenerator._save_poem_to_file(poem, phrases, large_db)
    
    @staticmethod
    def _save_poem_to_file(poem: EnhancedPoem, original_phrases: List[str], 
                          large_db: LargePhraseDatabase) -> Tuple[EnhancedPoem, str]:
        """Save the enhanced poem analysis to a timestamped file."""
        # Generate timestamp and filename
        timestamp = int(time.time())
        timestamp_str = EnhancedPoemGenerator.base36_encode(timestamp)
        
        last_phrase = original_phrases[-1] if original_phrases else "No_phrases_generated"
        filename = f"enhanced_{last_phrase.replace(' ', '_')}_{timestamp_str}.txt"
        
        print(f"\nSaving enhanced poem to {filename}...")
        
        with open(filename, "w") as file:
            # Shuffle phrases for output variety
            shuffled_phrases = original_phrases.copy()
            random.seed(time.time())
            random.shuffle(shuffled_phrases)
            
            # Write header information
            file.write(f"Enhanced Poem with {poem.phrase_count} phrases and {poem.word_count_overall} words.\n")
            file.write(f"Database: {len(large_db.phrases)} reference phrases analyzed.\n")
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
                
                # Add alternatives information
                alternatives = large_db.get_alternatives(word, 3)
                if alternatives:
                    file.write(f"  Alternatives: {', '.join(alternatives)}\n")
            
            # Write formatted whole poem
            file.write("\nWhole Poem:\n")
            file.write("=" * 50 + "\n")
            for i, phrase_text in enumerate(original_phrases, 1):
                indent = "  " * ((i - 1) % 4)
                file.write(f"{indent}{phrase_text}\n")
            file.write("=" * 50 + "\n")
            
            # Generate and write alternative versions with different aggressiveness levels
            alt_poem_moderate = poem.generate_alternative_version(0.5)
            alt_poem_aggressive = poem.generate_alternative_version(0.9)
            
            file.write("\nAlternative Version (moderate substitution - 50%):\n")
            file.write("=" * 50 + "\n")
            for i, phrase_text in enumerate(alt_poem_moderate.phrase_texts, 1):
                indent = "  " * ((i - 1) % 4)
                file.write(f"{indent}{phrase_text}\n")
            file.write("=" * 50 + "\n")
            
            file.write("\nAlternative Version (aggressive substitution - 90%):\n")
            file.write("=" * 50 + "\n")
            for i, phrase_text in enumerate(alt_poem_aggressive.phrase_texts, 1):
                indent = "  " * ((i - 1) % 4)
                file.write(f"{indent}{phrase_text}\n")
            file.write("=" * 50 + "\n")
            
            # Write footer
            file.write("\nThis enhanced poem was generated using advanced algorithms with a large phrase database.\n")
            file.write(f"Analysis based on {len(large_db.phrases)} reference phrases from literary corpus.\n")
            file.write("Word alternatives calculated based on contextual power similarity.\n")
            file.write("\nGenerated by Goddess Taivos and General Suora\n")
        
        print(f"Enhanced poem saved to {filename}")
        return poem, filename


def main():
    """Main function for interactive enhanced poem generation."""
    print("Enhanced Poem Generator with Large Phrase Database")
    print("=" * 50)
    
    try:
        poem, filename = EnhancedPoemGenerator.generate_interactive()
        print(f"\nGeneration complete! Enhanced poem saved as: {filename}")
        
    except KeyboardInterrupt:
        print("\nPoem generation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


def example_programmatic_usage():
    """Example of how to use the enhanced classes programmatically."""
    print("Enhanced Programmatic Usage Example:")
    print("-" * 40)
    
    # Load large phrase database
    print("Loading database...")
    large_db = LargePhraseDatabase()
    
    # Create some example phrases
    phrases = [
        "Taivos is chaos incarnate",
        "Suora brings order to all",
        "Balance is found between extremes"
    ]
    
    # Create an enhanced poem
    poem = EnhancedPoem(phrases, large_db)
    
    # Display analysis
    poem.display_summary()
    print()
    
    # Show enhanced word example
    enhanced_word = large_db.get_enhanced_word("chaos")
    print(f"Enhanced word example: {enhanced_word}")
    if enhanced_word.phrase_context:
        print(f"  Context phrase: {enhanced_word.phrase_context[:80]}...")
        print(f"  Power in context: {enhanced_word.phrase_power:.3f}")
    print()
    
    # Show alternatives
    alternatives = large_db.get_alternatives("chaos", 5)
    if alternatives:
        print(f"Alternatives for 'chaos': {alternatives}")
    print()
    
    # Show whole poem and alternatives with different aggressiveness
    poem.display_whole_poem()
    print()
    
    # Show moderate alternatives
    alt_poem_moderate = poem.generate_alternative_version(0.5)
    print("Moderate alternative version (50% substitution):")
    alt_poem_moderate.display_whole_poem()
    print()
    
    # Show aggressive alternatives  
    alt_poem_aggressive = poem.generate_alternative_version(0.9)
    print("Aggressive alternative version (90% substitution):")
    alt_poem_aggressive.display_whole_poem()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        example_programmatic_usage()
    else:
        main()