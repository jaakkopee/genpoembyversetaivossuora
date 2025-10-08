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
import sys
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import re


class Colors:
    """ANSI color codes for terminal output."""
    
    # Reset
    RESET = '\033[0m'
    
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    
    @staticmethod
    def colorize(text: str, color: str) -> str:
        """Apply color to text."""
        return f"{color}{text}{Colors.RESET}"
    
    @staticmethod
    def header(text: str) -> str:
        """Format text as a colorful header."""
        return f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{text}{Colors.RESET}"
    
    @staticmethod
    def success(text: str) -> str:
        """Format text as a success message."""
        return f"{Colors.BRIGHT_GREEN}{text}{Colors.RESET}"
    
    @staticmethod
    def warning(text: str) -> str:
        """Format text as a warning message."""
        return f"{Colors.BRIGHT_YELLOW}{text}{Colors.RESET}"
    
    @staticmethod
    def error(text: str) -> str:
        """Format text as an error message."""
        return f"{Colors.BRIGHT_RED}{text}{Colors.RESET}"
    
    @staticmethod
    def info(text: str) -> str:
        """Format text as an info message."""
        return f"{Colors.BRIGHT_BLUE}{text}{Colors.RESET}"
    
    @staticmethod
    def highlight(text: str) -> str:
        """Format text as highlighted."""
        return f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}{text}{Colors.RESET}"
    
    @staticmethod
    def dim(text: str) -> str:
        """Format text as dimmed."""
        return f"{Colors.DIM}{text}{Colors.RESET}"
    
    @staticmethod
    def phrase(text: str) -> str:
        """Format text as a phrase (magenta)."""
        return f"{Colors.BRIGHT_MAGENTA}{text}{Colors.RESET}"
    
    @staticmethod
    def word(text: str) -> str:
        """Format text as a word (cyan)."""
        return f"{Colors.CYAN}{text}{Colors.RESET}"
    
    @staticmethod
    def number(text: str) -> str:
        """Format text as a number (green)."""
        return f"{Colors.GREEN}{text}{Colors.RESET}"


class IterativeLoader:
    """An iteration-based loading animation system that updates per operation."""
    
    def __init__(self, message: str, style: str = "spinner"):
        """Initialize the loader with a message and style.
        
        Args:
            message: The loading message to display
            style: Animation style - 'spinner', 'dots', 'bars', 'pulse'
        """
        self.message = message
        self.style = style
        self.iteration = 0
        self.started = False
        
        # Define animation frames for different styles
        if style == "spinner":
            self.frames = "|/-\\"
        elif style == "dots":
            self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        elif style == "bars":
            self.frames = ["▱▱▱", "▰▱▱", "▰▰▱", "▰▰▰", "▱▰▰", "▱▱▰"]
        elif style == "pulse":
            self.frames = ["●", "◐", "◑", "◒", "◓", "◔", "◕", "○"]
        elif style == "arrow":
            self.frames = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]
        else:
            self.frames = "|/-\\"
    
    def start(self):
        """Start the loading animation."""
        if not self.started:
            print()
            print(f"{Colors.info(self.message)} ", end="", flush=True)
            self.started = True
            self.update()
    
    def update(self):
        """Update the animation to the next frame."""
        if self.started:
            # Clear previous frame
            if self.iteration > 0:
                prev_frame = self.frames[(self.iteration - 1) % len(self.frames)]
                if len(prev_frame) > 1:
                    print(f"\033[{len(prev_frame)}D" + " " * len(prev_frame) + f"\033[{len(prev_frame)}D", end="", flush=True)
                else:
                    print("\b \b", end="", flush=True)
            
            # Show current frame
            current_frame = self.frames[self.iteration % len(self.frames)]
            print(current_frame, end="", flush=True)
            self.iteration += 1
    
    def finish(self, success: bool = True):
        """Finish the loading animation with success or failure indicator."""
        if self.started:
            # Clear the last frame
            if self.iteration > 0:
                last_frame = self.frames[(self.iteration - 1) % len(self.frames)]
                if len(last_frame) > 1:
                    print(f"\033[{len(last_frame)}D" + " " * len(last_frame) + f"\033[{len(last_frame)}D", end="", flush=True)
                else:
                    print("\b \b", end="", flush=True)
            
            # Show completion indicator
            if success:
                print(Colors.success("✓"))
            else:
                print(Colors.error("✗"))
            
            self.started = False


def simple_loading_animation(message: str, iterations: int, style: str = "bars"):
    """Simple function for loading animation over a known number of iterations.
    
    Args:
        message: The loading message
        iterations: Total number of iterations to animate over
        style: Animation style
    
    Returns:
        Generator that yields progress updates
    """
    loader = IterativeLoader(message, style)
    loader.start()
    
    for i in range(iterations):
        if i % max(1, iterations // 20) == 0:  # Update every 5% of progress
            loader.update()
        yield i
    
    loader.finish()


class Word:
    """Represents a single word with its gematria and analysis properties."""
    
    # Scandinavian extended gematria values
    GEMATRIA_VALUES = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
        'K': 20, 'L': 30, 'M': 40, 'N': 50, 'O': 60, 'P': 70, 'Q': 80, 'R': 90, 'S': 100, 'T': 200,
        'U': 300, 'V': 400, 'W': 500, 'X': 600, 'Y': 700, 'Z': 800, 'Å': 900, 'Ä': 1000, 'Ö': 1100,
        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
        'k': 20, 'l': 30, 'm': 40, 'n': 50, 'o': 60, 'p': 70, 'q': 80, 'r': 90, 's': 100, 't': 200,
        'u': 300, 'v': 400, 'w': 500, 'x': 600, 'y': 700, 'z': 800, 'å': 900, 'ä': 1000, 'ö': 1100
    }
    
    def __init__(self, text: str, phrase_context: str = "", phrase_power: float = 0.0):
        """Initialize a Word with its text and optional phrase context."""
        self.original_text = text
        self.cleaned_text = self._clean_text(text)
        self.gematria_value = self._calculate_gematria()
        self.length = len(self.cleaned_text)
        
        # Word class (part of speech) information
        self.word_class = self._determine_word_class(self.cleaned_text)
        self.class_multiplier = self._get_class_multiplier(self.word_class)
        
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
    
    def _determine_word_class(self, text: str) -> str:
        """Determine the word class (part of speech) using simple rule-based classification."""
        if not text or len(text) == 0:
            return "unknown"
        
        # Convert to lowercase for analysis
        word = text.lower().strip()
        
        # Numbers
        if word.isdigit():
            return "num"
            
        # Common pronouns (closed class)
        pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                   'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
                   'this', 'that', 'these', 'those', 'who', 'whom', 'what', 'which', 'whose'}
        if word in pronouns:
            return "pron"
            
        # Common prepositions (closed class)
        prepositions = {'in', 'on', 'at', 'by', 'for', 'with', 'without', 'to', 'from', 'of', 'about',
                       'above', 'below', 'under', 'over', 'through', 'between', 'among', 'during',
                       'before', 'after', 'since', 'until', 'within', 'across', 'along', 'around'}
        if word in prepositions:
            return "prep"
            
        # Common conjunctions (closed class)
        conjunctions = {'and', 'or', 'but', 'yet', 'so', 'nor', 'for', 'because', 'since', 'although',
                       'though', 'while', 'whereas', 'if', 'unless', 'until', 'when', 'where', 'why', 'how'}
        if word in conjunctions:
            return "conj"
            
        # Common articles and determiners (closed class)
        determiners = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'any', 'all', 'every',
                      'each', 'either', 'neither', 'both', 'much', 'many', 'few', 'little', 'several'}
        if word in determiners:
            return "det"
            
        # Common auxiliary verbs (closed class)
        auxiliaries = {'be', 'is', 'am', 'are', 'was', 'were', 'being', 'been',
                      'have', 'has', 'had', 'having',
                      'do', 'does', 'did', 'doing',
                      'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'ought'}
        if word in auxiliaries:
            return "aux"
            
        # Common adverbs (often end in -ly, but not always)
        if (word.endswith('ly') and len(word) > 3 and 
            not word.endswith('ily') and not word.endswith('ally')):
            return "adv"
            
        # Common adverbs that don't end in -ly
        common_adverbs = {'very', 'quite', 'rather', 'too', 'so', 'more', 'most', 'less', 'least',
                         'now', 'then', 'here', 'there', 'where', 'when', 'how', 'why',
                         'always', 'never', 'sometimes', 'often', 'usually', 'rarely',
                         'yes', 'no', 'not', 'maybe', 'perhaps', 'probably'}
        if word in common_adverbs:
            return "adv"
            
        # Adjective patterns
        if (word.endswith('ful') or word.endswith('less') or word.endswith('ous') or 
            word.endswith('ive') or word.endswith('able') or word.endswith('ible') or
            word.endswith('ant') or word.endswith('ent') or word.endswith('ing') or
            word.endswith('ed') and not word.endswith('eed')):
            return "adj"
            
        # Verb patterns (past tense, present participle, etc.)
        if (word.endswith('ed') or word.endswith('ing') or word.endswith('en') or
            word.endswith('s') and len(word) > 2):
            # Could be verb, but need more context - default to verb for now
            return "v"
            
        # Noun patterns (plurals, abstract nouns, etc.)
        if (word.endswith('s') and len(word) > 2 or
            word.endswith('tion') or word.endswith('sion') or word.endswith('ment') or
            word.endswith('ness') or word.endswith('ity') or word.endswith('ism') or
            word.endswith('er') or word.endswith('or') or word.endswith('ist')):
            return "n"
            
        # Default classification based on common patterns
        # Short words (1-2 chars) are often function words
        if len(word) <= 2:
            return "func"
            
        # Medium length words are often nouns or verbs
        # Default to noun for unknown words (most common class)
        return "n"
    
    def _get_class_multiplier(self, word_class: str) -> float:
        """Get a multiplier for word power based on word class."""
        # Different word classes have different semantic weights
        # Content words (nouns, verbs, adjectives, adverbs) are more semantically important
        # Function words (prepositions, conjunctions, etc.) are less important
        
        multipliers = {
            'n': 1.2,      # Nouns - high semantic content
            'v': 1.1,      # Verbs - high semantic content  
            'adj': 1.0,    # Adjectives - moderate semantic content
            'adv': 0.9,    # Adverbs - moderate semantic content
            'pron': 0.7,   # Pronouns - low semantic content
            'prep': 0.6,   # Prepositions - low semantic content
            'conj': 0.5,   # Conjunctions - low semantic content
            'det': 0.5,    # Determiners - low semantic content
            'aux': 0.6,    # Auxiliary verbs - low semantic content
            'num': 0.8,    # Numbers - moderate semantic content
            'func': 0.4,   # Function words - very low semantic content
            'unknown': 0.8  # Unknown - moderate default
        }
        
        return multipliers.get(word_class, 0.8)
    
    def get_enhanced_power(self) -> float:
        """Calculate enhanced word power including class multiplier."""
        return self.phrase_power * self.class_multiplier
    
    def __str__(self) -> str:
        return self.original_text
    
    def __repr__(self) -> str:
        return f"Word('{self.original_text}', class={self.word_class}, gematria={self.gematria_value}, power={self.phrase_power:.3f})"


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
            # Show animated loading while reading file
            loader = IterativeLoader(f"Reading {self.filename}", "bars")
            loader.start()
            
            with open(self.filename, "r", encoding="utf-8", errors="ignore") as file:
                lines = []
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    
                    # Update animation every 100 lines
                    if line_num % 100 == 0:
                        loader.update()
                    
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
                loader.finish(True)
                print(Colors.success(f"Loaded {len(self.phrases)} meaningful phrases from {self.filename}"))
                
        except FileNotFoundError:
            loader.finish(False)
            print(Colors.warning(f"Warning: {self.filename} not found. Using empty database."))
            self.phrases = []
    
    def _analyze_phrases(self):
        """Analyze each phrase and calculate word powers within their contexts."""
        # Show animated loading while analyzing
        loader = IterativeLoader("Analyzing phrase contexts and word powers", "bars")
        loader.start()
        
        for i, phrase_text in enumerate(self.phrases):
            # Update animation every 50 phrases
            if i % 50 == 0:
                loader.update()
                
            # Create a temporary Phrase object to get word analysis
            temp_phrase = Phrase(phrase_text)
            analysis = temp_phrase.get_word_analysis()
            
            # Store word powers with their phrase contexts including word class info
            for word, (gematria, frequency, magnitude) in analysis.items():
                power = frequency * magnitude
                
                # Get word class information from the temporary phrase
                word_obj = next((w for w in temp_phrase.words if w.cleaned_text == word), None)
                word_class = word_obj.word_class if word_obj else "unknown"
                class_multiplier = word_obj.class_multiplier if word_obj else 1.0
                
                self.word_powers[word.lower()].append({
                    'power': power,
                    'phrase': phrase_text,
                    'frequency': frequency,
                    'magnitude': magnitude,
                    'gematria': gematria,
                    'word_class': word_class,
                    'class_multiplier': class_multiplier
                })
        
        loader.finish(True)
    
    def _calculate_alternatives(self):
        """Calculate word alternatives based on similar power levels."""
        # Show animated loading while calculating alternatives
        loader = IterativeLoader("Calculating word alternatives based on power similarity", "bars")
        loader.start()
        
        # Group words by power ranges for alternatives
        power_ranges = defaultdict(list)
        
        word_list = list(self.word_powers.items())
        for i, (word, power_list) in enumerate(word_list):
            # Update animation every 25 words (more frequent updates)
            if i % 25 == 0:
                loader.update()
                
            # Only consider valid words that appear in phrases AND pass validation
            if power_list and word and len(word) >= 1 and self._is_valid_alternative(word):
                avg_power = sum(entry['power'] for entry in power_list) / len(power_list)
                power_range = int(avg_power * 10)  # Group by tenths
                power_ranges[power_range].append((word, avg_power))
        
        loader.update()
        
        # Create alternative mappings
        power_range_list = list(power_ranges.items())
        for i, (power_range, word_list) in enumerate(power_range_list):
            # Update animation every 3 power ranges (more frequent updates)
            if i % 10 == 0:
                loader.update()
                
            if len(word_list) > 1:  # Only create alternatives if multiple words exist
                # Filter the word_list to only include valid alternatives
                valid_words = [(w, p) for w, p in word_list if self._is_valid_alternative(w)]
                
                if len(valid_words) > 1:  # Ensure we still have multiple words after filtering
                    for j, (word, power) in enumerate(valid_words):
                        # Update animation more frequently during intensive processing
                        if j % 10 == 0:
                            loader.update()
                            
                        # Additional validation: clean alternatives to ensure no punctuation gets through
                        alternatives = set()
                        for alt_word, alt_power in valid_words:
                            if alt_word != word and abs(alt_power - power) < 0.5:
                                # Double-check that alternative is clean and valid
                                if self._is_valid_alternative(alt_word) and not any(c in alt_word for c in "':\".,!?;()[]{}"):
                                    alternatives.add(alt_word)
                        
                        if alternatives:  # Only store if we have actual alternatives
                            self.word_alternatives[word] = alternatives
        
        loader.finish(True)
    
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
        # Also collect word class information
        word_class_info = {}
        for word in self.words:
            if word.cleaned_text:  # Only process valid words
                self.frequencies[word.cleaned_text] += 1
                self.gematria_values[word.cleaned_text] = word.gematria_value
                # Store word class info for the first occurrence
                if word.cleaned_text not in word_class_info:
                    word_class_info[word.cleaned_text] = {
                        'class': word.word_class,
                        'multiplier': word.class_multiplier
                    }
        
        # Calculate max gematria for magnitude normalization
        max_gematria = max(self.gematria_values.values()) if self.gematria_values else 1
        
        # Calculate magnitudes and word powers (enhanced with word class)
        for word_text in self.frequencies:
            gematria = self.gematria_values[word_text]
            frequency = self.frequencies[word_text]
            magnitude = (gematria * frequency) / max_gematria if max_gematria else 0
            
            # Apply word class multiplier to enhance semantic importance
            class_multiplier = word_class_info.get(word_text, {}).get('multiplier', 1.0)
            enhanced_magnitude = magnitude * class_multiplier
            
            self.magnitudes[word_text] = enhanced_magnitude
            self.word_powers[word_text] = frequency * enhanced_magnitude
    
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
        """Display detailed phrase information including word class and alternatives if available."""
        analysis = self.get_word_analysis()
        
        # Create word power string with class information
        word_info_parts = []
        for word, (gem, freq, mag) in analysis.items():
            # Find the word object to get class info
            word_obj = next((w for w in self.words if w.cleaned_text == word), None)
            if word_obj:
                class_info = f"{word_obj.word_class}:{word_obj.class_multiplier:.1f}"
                word_info_parts.append(f"'{word}': ({gem}, {freq}, {mag:.3f}, {class_info})")
            else:
                word_info_parts.append(f"'{word}': ({gem}, {freq}, {mag:.3f})")
        
        word_power_str = ', '.join(word_info_parts)
        print(f"{Colors.dim('Word Power:')} {{{word_power_str}}}")
        
        # Show alternatives if database is available
        if self.large_db:
            alternative_phrase = self.get_enhanced_alternatives(0.8)  # High aggressiveness for display
            if alternative_phrase != self.text:
                print(f"{Colors.highlight('Alternative:')} {Colors.phrase(alternative_phrase)}")
            else:
                # Try with maximum aggressiveness if no changes
                max_alt = self.get_enhanced_alternatives(1.0)
                if max_alt != self.text:
                    print(f"{Colors.highlight('Max Alternative:')} {Colors.phrase(max_alt)}")
    
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
        
        # Show progress bar for generating alternatives
        loader = IterativeLoader(f"Generating alternative version ({int(aggressiveness*100)}% aggressiveness)", "bars")
        loader.start()
        
        alternative_phrases = []
        for i, phrase in enumerate(self.phrases):
            # Update progress bar every phrase (since phrase count is usually small)
            if i % 8 == 0:
                loader.update()
                
            alt_phrase = phrase.get_enhanced_alternatives(aggressiveness)
            alternative_phrases.append(alt_phrase)
        
        loader.finish(True)
        return EnhancedPoem(alternative_phrases, self.large_db)
    
    def display_summary(self):
        """Display a summary of the poem with database info."""
        print(Colors.info(f"Enhanced Poem with {Colors.number(str(self.phrase_count))} phrases and {Colors.number(str(self.word_count_overall))} words."))
        if self.large_db:
            print(Colors.info(f"Database: {Colors.number(str(len(self.large_db.phrases)))} reference phrases loaded."))
    
    def display_phrases(self):
        """Display all phrases in the poem."""
        for phrase in self.phrases:
            print(phrase.text)
            print()
    
    def display_detailed_analysis(self):
        """Display detailed phrase-by-phrase analysis with alternatives."""
        print(Colors.header("Detailed Phrase Information:"))
        for i, phrase in enumerate(self.phrases, 1):
            print(f"{Colors.highlight(f'Phrase {i}:')}")
            phrase.display()
            print()
    
    def display_whole_poem(self):
        """Display the formatted whole poem with indentation."""
        print(Colors.header("Whole Poem:"))
        print(Colors.dim("=" * 50))
        phrase_count = 0
        for phrase_text in self.phrase_texts:
            if phrase_text.strip():  # Only display non-empty phrases
                phrase_count += 1
                # Create varying indentation levels
                indent = "  " * ((phrase_count - 1) % 4)
                print(f"{indent}{Colors.phrase(phrase_text)}")
        print(Colors.dim("=" * 50))


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
        import os
        
        # Show available text files in current directory
        txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
        if txt_files:
            print(Colors.info("Available text files in current directory:"))
            for i, file in enumerate(txt_files[:32], 1):  # Show first 32 files
                print(f"  {Colors.number(str(i))}. {Colors.dim(file)}")
            if len(txt_files) > 32:
                print(Colors.dim(f"  ... and {len(txt_files) - 32} more files"))
            print()
        
        # Get large phrase database filename from user
        large_db_filename = input(f"\n{Colors.highlight('Enter the large phrase database filename')} (default: large_phrases.txt): ") or "large_phrases.txt"
        
        # Initialize large phrase database with animated loading
        print()  # Add line break before loader starts
        loader = IterativeLoader(f"Initializing large phrase database from {large_db_filename}", "bars")
        loader.start()
        large_db = LargePhraseDatabase(large_db_filename)
        loader.finish(True)
        print()
        
        # Get filename from user
        file_name = input(f"\n{Colors.highlight('Enter the phrase filename')} (default: phrases.txt): ") or "phrases.txt"
        print()  # Add line break after input
        
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
        print(f"\n{Colors.header('Generated Enhanced Poem:')}\n")
        poem.display_summary()
        
        # Wait for user input
        try:
            input(f"\n{Colors.highlight('Press Enter to see detailed phrase information...')}\n")
        except EOFError:
            print(f"\n{Colors.highlight('Press Enter to see detailed phrase information...')}")
        
        # Display detailed analysis
        poem.display_detailed_analysis()
        
        # Display formatted whole poem
        poem.display_whole_poem()
        
        # Ask if user wants to see alternative version
        try:
            choice = input(f"\n{Colors.highlight('Generate alternative version using word alternatives?')} (y/N): ").lower()
        except EOFError:
            choice = 'n'
            
        if choice.startswith('y'):
            print(f"\n{Colors.info('Generating alternative versions...')}")
            
            # Moderate alternatives
            alt_poem_moderate = poem.generate_alternative_version(0.5)
            print(f"\n{Colors.header('Moderate Alternative (50% substitution):')}")
            alt_poem_moderate.display_whole_poem()
            
            # Aggressive alternatives
            alt_poem_aggressive = poem.generate_alternative_version(0.9)
            print(f"\n{Colors.header('Aggressive Alternative (90% substitution):')}")
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
        
        print(Colors.info(f"Saving enhanced poem to {filename}..."))
        
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
        
        print(Colors.success(f"Enhanced poem saved to {filename}"))
        return poem, filename


def main():
    """Main function for interactive enhanced poem generation."""
    print(Colors.header("Enhanced Poem Generator with Large Phrase Database"))
    print(Colors.dim("=" * 50))
    
    try:
        poem, filename = EnhancedPoemGenerator.generate_interactive()
        print(Colors.success(f"\nGeneration complete! Enhanced poem saved as: {filename}"))
        
    except KeyboardInterrupt:
        print(Colors.warning("\nPoem generation cancelled by user."))
    except Exception as e:
        print(Colors.error(f"An error occurred: {e}"))
        import traceback
        traceback.print_exc()


def example_programmatic_usage():
    """Example of how to use the enhanced classes programmatically."""
    print(Colors.header("Enhanced Programmatic Usage Example:"))
    print(Colors.dim("-" * 40))
    
    # Load large phrase database
    loader = IterativeLoader("Loading database", "bars")
    loader.start()
    large_db = LargePhraseDatabase()
    loader.finish(True)
    
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
    print(f"{Colors.info('Enhanced word example:')} {Colors.word(str(enhanced_word))}")
    if enhanced_word.phrase_context:
        print(f"  {Colors.dim('Context phrase:')} {enhanced_word.phrase_context[:80]}...")
        print(f"  {Colors.dim('Power in context:')} {Colors.number(f'{enhanced_word.phrase_power:.3f}')}")
    print()
    
    # Show alternatives
    alternatives = large_db.get_alternatives("chaos", 5)
    if alternatives:
        print(f"{Colors.info('Alternatives for')} {Colors.word('chaos')}: {Colors.phrase(str(alternatives))}")
    print()
    
    # Show whole poem and alternatives with different aggressiveness
    poem.display_whole_poem()
    print()
    
    # Show moderate alternatives
    alt_poem_moderate = poem.generate_alternative_version(0.5)
    print(Colors.header("Moderate alternative version (50% substitution):"))
    alt_poem_moderate.display_whole_poem()
    print()
    
    # Show aggressive alternatives  
    alt_poem_aggressive = poem.generate_alternative_version(0.9)
    print(Colors.header("Aggressive alternative version (90% substitution):"))
    alt_poem_aggressive.display_whole_poem()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        example_programmatic_usage()
    else:
        main()