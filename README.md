# Enhanced Poem Generation with Gematria Analysis

A sophisticated poetry generation system that uses Scandinavian extended gematria calculations to analyze and create poems with deep numerical significance. The system incorporates contextual word analysis from large literary corpora and provides comprehensive user control over generation parameters.

## 🌟 Features

### Core Algorithms
- **Gematria Calculation**: Uses Scandinavian extended values (A=1, B=2, ..., Z=800, Å=900, Ä=1000, Ö=1100)
- **Word Power Analysis**: Calculates frequency, magnitude, and contextual power for each word
- **Part-of-Speech Classification**: Analyzes word classes (noun, verb, adjective, etc.) with configurable multipliers
- **Phrase Depth Indentation**: Creates visual poetry structure with hierarchical spacing
- **Timestamped Output**: Generates unique base36-encoded timestamps for each poem

### Advanced Capabilities
- **Large Corpus Integration**: Analyzes 10,000+ reference phrases from literary works
- **Contextual Alternatives**: Suggests word replacements based on similar gematria power
- **Configurable Parameters**: Four global parameters for complete generation control
- **Interactive Generation**: User-friendly interface with real-time parameter adjustment
- **Animated Progress**: Multi-style loading animations (spinner, dots, bars, pulse, arrow)
- **Colored Output**: ANSI terminal coloring for enhanced readability

### Global Configuration Parameters
- **Gematria Influence** (1.6x default): Controls weight of letter values in calculations
- **POS Influence** (1.0x default): Controls part-of-speech class significance 
- **Power Substitution** (0.5 default): Controls word replacement aggressiveness (0.0-1.0)
- **Word Alternative Count** (5 default): Controls number of alternatives generated per word

## 📁 Repository Structure

```
genpoembyversetaivossuora/
├── genpoem.py                          # Clean, modular poem generator
├── genpoem_with_large_phrases.py       # Enhanced version with corpus analysis
├── # Define the word categories for our alg.py  # Original algorithm (legacy)
├── phrases.txt                         # Sample input phrases
├── test_simple.txt                     # Simple test input
├── kataTonya.txt                       # Finnish poetry input
├── saima_harmaja.txt                   # Finnish literary corpus
├── large_phrases.txt                   # Large reference corpus (31k+ lines)
└── *.txt                              # Generated poem outputs
```

## 🚀 Quick Start

### Basic Usage

```bash
# Simple poem generation
python3 genpoem.py

# Enhanced generation with corpus analysis
python3 genpoem_with_large_phrases.py

# See example usage
python3 genpoem.py --example
python3 genpoem_with_large_phrases.py --example
```

### Creating Custom Input Files

Create a text file with one phrase per line:

```
Taivos is chaos incarnate
Suora brings order to all
Balance is found between extremes
```

## 🎨 Algorithm Classes

### Word Class
Represents individual words with gematria analysis:
- **Scandinavian Extended Gematria**: A=1, B=2, C=3, ..., K=20, L=30, ..., T=200, ..., Z=800, Å=900, Ä=1000, Ö=1100
- **Context Tracking**: Maintains phrase context and power calculations
- **Punctuation Handling**: Cleanly separates words from punctuation for analysis

### Phrase Class
Manages collections of words within phrases:
- **Frequency Analysis**: Counts word occurrences within the phrase
- **Magnitude Calculation**: Normalizes word power relative to maximum gematria
- **Alternative Generation**: Creates variations using corpus-derived alternatives

### Poem Class
Orchestrates complete poems with multiple phrases:
- **Overall Statistics**: Combines analysis across all phrases
- **Encoded Output**: Generates comprehensive word power mappings
- **Visual Formatting**: Creates indented output with phrase depth visualization

### Enhanced Features (Large Corpus Version)
- **LargePhraseDatabase**: Analyzes thousands of reference phrases with power calculations
- **Power-Based Alternatives**: Finds words with similar contextual influence using configurable count
- **User-Controlled Substitution**: Configurable aggressiveness from 0% to 100% replacement
- **Interactive Parameter Tuning**: Real-time adjustment of all generation parameters
- **Comprehensive Output**: Settings tracking in file headers and display summaries

## 📊 Example Output

### Original Poem
```
Taivos is chaos incarnate
  Suora brings order to all
    Balance is found between extremes
```

### Analysis
```
Word Powers:
Taivos: (Gematria=770, Frequency=1, Magnitude=1.0)
chaos: (Gematria=172, Frequency=1, Magnitude=0.223)
incarnate: (Gematria=409, Frequency=1, Magnitude=0.531)
```

### Alternative Versions
```
Alternative Version
Settings: Gematria: 1.6x, POS: 1.0x, Substitution: 0.8 (80%), Alternatives: 5

Original:
Balance eternal, their fates aligned

Alternative:
alkali urheka isis fates aligned
```

## 🔮 Gematria System

### Scandinavian Extended Values
```
A=1   B=2   C=3   D=4   E=5   F=6   G=7   H=8   I=9   J=10
K=20  L=30  M=40  N=50  O=60  P=70  Q=80  R=90  S=100 T=200
U=300 V=400 W=500 X=600 Y=700 Z=800 Å=900 Ä=1000 Ö=1100
```

### Power Calculations
- **Frequency**: How often a word appears in the poem
- **Magnitude**: (Gematria × Frequency) ÷ Maximum_Gematria_in_Phrase  
- **Word Power**: Frequency × Magnitude

## 📝 Input Files

### Supported Formats
- **Plain text files**: One phrase per line
- **UTF-8 encoding**: Supports international characters
- **Flexible length**: From single phrases to extensive collections

### Sample Files Included
- `phrases.txt`: Collection of mystical/philosophical phrases
- `large_phrases.txt`: Large corpus for contextual analysis (10,000+ phrases)
- `saima_harmaja.txt`: Finnish literary work
- `kataTonya.txt`: Finnish poetry input
- `book_of_lies.txt`: Esoteric text corpus
- `kalevala_whole.txt`: Finnish National Epic Poem
- `kalevala9s.txt`: Ninth Poem of Kalevala (Birth of Iron)



### Generated Output Files
Each generation creates a timestamped file with complete analysis:

```
Enhanced Poem with 3 phrases and 15 words.
Gematria influence: 1.8x
POS influence: 1.2x  
Power substitution: 0.7 (70% aggressiveness)
Word alternative count: 8
Database: 10000 reference phrases analyzed.
Timestamp: T3SNA8
Last Phrase: Balance eternal, their fates aligned

Generated Phrases:
Balance eternal, their fates aligned
[Additional phrases...]
```

## 🎯 Use Cases

### Creative Writing
- Generate alternative phrasings for existing poems
- Explore numerical relationships in text
- Create variations with similar "energy" or power

### Mystical/Esoteric Analysis
- Calculate gematria values for spiritual texts
- Find words with similar numerical significance
- Analyze the "power" distribution in sacred writings

### Linguistic Research
- Study word frequency patterns in different contexts
- Explore alternative word relationships
- Analyze corpus-based semantic similarities

## 🛠️ Technical Details

### Dependencies
- Python 3.6+
- Standard library only (no external packages required)
- UTF-8 terminal support for colored output and Unicode animations
- Large corpus file (`large_phrases.txt`) for enhanced features

### Performance
- Handles up to 10,000+ reference phrases efficiently
- Memory-optimized for large corpus analysis with progress animations
- Configurable parameters for performance and quality tuning
- Iteration-based loading animations prevent interface freezing

### Output Formats
- Console display with ANSI colored formatting and progress bars
- Timestamped text files with complete parameter tracking
- Base36-encoded filenames for unique identification
- Comprehensive settings documentation in file headers

### Animation Styles
Available progress animation styles:
- **Spinner**: Traditional rotating character |/-\
- **Dots**: Unicode Braille pattern animation ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏
- **Bars**: Progress bar visualization ▱▱▱ → ▰▰▰  
- **Pulse**: Circular fill animation ●◐◑◒◓◔◕○
- **Arrow**: Directional movement indicators

## 📋 Command Line Options

```bash
# Enhanced interactive mode with full parameter control
python3 genpoem_with_large_phrases.py

# Enhanced example/demo mode
python3 genpoem_with_large_phrases.py --example
```

## ⚙️ Configuration Parameters

### Interactive Parameter Setting
The enhanced version prompts for four configurable parameters:

```
Current gematria influence: 1.6x
Controls letter value weight in calculations. 1.0=normal, >1.0=enhanced, <1.0=reduced
Enter gematria influence multiplier (default: 1.6): 

Current POS influence: 1.0x  
Controls word class weight (noun/verb/etc). 1.0=normal, >1.0=amplified, <1.0=reduced
Enter POS influence multiplier (default: 1.0):

Current power substitution: 0.5 (50% aggressiveness)
Controls word replacement aggressiveness. 0.0=none, 0.5=moderate, 1.0=maximum  
Enter power substitution aggressiveness (default: 0.5):

Current word alternative count: 5
Controls how many alternative words are generated for each word. Higher=more variety
Enter word alternative count (default: 5):
```

### Programmatic Parameter Control
```python
from genpoem_with_large_phrases import *

# Adjust calculation weights
set_gematria_influence(2.0)    # Enhance gematria significance
set_pos_influence(1.5)         # Amplify part-of-speech differences  
set_power_substitution(0.8)    # High replacement aggressiveness
set_word_alternative_count(10) # Generate more alternatives per word
```

## 🎭 The Philosophy: Taivos and Suora

The algorithms embody the philosophical duality of **Taivos** (Chaos) and **Suora** (Order):

- **Taivos** represents the creative chaos of word alternatives and random variations
- **Suora** represents the ordered structure of gematria calculations and systematic analysis
- Together they create poems that balance mathematical precision with creative transformation

## 🔍 Example Session

```bash
$ python3 genpoem_with_large_phrases.py
Enhanced Poem Generator with Large Phrase Database
==================================================
Current gematria influence: 1.6x
Controls letter value weight in calculations. 1.0=normal, >1.0=enhanced, <1.0=reduced

Enter gematria influence multiplier (default: 1.6): 1.8
Gematria influence updated to 1.8x (enhanced)

Current POS influence: 1.0x
Controls word class weight (noun/verb/etc). 1.0=normal, >1.0=amplified, <1.0=reduced

Enter POS influence multiplier (default: 1.0): 1.2  
POS influence updated to 1.2x (amplified)

Current power substitution: 0.5 (50% aggressiveness)
Controls word replacement aggressiveness. 0.0=none, 0.5=moderate, 1.0=maximum

Enter power substitution aggressiveness (default: 0.5): 0.7
Power substitution updated to 0.7 (high)

Current word alternative count: 5
Controls how many alternative words are generated for each word. Higher=more variety

Enter word alternative count (default: 5): 8
Word alternative count updated to 8 (high variety)

Available text files in current directory:
  1. large_phrases.txt
  2. phrases.txt
  [...]

Enter the large phrase database filename (default: large_phrases.txt): 

Reading large_phrases.txt ✓  
Loaded 10000 meaningful phrases from large_phrases.txt

Analyzing phrase contexts and word powers ✓  
Calculating word alternatives based on power similarity ✓  

Enter the phrase filename (default: phrases.txt): 

Generated Enhanced Poem:
Enhanced Poem with 3 phrases and 15 words.
Database: 10000 reference phrases loaded.

[Detailed analysis follows...]

Generate alternative version using word alternatives? (y/N): y

Generating alternative version ✓  

Alternative Version
Settings: Gematria: 1.8x, POS: 1.2x, Substitution: 0.7 (70%), Alternatives: 8

[Alternative poem with 70% word replacement follows...]
```

## 📄 License

Generated by **Goddess Taivos** and **General Suora**

This project explores the intersection of numerology, linguistics, and creative generation through the lens of gematria traditions.

---

*"From pain becomes pleasure To suspension is found at order While lasts chaos becomes system At the end a force made by bending is The Foundation"*