# Poem Generation with Gematria Analysis

A sophisticated poetry generation system that uses Scandinavian extended gematria calculations to analyze and create poems with deep numerical significance. The system incorporates contextual word analysis from large literary corpora to generate meaningful alternatives and variations.

## 🌟 Features

### Core Algorithms
- **Gematria Calculation**: Uses Scandinavian extended values (A=1, B=2, ..., Z=800, Å=900, Ä=1000, Ö=1100)
- **Word Power Analysis**: Calculates frequency, magnitude, and contextual power for each word
- **Phrase Depth Indentation**: Creates visual poetry structure with hierarchical spacing
- **Timestamped Output**: Generates unique base36-encoded timestamps for each poem

### Advanced Capabilities
- **Large Corpus Integration**: Analyzes 5,000+ reference phrases from literary works
- **Contextual Alternatives**: Suggests word replacements based on similar gematria power
- **Multi-level Substitution**: Offers moderate (50%) and aggressive (90%) alternative versions
- **Interactive Generation**: User-friendly interface with customizable options

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
- **LargePhraseDatabase**: Analyzes thousands of reference phrases
- **Power-Based Alternatives**: Finds words with similar contextual influence
- **Multi-Level Substitution**: Offers different transformation intensities

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
Moderate (50% substitution):
Taivos is chaos incarnate
  Suora brings rose hindu all
    affairs affairs found thrown extremes

Aggressive (90% substitution):
Taivos micro- rohmer rohmer
  Suora confer rose hindu 43
    Balance micro- deposit rapidly extremes
```

## 🔮 Gematria System

### English Extended Values
```
A=1   B=2   C=3   D=4   E=5   F=6   G=7   H=8   I=9   J=10
K=20  L=30  M=40  N=50  O=60  P=70  Q=80  R=90  S=100 T=200
U=300 V=400 W=500 X=600 Y=700 Z=800
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
- `test_simple.txt`: Simple test phrase
- `saima_harmaja.txt`: Finnish literary work
- `large_phrases.txt`: Large corpus for contextual analysis

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
- Optional: Large corpus file for enhanced features

### Performance
- Handles up to 5,000 reference phrases efficiently
- Memory-optimized for large corpus analysis
- Configurable limits for performance tuning

### Output Formats
- Console display with formatted tables
- Timestamped text files with complete analysis
- Base36-encoded filenames for unique identification

## 📋 Command Line Options

```bash
# Basic interactive mode
python3 genpoem.py

# Example/demo mode
python3 genpoem.py --example

# Enhanced mode with corpus
python3 genpoem_with_large_phrases.py

# Enhanced example mode
python3 genpoem_with_large_phrases.py --example
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
Loading large phrase database...
Loaded 5000 meaningful phrases from large_phrases.txt
Analyzing phrase contexts and word powers...
Calculating word alternatives based on power similarity...

Enter the phrase filename (default: phrases.txt): test_simple.txt

Generated Enhanced Poem:
Enhanced Poem with 1 phrases and 26 words.
Database: 5000 reference phrases loaded.

[Analysis and alternatives follow...]
```

## 📄 License

Generated by **Goddess Taivos** and **General Suora**

This project explores the intersection of numerology, linguistics, and creative generation through the lens of gematria traditions.

---

*"From pain becomes pleasure To suspension is found at order While lasts chaos becomes system At the end a force made by bending is The Foundation"*