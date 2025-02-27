# LLM Research Data Comparative Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Research](https://img.shields.io/badge/Research-NLP-orange)

> A comprehensive framework for comparative analysis of research data using Large Language Models in the construction technology domain.

## 📊 Project Overview

This repository contains tools and datasets for conducting comparative analyses of research data using different Large Language Models. The project focuses on analyzing publications related to AI and digitalized construction while providing visualization tools to interpret the results.

![Sample Comparison](images/llm_comparison_radar.png)

## 🌟 Features

- **Data Collection**: Scripts for gathering research data from Scopus and other academic databases
- **Data Processing**: Tools to clean and transform research data for LLM processing
- **LLM Integration**: Interfaces to multiple LLMs for comparative analysis
- **Visualization**: Various visualizations of research trends and LLM performance comparisons
- **Evaluation Metrics**: Standard metrics for evaluating LLM performance on research data tasks

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/research-data-LLM-comparative.git
cd research-data-LLM-comparative
```

2. Run the visualization script:
```bash
python code/visualize.py
```

3. Process your own Scopus data:
```bash
python code/process_scopus.py data/your_scopus_data.csv
```

## 📈 Example Results

The repository includes visualization capabilities for analyzing research trends:

### Publication Trends
![Publication Trends](images/publications_by_year.png)

### Research Areas
![Research Areas](images/top_research_areas.png)

## 📁 Repository Structure

```
├── code/                 # Python scripts and notebooks
│   ├── visualize.py      # Visualization tools
│   ├── process_scopus.py # Data processing utilities
│   └── llm_compare.py    # LLM comparison framework
├── data/                 # Data files
│   └── scopus_27_Feb.csv # Scopus dataset (example)
├── images/               # Generated visualizations
├── notebooks/            # Jupyter notebooks with examples
└── README.md             # This file
```

## 🔍 Use Cases

- Trend analysis in construction technology and AI research
- Comparative performance evaluation of different LLMs on research data
- Identification of research gaps and opportunities
- Visualization of publication trends and impact

## 🔗 Related Work

This repository extends upon research in AI applications in construction technology, particularly focusing on:

- Natural Language Processing in construction engineering
- Digital twin technology and BIM
- Automated literature reviews using AI

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact

Dr. M. Reza Hosseini - reza.hosseini@unimelb.edu.au

---

*Faculty of Architecture, Building and Planning, The University of Melbourne*
