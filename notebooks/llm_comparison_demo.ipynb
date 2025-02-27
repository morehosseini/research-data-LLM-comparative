{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LLM Research Data Comparative Analysis Demo\n",
    "\n",
    "This notebook demonstrates how to use the tools in this repository to analyze research data with Large Language Models and visualize the results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "\n",
    "# Import our custom modules\n",
    "import sys\n",
    "sys.path.append('../code/')\n",
    "from visualize import create_comparative_analysis, visualize_publication_trends"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Loading Scopus Data\n",
    "\n",
    "First, let's load the Scopus data file and explore its structure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load the data\n",
    "try:\n",
    "    df = pd.read_csv('../data/scopus_27_Feb.csv')\n",
    "    print(f\"Loaded {len(df)} records from Scopus\")\n",
    "    \n",
    "    # Display the first few rows\n",
    "    display(df.head())\n",
    "    \n",
    "    # Display basic information about the dataset\n",
    "    print(\"\\nDataset information:\")\n",
    "    print(f\"Number of publications: {len(df)}\")\n",
    "    print(f\"Date range: {df['Year'].min()} - {df['Year'].max()}\")\n",
    "    \n",
    "    # Display column information\n",
    "    print(\"\\nColumns in the dataset:\")\n",
    "    for col in df.columns:\n",
    "        print(f\"- {col}\")\n",
    "except Exception as e:\n",
    "    print(f\"Error: {e}\")\n",
    "    print(\"Using sample data for demonstration...\")\n",
    "    # Create sample data for demonstration\n",
    "    years = range(2017, 2025)\n",
    "    publications = [5, 12, 28, 45, 98, 187, 342, 421]\n",
    "    df = pd.DataFrame({'Year': years, 'Count': publications})\n",
    "    display(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Analyzing Publication Trends\n",
    "\n",
    "Let's visualize the publication trends over time to identify patterns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Set plotting style\n",
    "sns.set(style=\"whitegrid\")\n",
    "plt.rcParams['figure.figsize'] = (12, 8)\n",
    "\n",
    "# Plot publication trends by year\n",
    "plt.figure(figsize=(12, 6))\n",
    "try:\n",
    "    year_counts = df['Year'].value_counts().sort_index()\n",
    "    ax = year_counts.plot(kind='bar', color='skyblue')\n",
    "    plt.title('Number of LLM Publications by Year', fontsize=16)\n",
    "    plt.xlabel('Year', fontsize=14)\n",
    "    plt.ylabel('Number of Publications', fontsize=14)\n",
    "    plt.xticks(rotation=45)\n",
    "except Exception as e:\n",
    "    # Use sample data if the real data doesn't have the expected structure\n",
    "    print(f\"Using sample data: {e}\")\n",
    "    sample_df = pd.DataFrame({\n",
    "        'Year': range(2017, 2025),\n",
    "        'Count': [5, 12, 28, 45, 98, 187, 342, 421]\n",
    "    })\n",
    "    ax = sample_df.plot(x='Year', y='Count', kind='bar', color='skyblue')\n",
    "    plt.title('Sample: Number of LLM Publications by Year', fontsize=16)\n",
    "    plt.xlabel('Year', fontsize=14)\n",
    "    plt.ylabel('Number of Publications', fontsize=14)\n",
    "    plt.xticks(rotation=45)\n",
    "    \n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. LLM Comparative Analysis\n",
    "\n",
    "Now, let's demonstrate a comparative analysis of different LLMs on research data tasks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Sample performance data for different LLMs\n",
    "models = ['GPT-4', 'Claude-3', 'Llama-3', 'Gemini']\n",
    "metrics = ['MMLU', 'HumanEval', 'TruthfulQA', 'GSM8K']\n",
    "\n",
    "# Create a comparative analysis\n",
    "comparison_df = create_comparative_analysis(models=models, metrics=metrics)\n",
    "\n",
    "# Display the performance data\n",
    "display(comparison_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Analyzing Citation Impact\n",
    "\n",
    "Let's analyze the citation impact of publications in our dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create sample citation data if real data not available\n",
    "try:\n",
    "    if 'Cited by' in df.columns:\n",
    "        citation_data = df['Cited by'].dropna()\n",
    "    else:\n",
    "        raise ValueError(\"'Cited by' column not found\")\n",
    "except Exception as e:\n",
    "    print(f\"Using synthetic citation data: {e}\")\n",
    "    # Generate synthetic citation data with a skewed distribution\n",
    "    np.random.seed(42)\n",
    "    citation_data = np.random.exponential(scale=10, size=500)\n",
    "    citation_data = citation_data.astype(int)\n",
    "    citation_data = citation_data[citation_data < 100]  # Cap at 100 citations\n",
    "\n",
    "# Plot citation distribution\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(citation_data, bins=30, kde=True, color='coral')\n",
    "plt.title('Distribution of Citation Counts', fontsize=16)\n",
    "plt.xlabel('Number of Citations', fontsize=14)\n",
    "plt.ylabel('Frequency', fontsize=14)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Display summary statistics\n",
    "citation_stats = pd.Series(citation_data).describe()\n",
    "print(\"Citation Statistics:\")\n",
    "print(citation_stats)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Topic Analysis\n",
    "\n",
    "Let's identify the common topics in the publications using simple frequency analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Sample topics related to construction and AI\n",
    "topics = [\n",
    "    'Building Information Modeling', 'Digital Twin', 'Machine Learning',\n",
    "    'Natural Language Processing', 'Construction Management', 'Automation',\n",
    "    'Deep Learning', 'Smart Buildings', 'Construction Safety', 'Robotics',\n",
    "    'IoT', 'Knowledge Management', 'Computer Vision', 'Sustainable Construction',\n",
    "    'Augmented Reality', 'Virtual Reality', 'Data Analytics'\n",
    "]\n",
    "\n",
    "# Generate sample frequency data\n",
    "np.random.seed(42)\n",
    "frequencies = np.random.randint(10, 100, size=len(topics))\n",
    "topic_df = pd.DataFrame({'Topic': topics, 'Frequency': frequencies})\n",
    "topic_df = topic_df.sort_values('Frequency', ascending=False)\n",
    "\n",
    "# Plot topic frequencies\n",
    "plt.figure(figsize=(14, 8))\n",
    "sns.barplot(x='Frequency', y='Topic', data=topic_df, palette='viridis')\n",
    "plt.title('Frequency of Topics in Construction AI Research', fontsize=16)\n",
    "plt.xlabel('Number of Publications', fontsize=14)\n",
    "plt.ylabel('Topic', fontsize=14)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Conclusion\n",
    "\n",
    "In this notebook, we've demonstrated:\n",
    "- Loading and exploring Scopus research data\n",
    "- Visualizing publication trends\n",
    "- Comparing different LLMs on research tasks\n",
    "- Analyzing citation impact\n",
    "- Examining topic frequency\n",
    "\n",
    "These tools can be used to conduct in-depth analyses of research in the construction technology domain and evaluate how different LLMs perform on related tasks."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
