import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_scopus_data(filepath='data/scopus_27_Feb.csv'):
    """
    Load and process Scopus data
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from Scopus")
    return df

def visualize_publication_trends(df, save_path='images'):
    """
    Create visualizations for publication trends in LLM research
    """
    # Create save directory if it doesn't exist
    Path(save_path).mkdir(exist_ok=True)
    
    # Set styling for plots
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # Example 1: Publications per year
    plt.figure(figsize=(12, 6))
    year_counts = df['Year'].value_counts().sort_index()
    ax = year_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of LLM Publications by Year', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Publications', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/publications_by_year.png', dpi=300)
    
    # Example 2: Top research areas (assuming there's a Subject column)
    if 'Subject' in df.columns:
        plt.figure(figsize=(12, 8))
        subjects = df['Subject'].str.split(';').explode().str.strip()
        top_subjects = subjects.value_counts().head(10)
        ax = top_subjects.plot(kind='barh', color='lightgreen')
        plt.title('Top 10 Research Areas in LLM Studies', fontsize=16)
        plt.xlabel('Number of Publications', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_path}/top_research_areas.png', dpi=300)
    
    # Example 3: Citation impact (if citation data is available)
    if 'Cited by' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Cited by'].dropna(), bins=30, kde=True, color='coral')
        plt.title('Distribution of Citation Counts', fontsize=16)
        plt.xlabel('Number of Citations', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_path}/citation_distribution.png', dpi=300)
    
    print(f"Visualizations saved to {save_path}/ directory")

def create_comparative_analysis(models=['GPT-4', 'Claude-3', 'Llama-3', 'Gemini'], 
                              metrics=['MMLU', 'HumanEval', 'TruthfulQA', 'GSM8K']):
    """
    Create a sample comparative analysis visualization of LLM performance
    """
    # Sample data - replace with your actual benchmarks
    np.random.seed(42)
    data = {}
    
    for metric in metrics:
        # Generate random scores between 60-95 for demonstration
        scores = np.random.uniform(60, 95, size=len(models))
        if metric == 'TruthfulQA':  # Make scores more realistic
            scores = np.random.uniform(40, 85, size=len(models))
        data[metric] = scores
    
    # Create DataFrame
    df = pd.DataFrame(data, index=models)
    
    # Plot radar chart
    plt.figure(figsize=(10, 8))
    
    # Plot radar chart using matplotlib
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, model in enumerate(models):
        values = df.loc[model].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_ylim(0, 100)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('LLM Performance Comparison', fontsize=16, pad=20)
    plt.tight_layout()
    
    Path('images').mkdir(exist_ok=True)
    plt.savefig('images/llm_comparison_radar.png', dpi=300, bbox_inches='tight')
    print("LLM comparison visualization saved to images/llm_comparison_radar.png")
    
    return df

if __name__ == "__main__":
    # Create the comparative visualization
    model_comparison = create_comparative_analysis()
    print(model_comparison)
    
    # Optional: Load and visualize Scopus data if available
    try:
        scopus_df = load_scopus_data()
        visualize_publication_trends(scopus_df)
    except Exception as e:
        print(f"Note: Scopus data visualization skipped - {e}")
        print("Run with Scopus data file available to generate publication trend visualizations")
