"""
Generate and save analysis results automatically.
This module creates comprehensive result files from analysis output.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import os


def save_analysis_results(results: Dict[str, Any], output_dir: str = "results"):
    """
    Save comprehensive analysis results to multiple file formats.
    
    Args:
        results: Dictionary containing analysis results
        output_dir: Directory to save results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    json_file = os.path.join(output_dir, f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save CSV summary
    csv_file = os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_data = []
    
    for ticker, data in results.get('stocks', {}).items():
        summary_data.append({
            'Stock': ticker,
            'Current_Price': data.get('current_price', 0),
            'Sentiment_Score': data.get('sentiment_score', 0),
            'Trading_Return': data.get('trading_return', 0),
            'Sharpe_Ratio': data.get('sharpe_ratio', 0),
            'Investment_Signal': data.get('signal', 'HOLD')
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False)
    
    # Save detailed report
    report_file = os.path.join(output_dir, f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    generate_markdown_report(results, report_file)
    
    print(f"✅ Results saved to {output_dir}/")
    print(f"   - JSON: {json_file}")
    print(f"   - CSV: {csv_file}")
    print(f"   - Report: {report_file}")


def generate_markdown_report(results: Dict[str, Any], filename: str):
    """Generate a detailed markdown report."""
    with open(filename, 'w') as f:
        f.write("# Stock Market Analysis Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Analysis Date**: {results.get('analysis_date', 'N/A')}\n")
        f.write(f"- **Data Source**: {results.get('data_source', 'N/A')}\n")
        f.write(f"- **Stocks Analyzed**: {len(results.get('stocks', {}))}\n\n")
        
        f.write("## Individual Results\n\n")
        for ticker, data in results.get('stocks', {}).items():
            f.write(f"### {ticker}\n")
            f.write(f"- **Price**: ${data.get('current_price', 0):.2f}\n")
            f.write(f"- **Sentiment**: {data.get('sentiment_score', 0):.3f}\n")
            f.write(f"- **Return**: {data.get('trading_return', 0):.2f}%\n")
            f.write(f"- **Signal**: {data.get('signal', 'HOLD')}\n\n")


def create_sample_results():
    """Create sample results based on our demo output."""
    sample_results = {
        'analysis_date': '2025-07-24T14:22:00Z',
        'data_source': 'yahoo_finance',
        'system_version': '1.0.0',
        'stocks': {
            'AAPL': {
                'current_price': 214.15,
                'sentiment_score': 1.000,
                'trading_return': 37.17,
                'sharpe_ratio': 1.664,
                'signal': 'BUY'
            },
            'GOOGL': {
                'current_price': 190.23,
                'sentiment_score': 1.000,
                'trading_return': 28.66,
                'sharpe_ratio': 1.500,
                'signal': 'BUY'
            },
            'MSFT': {
                'current_price': 505.87,
                'sentiment_score': 1.000,
                'trading_return': 44.80,
                'sharpe_ratio': 1.577,
                'signal': 'BUY'
            }
        },
        'portfolio_summary': {
            'average_return': 36.88,
            'average_sharpe': 1.58,
            'best_performer': 'MSFT'
        }
    }
    
    return sample_results


if __name__ == "__main__":
    # Generate sample results based on our demo
    results = create_sample_results()
    save_analysis_results(results)
    print("Sample results generated successfully!")
